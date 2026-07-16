use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

use nsrl_core::{
    SoftmaxNormalization, base2_exp_neg_q15, base2_softmax_nll_millibits, base2_softmax_nll_q47_q32,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};
use nsrl_train::production::{
    ProductionFullTrainConfig, ProductionGradientAlignmentConfig, ProductionGradientProposalLane,
    ProductionModelV1, audit_production_gradient_alignment, decode_bound_token_stream,
    forward_production_model,
};

const CONTEXT_TOKENS: usize = 64;
const WINDOWS_PER_PASSAGE: usize = 2;
const ZERO_FLOOR_Q32: u64 = 32_u64 << 32;
const ZERO_FLOOR_MILLIBITS: u64 = 32_000;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const STATES: [&str; 7] = ["empty", "H", "T", "HH", "HT", "TH", "TT"];
const ACTIONS: [&str; 2] = ["H", "T"];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Move {
    group: usize,
    coordinate: usize,
    delta: i8,
}

#[derive(Clone)]
struct Window {
    context: Vec<u32>,
    target: usize,
}

#[derive(Clone)]
struct PanelRow {
    document: usize,
    family: String,
    source_id: String,
    passage: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Endpoint {
    nll_millibits: u128,
    zero_probability_windows: usize,
}

#[derive(Clone, Debug)]
struct Decision {
    policy: String,
    document: usize,
    family: String,
    source_id: String,
    passage: usize,
    state_before: String,
    action: String,
    certified_upper_q32: Option<i128>,
    exact_contrast_q32: i128,
    state_after: String,
}

#[derive(Clone, Debug)]
struct Trajectory {
    final_state: String,
    accepted: usize,
    head_fires: usize,
    trunk_fires: usize,
    positive_regret_q32: u128,
    signed_regret_q32: i128,
    decisions: Vec<Decision>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let command = args
        .next()
        .ok_or("missing command: fit-actions, calibrate, or evaluate")?;
    let options = parse_options(args)?;
    match command.as_str() {
        "fit-actions" => fit_actions(&options),
        "calibrate" => calibrate_command(&options),
        "evaluate" => evaluate(&options),
        _ => Err(format!("unknown command: {command}").into()),
    }
}

fn parse_options(
    args: impl Iterator<Item = String>,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut values = BTreeMap::new();
    let mut args = args.peekable();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("expected option, got {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        values.insert(key, value);
    }
    Ok(values)
}

fn required(options: &BTreeMap<String, String>, key: &str) -> Result<String, Box<dyn Error>> {
    options
        .get(key)
        .cloned()
        .ok_or_else(|| format!("missing {key}").into())
}

fn parallel_map<T, F>(length: usize, operation: F) -> Result<Vec<T>, Box<dyn Error>>
where
    T: Send,
    F: Fn(usize) -> Result<T, Box<dyn Error>> + Sync,
{
    if length == 0 {
        return Ok(Vec::new());
    }
    let workers = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(length);
    let chunks = thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers {
            let operation = &operation;
            handles.push(scope.spawn(move || {
                (worker..length)
                    .step_by(workers)
                    .map(|index| {
                        operation(index)
                            .map(|value| (index, value))
                            .map_err(|error| error.to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()
            }));
        }
        let mut chunks = Vec::new();
        for handle in handles {
            chunks.push(
                handle
                    .join()
                    .map_err(|_| "parallel evaluator panicked".to_string())??,
            );
        }
        Ok::<_, String>(chunks)
    })
    .map_err(|error| -> Box<dyn Error> { error.into() })?;
    let mut ordered = (0..length).map(|_| None).collect::<Vec<Option<T>>>();
    for (index, value) in chunks.into_iter().flatten() {
        ordered[index] = Some(value);
    }
    ordered
        .into_iter()
        .map(|value| value.ok_or_else(|| "parallel evaluator omitted a result".into()))
        .collect()
}

fn read_model(path: impl AsRef<Path>) -> Result<ProductionModelV1, Box<dyn Error>> {
    Ok(ProductionModelV1::from_bytes(&fs::read(path)?)?)
}

fn write_model(path: impl AsRef<Path>, model: &ProductionModelV1) -> Result<(), Box<dyn Error>> {
    fs::write(path, model.try_to_bytes()?)?;
    Ok(())
}

fn read_tokens(
    path: impl AsRef<Path>,
    model: &ProductionModelV1,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let (tokens, _) = decode_bound_token_stream(
        &fs::read(path)?,
        model.tokenizer_hash,
        model.config.vocab_size,
    )?;
    Ok(tokens)
}

fn document_windows(tokens: &[u32]) -> Vec<Vec<Window>> {
    let mut documents = Vec::new();
    let mut current = Vec::new();
    let mut active = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            current.clear();
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active {
                let windows = if current.len() > CONTEXT_TOKENS {
                    (0..current.len() - CONTEXT_TOKENS)
                        .take(WINDOWS_PER_PASSAGE)
                        .map(|start| Window {
                            context: current[start..start + CONTEXT_TOKENS].to_vec(),
                            target: current[start + CONTEXT_TOKENS] as usize,
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                documents.push(windows);
            }
            current.clear();
            active = false;
        } else if active {
            current.push(token);
        }
    }
    documents
}

fn read_panels(path: impl AsRef<Path>) -> Result<Vec<PanelRow>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next()
        != Some("document\tfamily\tsource_id\tindependence_key\tpassage_ordinal\trole\tsha256")
    {
        return Err("invalid panel TSV header".into());
    }
    let mut rows = Vec::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err("invalid panel TSV row".into());
        }
        rows.push(PanelRow {
            document: fields[0].parse()?,
            family: fields[1].to_string(),
            source_id: fields[2].to_string(),
            passage: fields[4].parse()?,
        });
    }
    for (document, row) in rows.iter().enumerate() {
        if row.document != document {
            return Err("panel TSV document order changed".into());
        }
    }
    Ok(rows)
}

fn mutate(model: &mut ProductionModelV1, movement: &Move) -> Result<(), Box<dyn Error>> {
    macro_rules! add {
        ($field:ident, $type:ty) => {{
            let value = model
                .$field
                .get_mut(movement.coordinate)
                .ok_or("move coordinate out of range")?;
            *value = <$type>::try_from(i64::from(*value) + i64::from(movement.delta))?;
        }};
    }
    match movement.group {
        0 => add!(embeddings, i16),
        1 => add!(attention_rms_weights, i16),
        2 => add!(mlp_rms_weights, i16),
        3 => add!(final_rms_weights, i16),
        4 => add!(q_weights, i8),
        5 => add!(k_weights, i8),
        6 => add!(v_weights, i8),
        7 => add!(o_weights, i8),
        8 => add!(up_weights, i8),
        9 => add!(gate_weights, i8),
        10 => add!(down_weights, i8),
        11 => add!(output_weights, i16),
        12 => add!(output_bias_q8, i32),
        _ => return Err("invalid parameter group".into()),
    }
    Ok(())
}

fn apply_action(
    model: &ProductionModelV1,
    action: &[Move],
) -> Result<ProductionModelV1, Box<dyn Error>> {
    let mut candidate = model.clone();
    for movement in action {
        mutate(&mut candidate, movement)?;
    }
    candidate.validate()?;
    Ok(candidate)
}

fn fitting_training_config() -> ProductionFullTrainConfig {
    ProductionFullTrainConfig {
        context_tokens: CONTEXT_TOKENS,
        max_windows: 8,
        spread_windows: false,
        targets_per_window: 1,
        epochs: 1,
        matrix_learning_rate_shift: 25,
        q_learning_rate_shift: Some(29),
        k_learning_rate_shift: Some(26),
        v_learning_rate_shift: Some(30),
        o_learning_rate_shift: Some(25),
        up_learning_rate_shift: Some(22),
        gate_learning_rate_shift: Some(23),
        down_learning_rate_shift: Some(25),
        vector_learning_rate_shift: 23,
        final_rms_learning_rate_shift: None,
        embedding_learning_rate_shift: 17,
        embedding_learning_rate_boost_shift: 0,
        output_learning_rate_shift: 33,
        output_backward_shift: Some(8),
        probability_gradient_fractional_bits: 23,
        probability_normalization: SoftmaxNormalization::Q47Newton1,
        batch_windows: 4,
        max_optimizer_steps: usize::MAX,
        evaluation_windows: usize::MAX,
    }
}

fn derive_action_pool(
    model: &ProductionModelV1,
    tokens: &[u32],
) -> Result<BTreeMap<String, Vec<Move>>, Box<dyn Error>> {
    let config = ProductionGradientAlignmentConfig {
        proposal_windows: 4,
        transfer_windows: 4,
        documents_per_surface: 4,
        rescue_stratified_sampling: true,
        include_mass_corrected_no_rescue: true,
        include_systematic_fixed_mass: false,
        coordinates_per_group: 1,
        sample_seed: 43 ^ model.model_hash(),
    };
    let trace = audit_production_gradient_alignment(
        model,
        tokens,
        token_hash(tokens),
        fitting_training_config(),
        config,
    )?;
    let lane_priority = [
        ProductionGradientProposalLane::MassCorrectedNormalized,
        ProductionGradientProposalLane::NormalizedRescued,
        ProductionGradientProposalLane::ReciprocalFreeRescued,
        ProductionGradientProposalLane::ReciprocalFreeLateRhu,
        ProductionGradientProposalLane::ReciprocalFreeLateStochastic,
        ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue,
    ];
    let mut head = Vec::new();
    let mut trunk = Vec::new();
    for sample in &trace.samples {
        let delta = lane_priority
            .iter()
            .find_map(|lane| {
                sample
                    .lanes
                    .iter()
                    .find(|candidate| candidate.lane == *lane)
                    .map(|candidate| candidate.predicted_parameter_delta)
                    .filter(|value| matches!(value, -1 | 1))
            })
            .or_else(|| {
                matches!(sample.proposal.better_neighbor_delta, -1 | 1)
                    .then_some(sample.proposal.better_neighbor_delta)
            })
            .or_else(|| {
                matches!(sample.transfer.better_neighbor_delta, -1 | 1)
                    .then_some(sample.transfer.better_neighbor_delta)
            })
            .unwrap_or(sample.random_control_delta);
        if !matches!(delta, -1 | 1) {
            return Err("fitting trace did not expose a nonzero unit move".into());
        }
        let movement = Move {
            group: sample.group_index,
            coordinate: sample.coordinate,
            delta,
        };
        let block = if movement.group >= 11 {
            &mut head
        } else {
            &mut trunk
        };
        if !block.iter().any(|candidate: &Move| {
            candidate.group == movement.group && candidate.coordinate == movement.coordinate
        }) {
            block.push(movement);
        }
    }
    let pool: BTreeMap<String, Vec<Move>> =
        BTreeMap::from([("H".to_string(), head), ("T".to_string(), trunk)]);
    if pool.values().any(|moves| moves.len() < 2)
        || pool
            .values()
            .flatten()
            .any(|movement| !matches!(movement.delta, -1 | 1))
        || pool["H"].iter().any(|movement| movement.group < 11)
        || pool["T"].iter().any(|movement| movement.group >= 11)
    {
        return Err("fitting-derived physical action pool changed".into());
    }
    Ok(pool)
}

fn action_selection_hash(state_model_hash: u64, action: &str, movement: &Move, domain: u8) -> u64 {
    state_model_hash
        .to_le_bytes()
        .into_iter()
        .chain(action.bytes())
        .chain(movement.group.to_le_bytes())
        .chain(movement.coordinate.to_le_bytes())
        .chain(movement.delta.to_le_bytes())
        .chain([domain])
        .fold(FNV_OFFSET, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
}

fn derive_actions(
    model: &ProductionModelV1,
    pool: &BTreeMap<String, Vec<Move>>,
) -> Result<BTreeMap<String, Vec<Move>>, Box<dyn Error>> {
    let state_hash = model.model_hash();
    let mut actions = BTreeMap::new();
    for action in ACTIONS {
        let mut candidates = pool[action].clone();
        candidates.sort_unstable_by_key(|movement| {
            action_selection_hash(state_hash, action, movement, 0)
        });
        let selected = candidates
            .into_iter()
            .take(2)
            .map(|mut movement| {
                if action_selection_hash(state_hash, action, &movement, 1) & 1 == 1 {
                    movement.delta = -movement.delta;
                }
                movement
            })
            .collect::<Vec<_>>();
        actions.insert(action.to_string(), selected);
    }
    if actions.values().any(|moves| moves.len() != 2) {
        return Err("state-specific action selection changed".into());
    }
    Ok(actions)
}

fn token_hash(tokens: &[u32]) -> u64 {
    tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .fold(FNV_OFFSET, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
}

fn function_hash(
    model: &ProductionModelV1,
    documents: &[Vec<Window>],
) -> Result<u64, Box<dyn Error>> {
    let windows = documents.iter().flatten().collect::<Vec<_>>();
    let logits = parallel_map(windows.len(), |index| {
        Ok(forward_production_model(model, &windows[index].context)?.logits_q8)
    })?;
    let mut hash = FNV_OFFSET;
    for output in logits {
        for value in output {
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
            }
        }
    }
    Ok(hash)
}

fn q32_document_loss(model: &ProductionModelV1, windows: &[Window]) -> Result<u64, Box<dyn Error>> {
    if windows.len() != WINDOWS_PER_PASSAGE {
        return Err("passage does not expose exactly two windows".into());
    }
    let mut total = 0_u64;
    for window in windows {
        let output = forward_production_model(model, &window.context)?;
        total = total
            .checked_add(
                base2_softmax_nll_q47_q32(&output.logits_q8, window.target, ZERO_FLOOR_Q32)
                    .ok_or("Q32 NLL failed")?,
            )
            .ok_or("Q32 document loss overflow")?;
    }
    Ok(total)
}

fn fit_actions(options: &BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    let model_path = required(options, "--model")?;
    let fitting_tokens_path = required(options, "--fitting-tokens")?;
    let fitting_panels_path = required(options, "--fitting-panels")?;
    let output_directory = PathBuf::from(required(options, "--out-dir")?);
    let trace_path = PathBuf::from(required(options, "--trace")?);
    fs::create_dir_all(&output_directory)?;

    let base = read_model(model_path)?;
    let fitting_tokens = read_tokens(fitting_tokens_path, &base)?;
    let fitting_documents = document_windows(&fitting_tokens);
    let fitting_panels = read_panels(fitting_panels_path)?;
    if fitting_documents.len() != 144
        || fitting_panels.len() != 144
        || fitting_documents
            .iter()
            .any(|windows| windows.len() != WINDOWS_PER_PASSAGE)
    {
        return Err("fitting surface must contain 36 panels and 144 passages".into());
    }

    let mut models = BTreeMap::from([("empty".to_string(), base)]);
    let mut actions = BTreeMap::<(String, String), Vec<Move>>::new();
    let action_pool = derive_action_pool(&models["empty"], &fitting_tokens)?;
    for state in ["empty", "H", "T"] {
        let state_model = models
            .get(state)
            .ok_or("state construction order failed")?
            .clone();
        let derived = derive_actions(&state_model, &action_pool)?;
        for action in ACTIONS {
            actions.insert(
                (state.to_string(), action.to_string()),
                derived[action].clone(),
            );
        }
        if state == "empty" {
            models.insert("H".to_string(), apply_action(&state_model, &derived["H"])?);
            models.insert("T".to_string(), apply_action(&state_model, &derived["T"])?);
        } else if state == "H" {
            models.insert("HH".to_string(), apply_action(&state_model, &derived["H"])?);
            models.insert("HT".to_string(), apply_action(&state_model, &derived["T"])?);
        } else {
            models.insert("TH".to_string(), apply_action(&state_model, &derived["H"])?);
            models.insert("TT".to_string(), apply_action(&state_model, &derived["T"])?);
        }
    }
    for state in ["HH", "HT", "TH", "TT"] {
        let derived = derive_actions(
            models.get(state).ok_or("depth-two state missing")?,
            &action_pool,
        )?;
        for action in ACTIONS {
            actions.insert(
                (state.to_string(), action.to_string()),
                derived[action].clone(),
            );
        }
    }
    if models.len() != STATES.len() || actions.len() != STATES.len() * ACTIONS.len() {
        return Err("reachable state/action manifest is incomplete".into());
    }

    let ht_model = models.get("HT").ok_or("HT missing")?;
    let th_model = models.get("TH").ok_or("TH missing")?;
    let ht_function_hash = function_hash(ht_model, &fitting_documents)?;
    let th_function_hash = function_hash(th_model, &fitting_documents)?;
    if ht_model.model_hash() == th_model.model_hash() || ht_function_hash == th_function_hash {
        return Err(
            "noncommutativity gate falsified: HT and TH must differ in model and function hash"
                .into(),
        );
    }

    for (state, model) in &models {
        write_model(
            output_directory.join(format!("state-{state}.nsrlpm")),
            model,
        )?;
    }
    let mut action_tsv = String::from("state\taction\twrite\tgroup\tcoordinate\tdelta\n");
    for state in STATES {
        for action in ACTIONS {
            for (write, movement) in actions[&(state.to_string(), action.to_string())]
                .iter()
                .enumerate()
            {
                action_tsv.push_str(&format!(
                    "{state}\t{action}\t{write}\t{}\t{}\t{}\n",
                    movement.group, movement.coordinate, movement.delta
                ));
            }
        }
    }
    fs::write(output_directory.join("actions.tsv"), action_tsv)?;

    let mut fitting_cube =
        String::from("document\tfamily\tsource_id\tpassage\tstate\taction\tcontrast_q32\n");
    let mut by_key = BTreeMap::<(String, String, String), Vec<i128>>::new();
    for state in STATES {
        let state_model = &models[state];
        let base_losses = parallel_map(fitting_documents.len(), |document| {
            q32_document_loss(state_model, &fitting_documents[document])
        })?;
        for action in ACTIONS {
            let candidate = apply_action(
                state_model,
                &actions[&(state.to_string(), action.to_string())],
            )?;
            let contrasts = parallel_map(fitting_documents.len(), |document| {
                Ok(
                    i128::from(q32_document_loss(&candidate, &fitting_documents[document])?)
                        - i128::from(base_losses[document]),
                )
            })?;
            for (document, contrast) in contrasts.into_iter().enumerate() {
                let panel = &fitting_panels[document];
                fitting_cube.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{state}\t{action}\t{contrast}\n",
                    document, panel.family, panel.source_id, panel.passage
                ));
                by_key
                    .entry((panel.family.clone(), state.to_string(), action.to_string()))
                    .or_default()
                    .push(contrast);
            }
        }
    }
    fs::write(output_directory.join("fitting-cube.tsv"), fitting_cube)?;
    let mut predictor =
        String::from("family\tstate\taction\tlower_median_contrast_q32\tobservations\n");
    for ((family, state, action), mut values) in by_key {
        values.sort_unstable();
        if values.len() != 48 {
            return Err("predictor must use 12 fitting panels and four passages per family".into());
        }
        let median = values[(values.len() - 1) / 2];
        predictor.push_str(&format!(
            "{family}\t{state}\t{action}\t{median}\t{}\n",
            values.len()
        ));
    }
    fs::write(output_directory.join("predictor.tsv"), predictor)?;

    let trace = format!(
        concat!(
            "{{\"schema\":\"nsrl.adaptive_composition_action_manifest.v1\",",
            "\"analysis_role\":\"fitting_only_before_calibration\",",
            "\"base_model_hash\":\"0x{:016x}\",",
            "\"state_model_hashes\":{{{} }},",
            "\"noncommutativity\":{{\"ht_model_hash\":\"0x{:016x}\",",
            "\"th_model_hash\":\"0x{:016x}\",",
            "\"ht_function_hash\":\"0x{:016x}\",",
            "\"th_function_hash\":\"0x{:016x}\",\"passed\":true}},",
            "\"actions_per_state\":2,\"writes_per_action\":2,",
            "\"action_derivation\":\"one canonical fitting gradient pool; state-model-hash coordinate and sign selection; frozen lane priority; exact-neighbor then seeded-unit fallback\",",
            "\"predictor\":\"within-family lower median fitting contrast\",",
            "\"calibration_outcomes_read\":false,\"adaptive_outcomes_read\":false,",
            "\"endpoint_outcomes_read\":false}}\n"
        ),
        models["empty"].model_hash(),
        STATES
            .iter()
            .map(|state| format!("\"{state}\":\"0x{:016x}\"", models[*state].model_hash()))
            .collect::<Vec<_>>()
            .join(","),
        ht_model.model_hash(),
        th_model.model_hash(),
        ht_function_hash,
        th_function_hash,
    );
    fs::write(trace_path, trace)?;
    println!(
        "{{\"schema\":\"nsrl.adaptive_composition_fit_actions.v1\",\"states\":7,\"actions\":14,\"noncommutativity_passed\":true}}"
    );
    Ok(())
}

fn read_actions(
    path: impl AsRef<Path>,
) -> Result<BTreeMap<(String, String), Vec<Move>>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("state\taction\twrite\tgroup\tcoordinate\tdelta") {
        return Err("invalid action TSV header".into());
    }
    let mut actions = BTreeMap::<(String, String), Vec<Move>>::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err("invalid action TSV row".into());
        }
        let moves = actions
            .entry((fields[0].to_string(), fields[1].to_string()))
            .or_default();
        if fields[2].parse::<usize>()? != moves.len() {
            return Err("action write order changed".into());
        }
        moves.push(Move {
            group: fields[3].parse()?,
            coordinate: fields[4].parse()?,
            delta: fields[5].parse()?,
        });
    }
    if actions.len() != STATES.len() * ACTIONS.len()
        || actions.values().any(|moves| moves.len() != 2)
    {
        return Err("action manifest shape changed".into());
    }
    Ok(actions)
}

fn read_predictor(
    path: impl AsRef<Path>,
) -> Result<BTreeMap<(String, String, String), i128>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("family\tstate\taction\tlower_median_contrast_q32\tobservations") {
        return Err("invalid predictor TSV header".into());
    }
    let mut predictor = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 5 || fields[4] != "48" {
            return Err("invalid predictor TSV row".into());
        }
        predictor.insert(
            (
                fields[0].to_string(),
                fields[1].to_string(),
                fields[2].to_string(),
            ),
            fields[3].parse()?,
        );
    }
    if predictor.len() != 3 * STATES.len() * ACTIONS.len() {
        return Err("predictor surface is incomplete".into());
    }
    Ok(predictor)
}

fn load_state_models(
    directory: &Path,
) -> Result<BTreeMap<String, ProductionModelV1>, Box<dyn Error>> {
    let mut models = BTreeMap::new();
    for state in STATES {
        models.insert(
            state.to_string(),
            read_model(directory.join(format!("state-{state}.nsrlpm")))?,
        );
    }
    if models["HT"].model_hash() == models["TH"].model_hash() {
        return Err("noncommutativity model-hash gate failed at evaluation".into());
    }
    Ok(models)
}

fn candidate_models(
    models: &BTreeMap<String, ProductionModelV1>,
    actions: &BTreeMap<(String, String), Vec<Move>>,
) -> Result<BTreeMap<(String, String), ProductionModelV1>, Box<dyn Error>> {
    let mut candidates = BTreeMap::new();
    for state in STATES {
        for action in ACTIONS {
            candidates.insert(
                (state.to_string(), action.to_string()),
                apply_action(
                    &models[state],
                    &actions[&(state.to_string(), action.to_string())],
                )?,
            );
        }
    }
    Ok(candidates)
}

fn contrast_q32(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    windows: &[Window],
) -> Result<i128, Box<dyn Error>> {
    Ok(i128::from(q32_document_loss(candidate, windows)?)
        - i128::from(q32_document_loss(source, windows)?))
}

fn calibrate(
    documents: &[Vec<Window>],
    panels: &[PanelRow],
    models: &BTreeMap<String, ProductionModelV1>,
    candidates: &BTreeMap<(String, String), ProductionModelV1>,
    predictor: &BTreeMap<(String, String, String), i128>,
    output_directory: &Path,
) -> Result<BTreeMap<String, i128>, Box<dyn Error>> {
    if documents.len() != 1_428 || panels.len() != 1_428 {
        return Err("calibration must contain 357 source panels and 1428 passages".into());
    }
    let mut cube = String::from(
        "document\tfamily\tsource_id\tpassage\tstate\taction\tcontrast_q32\tpredicted_q32\tresidual_q32\n",
    );
    let mut source_scores = BTreeMap::<(String, String), i128>::new();
    let document_rows = parallel_map(documents.len(), |document| {
        let panel = &panels[document];
        let windows = &documents[document];
        let mut rows = String::new();
        let mut document_score = i128::MIN;
        for state in STATES {
            let source_loss = q32_document_loss(&models[state], windows)?;
            for action in ACTIONS {
                let candidate_loss = q32_document_loss(
                    &candidates[&(state.to_string(), action.to_string())],
                    windows,
                )?;
                let contrast = i128::from(candidate_loss) - i128::from(source_loss);
                let predicted =
                    predictor[&(panel.family.clone(), state.to_string(), action.to_string())];
                let residual = contrast - predicted;
                rows.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{state}\t{action}\t{contrast}\t{predicted}\t{residual}\n",
                    document, panel.family, panel.source_id, panel.passage
                ));
                document_score = document_score.max(residual);
            }
        }
        Ok((rows, document_score))
    })?;
    for (document, (rows, document_score)) in document_rows.into_iter().enumerate() {
        cube.push_str(&rows);
        let panel = &panels[document];
        let score = source_scores
            .entry((panel.family.clone(), panel.source_id.clone()))
            .or_insert(i128::MIN);
        *score = (*score).max(document_score);
    }
    fs::write(output_directory.join("calibration-cube.tsv"), cube)?;
    let mut scores_tsv = String::from("family\tsource_id\tsimultaneous_score_q32\n");
    for ((family, source), score) in &source_scores {
        scores_tsv.push_str(&format!("{family}\t{source}\t{score}\n"));
    }
    fs::write(output_directory.join("calibration-scores.tsv"), scores_tsv)?;
    let mut corrections = BTreeMap::new();
    let mut corrections_tsv = String::from("family\tcorrection_q32\trank\tcalibration_sources\n");
    for family in ["federal_register", "rfc", "science"] {
        let mut scores = source_scores
            .iter()
            .filter_map(|((candidate_family, _), score)| {
                (candidate_family == family).then_some(*score)
            })
            .collect::<Vec<_>>();
        scores.sort_unstable();
        if scores.len() != 119 {
            return Err(format!("{family} must have 119 calibration scores").into());
        }
        let correction = scores[118];
        corrections.insert(family.to_string(), correction);
        corrections_tsv.push_str(&format!("{family}\t{correction}\t119\t119\n"));
    }
    fs::write(output_directory.join("corrections.tsv"), corrections_tsv)?;
    Ok(corrections)
}

fn calibrate_command(options: &BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    let manifest_directory = PathBuf::from(required(options, "--manifest-dir")?);
    let output_directory = PathBuf::from(required(options, "--out-dir")?);
    let trace_path = PathBuf::from(required(options, "--trace")?);
    fs::create_dir_all(&output_directory)?;
    let models = load_state_models(&manifest_directory)?;
    let actions = read_actions(manifest_directory.join("actions.tsv"))?;
    let predictor = read_predictor(manifest_directory.join("predictor.tsv"))?;
    let candidates = candidate_models(&models, &actions)?;
    let calibration_tokens =
        read_tokens(required(options, "--calibration-tokens")?, &models["empty"])?;
    let calibration_documents = document_windows(&calibration_tokens);
    let calibration_panels = read_panels(required(options, "--calibration-panels")?)?;
    let corrections = calibrate(
        &calibration_documents,
        &calibration_panels,
        &models,
        &candidates,
        &predictor,
        &output_directory,
    )?;
    let correction_json = corrections
        .iter()
        .map(|(family, value)| format!("\"{family}\":\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(
        &trace_path,
        format!(
            concat!(
                "{{\"schema\":\"nsrl.adaptive_composition_calibration.v1\",",
                "\"analysis_role\":\"calibration_only_before_adaptive_endpoint\",",
                "\"cube_rows\":19992,\"source_scores\":357,",
                "\"corrections_q32\":{{{}}},\"adaptive_outcomes_read\":false,",
                "\"endpoint_outcomes_read\":false}}\n"
            ),
            correction_json
        ),
    )?;
    println!(
        "{{\"schema\":\"nsrl.adaptive_composition_calibration_execution.v1\",\"trace\":\"{}\"}}",
        trace_path.display()
    );
    Ok(())
}

fn read_corrections(path: impl AsRef<Path>) -> Result<BTreeMap<String, i128>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    if lines.next() != Some("family\tcorrection_q32\trank\tcalibration_sources") {
        return Err("invalid correction TSV header".into());
    }
    let mut corrections = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 || fields[2] != "119" || fields[3] != "119" {
            return Err("invalid correction TSV row".into());
        }
        corrections.insert(fields[0].to_string(), fields[1].parse()?);
    }
    if corrections.len() != 3
        || !["federal_register", "rfc", "science"]
            .iter()
            .all(|family| corrections.contains_key(*family))
    {
        return Err("calibration correction surface is incomplete".into());
    }
    Ok(corrections)
}

fn next_state(state: &str, action: &str) -> Result<String, Box<dyn Error>> {
    let next = if state == "empty" {
        action.to_string()
    } else {
        format!("{state}{action}")
    };
    if !STATES.contains(&next.as_str()) {
        return Err(format!("state escaped bounded reachable set: {next}").into());
    }
    Ok(next)
}

fn run_policy(
    policy: &str,
    documents: &[Vec<Window>],
    panels: &[PanelRow],
    models: &BTreeMap<String, ProductionModelV1>,
    candidates: &BTreeMap<(String, String), ProductionModelV1>,
    predictor: &BTreeMap<(String, String, String), i128>,
    corrections: &BTreeMap<String, i128>,
) -> Result<Trajectory, Box<dyn Error>> {
    if documents.len() != 24 || panels.len() != 24 {
        return Err("adaptive role must contain six source panels and 24 passages".into());
    }
    let mut state = "empty".to_string();
    let mut accepted = 0_usize;
    let mut head_fires = 0_usize;
    let mut trunk_fires = 0_usize;
    let mut positive_regret_q32 = 0_u128;
    let mut signed_regret_q32 = 0_i128;
    let mut decisions = Vec::new();
    for (document, windows) in documents.iter().enumerate() {
        let panel = &panels[document];
        let state_before = state.clone();
        let allowed = match policy {
            "adaptive" => ACTIONS.as_slice(),
            "head_only" => &ACTIONS[..1],
            "trunk_only" => &ACTIONS[1..],
            "always_abstain" => &ACTIONS[..0],
            _ => return Err("unknown policy".into()),
        };
        let selected = if accepted >= 2 {
            None
        } else {
            allowed
                .iter()
                .filter_map(|action| {
                    let upper = predictor
                        [&(panel.family.clone(), state.clone(), (*action).to_string())]
                        + corrections[&panel.family];
                    (upper < 0).then_some((*action, upper))
                })
                .min_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(right.0)))
        };
        let (action, upper, contrast) = if let Some((action, upper)) = selected {
            let contrast = contrast_q32(
                &models[&state],
                &candidates[&(state.clone(), action.to_string())],
                windows,
            )?;
            state = next_state(&state, action)?;
            accepted += 1;
            head_fires += usize::from(action == "H");
            trunk_fires += usize::from(action == "T");
            positive_regret_q32 = positive_regret_q32
                .checked_add(contrast.max(0) as u128)
                .ok_or("positive regret overflow")?;
            signed_regret_q32 = signed_regret_q32
                .checked_add(contrast)
                .ok_or("signed regret overflow")?;
            (action.to_string(), Some(upper), contrast)
        } else {
            ("abstain".to_string(), None, 0)
        };
        decisions.push(Decision {
            policy: policy.to_string(),
            document,
            family: panel.family.clone(),
            source_id: panel.source_id.clone(),
            passage: panel.passage,
            state_before,
            action,
            certified_upper_q32: upper,
            exact_contrast_q32: contrast,
            state_after: state.clone(),
        });
    }
    Ok(Trajectory {
        final_state: state,
        accepted,
        head_fires,
        trunk_fires,
        positive_regret_q32,
        signed_regret_q32,
        decisions,
    })
}

fn endpoint(
    model: &ProductionModelV1,
    documents: &[Vec<Window>],
) -> Result<Endpoint, Box<dyn Error>> {
    let windows = documents.iter().flatten().collect::<Vec<_>>();
    let values = parallel_map(windows.len(), |index| {
        let window = windows[index];
        let output = forward_production_model(model, &window.context)?;
        let max_logit = output
            .logits_q8
            .iter()
            .copied()
            .max()
            .ok_or("empty logits")?;
        let target_weight =
            base2_exp_neg_q15(output.logits_q8[window.target].saturating_sub(max_logit));
        let nll =
            base2_softmax_nll_millibits(&output.logits_q8, window.target, ZERO_FLOOR_MILLIBITS)
                .ok_or("canonical NLL failed")?;
        Ok((nll, target_weight == 0))
    })?;
    let mut endpoint = Endpoint::default();
    for (nll, zero_probability) in values {
        endpoint.zero_probability_windows += usize::from(zero_probability);
        endpoint.nll_millibits = endpoint
            .nll_millibits
            .checked_add(u128::from(nll))
            .ok_or("endpoint NLL overflow")?;
    }
    Ok(endpoint)
}

fn write_decisions(
    path: impl AsRef<Path>,
    trajectories: &[&Trajectory],
) -> Result<(), Box<dyn Error>> {
    let mut tsv = String::from(
        "policy\tdocument\tfamily\tsource_id\tpassage\tstate_before\taction\tcertified_upper_q32\texact_contrast_q32\tstate_after\n",
    );
    for trajectory in trajectories {
        for decision in &trajectory.decisions {
            tsv.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                decision.policy,
                decision.document,
                decision.family,
                decision.source_id,
                decision.passage,
                decision.state_before,
                decision.action,
                decision
                    .certified_upper_q32
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                decision.exact_contrast_q32,
                decision.state_after
            ));
        }
    }
    fs::write(path, tsv)?;
    Ok(())
}

fn evaluate(options: &BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    let manifest_directory = PathBuf::from(required(options, "--manifest-dir")?);
    let calibration_directory = PathBuf::from(required(options, "--calibration-dir")?);
    let output_directory = PathBuf::from(required(options, "--out-dir")?);
    let trace_path = PathBuf::from(required(options, "--trace")?);
    fs::create_dir_all(&output_directory)?;
    let models = load_state_models(&manifest_directory)?;
    let actions = read_actions(manifest_directory.join("actions.tsv"))?;
    let predictor = read_predictor(manifest_directory.join("predictor.tsv"))?;
    let candidates = candidate_models(&models, &actions)?;
    let corrections = read_corrections(calibration_directory.join("corrections.tsv"))?;

    let adaptive_tokens = read_tokens(required(options, "--adaptive-tokens")?, &models["empty"])?;
    let adaptive_documents = document_windows(&adaptive_tokens);
    let adaptive_panels = read_panels(required(options, "--adaptive-panels")?)?;
    let adaptive = run_policy(
        "adaptive",
        &adaptive_documents,
        &adaptive_panels,
        &models,
        &candidates,
        &predictor,
        &corrections,
    )?;
    let abstain = run_policy(
        "always_abstain",
        &adaptive_documents,
        &adaptive_panels,
        &models,
        &candidates,
        &predictor,
        &corrections,
    )?;
    let head = run_policy(
        "head_only",
        &adaptive_documents,
        &adaptive_panels,
        &models,
        &candidates,
        &predictor,
        &corrections,
    )?;
    let trunk = run_policy(
        "trunk_only",
        &adaptive_documents,
        &adaptive_panels,
        &models,
        &candidates,
        &predictor,
        &corrections,
    )?;
    write_decisions(
        output_directory.join("decisions.tsv"),
        &[&adaptive, &abstain, &head, &trunk],
    )?;

    let endpoint_tokens = read_tokens(required(options, "--endpoint-tokens")?, &models["empty"])?;
    let endpoint_documents = document_windows(&endpoint_tokens);
    let endpoint_panels = read_panels(required(options, "--endpoint-panels")?)?;
    if endpoint_documents.len() != 228 || endpoint_panels.len() != 228 {
        return Err("endpoint must contain 57 source panels and 228 passages".into());
    }
    let adaptive_endpoint = endpoint(&models[&adaptive.final_state], &endpoint_documents)?;
    let abstain_endpoint = endpoint(&models[&abstain.final_state], &endpoint_documents)?;
    let head_endpoint = endpoint(&models[&head.final_state], &endpoint_documents)?;
    let trunk_endpoint = endpoint(&models[&trunk.final_state], &endpoint_documents)?;
    let best_fixed_nll = head_endpoint
        .nll_millibits
        .min(trunk_endpoint.nll_millibits);
    let beats_abstain = adaptive_endpoint.nll_millibits < abstain_endpoint.nll_millibits;
    let beats_best_fixed = adaptive_endpoint.nll_millibits < best_fixed_nll;
    let zero_nonincrease =
        adaptive_endpoint.zero_probability_windows <= abstain_endpoint.zero_probability_windows;
    let both_families_fire = adaptive.head_fires > 0 && adaptive.trunk_fires > 0;
    let all_fired_strictly_negative = adaptive
        .decisions
        .iter()
        .filter(|decision| decision.action != "abstain")
        .all(|decision| decision.exact_contrast_q32 < 0);
    let zero_positive_regret = adaptive.positive_regret_q32 == 0 && all_fired_strictly_negative;
    let supported = beats_abstain
        && beats_best_fixed
        && zero_nonincrease
        && both_families_fire
        && zero_positive_regret;
    let verdict = if supported { "supported" } else { "falsified" };

    for (name, trajectory) in [
        ("adaptive", &adaptive),
        ("always-abstain", &abstain),
        ("head-only", &head),
        ("trunk-only", &trunk),
    ] {
        write_model(
            output_directory.join(format!("{name}-final.nsrlpm")),
            &models[&trajectory.final_state],
        )?;
    }
    let corrections_json = corrections
        .iter()
        .map(|(family, value)| format!("\"{family}\":\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    let endpoint_json = |name: &str, trajectory: &Trajectory, value: Endpoint| {
        format!(
            "\"{name}\":{{\"final_state\":\"{}\",\"accepted_actions\":{},\"total_nll_millibits\":\"{}\",\"zero_probability_windows\":{}}}",
            trajectory.final_state,
            trajectory.accepted,
            value.nll_millibits,
            value.zero_probability_windows
        )
    };
    let result = format!(
        concat!(
            "{{\"schema\":\"nsrl.adaptive_composition_result.v1\",",
            "\"analysis_role\":\"preregistered_fresh_source_execution\",",
            "\"verdict\":\"{}\",\"corrections_q32\":{{{}}},",
            "\"adaptive_trajectory\":{{\"final_state\":\"{}\",",
            "\"accepted_actions\":{},\"head_fires\":{},\"trunk_fires\":{},",
            "\"signed_regret_q32\":\"{}\",\"positive_regret_q32\":\"{}\"}},",
            "\"endpoints\":{{{},{},{},{}}},",
            "\"support_gates\":{{\"beats_always_abstain\":{},",
            "\"beats_best_fixed_policy\":{},\"zero_probability_nonincrease\":{},",
            "\"both_physical_families_fire\":{},\"zero_positive_regret\":{},",
            "\"all_fired_exact_contrasts_strictly_negative\":{},",
            "\"all_passed\":{}}},",
            "\"canonical_objective\":\"integer_base2_softmax_nll_millibits\",",
            "\"zero_probability_floor_millibits\":32000}}\n"
        ),
        verdict,
        corrections_json,
        adaptive.final_state,
        adaptive.accepted,
        adaptive.head_fires,
        adaptive.trunk_fires,
        adaptive.signed_regret_q32,
        adaptive.positive_regret_q32,
        endpoint_json("adaptive", &adaptive, adaptive_endpoint),
        endpoint_json("always_abstain", &abstain, abstain_endpoint),
        endpoint_json("head_only", &head, head_endpoint),
        endpoint_json("trunk_only", &trunk, trunk_endpoint),
        beats_abstain,
        beats_best_fixed,
        zero_nonincrease,
        both_families_fire,
        zero_positive_regret,
        all_fired_strictly_negative,
        supported,
    );
    fs::write(&trace_path, result)?;
    println!(
        "{{\"schema\":\"nsrl.adaptive_composition_execution.v1\",\"verdict\":\"{verdict}\",\"result\":\"{}\"}}",
        trace_path.display()
    );
    Ok(())
}
