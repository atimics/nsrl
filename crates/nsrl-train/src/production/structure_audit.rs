use std::collections::BTreeSet;
use std::fmt::Write;

use nsrl_core::{Base2SoftmaxNllQ47Components, base2_softmax_nll_q47_components};

use super::alignment::{
    DocumentWindow, can_perturb_both, document_windows_with_coordinates, set_parameter_delta,
};
use super::boolean_jet::{
    ProductionBooleanJetMove, ProductionBooleanJetProtocolBindings,
    production_boolean_jet_source_fnv64,
};
use super::{ProductionModelV1, TrainError, forward_production_model};

const RANK: usize = 6;
const VERTICES: usize = 1 << RANK;
const PROPOSAL_DOCUMENT_START: usize = 8;
const PROPOSAL_DOCUMENTS: usize = 64;
const WINDOWS_PER_DOCUMENT: usize = 2;
const TRANSFER_DOCUMENT_START: usize = 72;
const CONFIRMATION_DOCUMENT_START: usize = 136;
const CONFIRMATION_DOCUMENTS: usize = 64;
const CONFIRMATION_HARD_STOP: usize = 200;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionAtomicStructureRole {
    ProposalOnlyCalibration,
    UntouchedConfirmation,
}

impl ProductionAtomicStructureRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ProposalOnlyCalibration => "proposal_only_calibration",
            Self::UntouchedConfirmation => "untouched_confirmation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionAtomicStructureContract {
    pub analysis_role: ProductionAtomicStructureRole,
    pub protocol_bindings: ProductionBooleanJetProtocolBindings,
    pub model_hash: u64,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub source_index_hash: u64,
    pub proposal_source_cluster_hash: u64,
    pub proposal_source_clusters: usize,
    pub context_tokens: usize,
    pub document_start: usize,
    pub documents: usize,
    pub windows_per_document: usize,
    pub hard_stop_before_document: usize,
    pub move_fingerprint: u64,
    pub manifest_hash: u64,
    pub moves: Vec<ProductionBooleanJetMove>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionAtomicSourceBinding {
    pub source_index_hash: u64,
    pub proposal_source_cluster_hash: u64,
    pub proposal_source_clusters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProductionRepresentationConcordance {
    pub both_zero: usize,
    pub q20_zero_q32_nonzero: usize,
    pub q20_nonzero_q32_zero: usize,
    pub both_nonzero_sign_agree: usize,
    pub both_nonzero_sign_disagree: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionRepresentationDiscrepancy {
    pub q20_to_q32_multiplier: u64,
    pub residual_minimum: i128,
    pub residual_maximum: i128,
    pub residual_oscillation: u128,
    pub q20_minimizers: Vec<usize>,
    pub q32_minimizers: Vec<usize>,
    pub shared_minimizers: Vec<usize>,
    pub maximum_q32_regret_of_q20_minimizer: u64,
    pub certificate_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionDocumentRepresentationDiscrepancy {
    pub document: usize,
    pub residual_oscillation: u128,
    pub shared_minimizer: bool,
    pub maximum_q32_regret_of_q20_minimizer: u64,
    pub certificate_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProductionBoundaryTaxonomy {
    pub fine_grid_inactive: usize,
    pub phase_masked: usize,
    pub component_cancelled: usize,
    pub objective_visible: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionAtomicDocumentCoefficients {
    pub document: usize,
    pub coefficients: Vec<i128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionInteractionTailTrace {
    pub retained_order: usize,
    pub population_absolute_tail: u128,
    pub environmental_absolute_tail: u128,
    pub cancellation_mass: u128,
    pub truncated_minimizer: usize,
    pub exact_gap: u128,
    pub tail_regret_bound: u128,
    pub certificate_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionExchangeTrace {
    pub cardinality: usize,
    pub uniform_defect: u128,
    pub exchange_local_minima: usize,
    pub maximum_local_gap: u128,
    pub cardinality_defect_bound: u128,
    pub certificate_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionInteractionWidthTrace {
    pub support_hyperedges: Vec<usize>,
    pub elimination_orders_evaluated: usize,
    pub best_induced_width: usize,
    pub best_order: [usize; RANK],
    pub width_histogram: [usize; RANK],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionAtomicObjectiveTrace {
    pub fractional_bits: u8,
    pub vertex_losses: Vec<u64>,
    pub coefficients: Vec<i128>,
    pub documents: Vec<ProductionAtomicDocumentCoefficients>,
    pub population_absolute_mass_by_order: [u128; RANK + 1],
    pub environmental_absolute_mass_by_order: [u128; RANK + 1],
    pub tails: Vec<ProductionInteractionTailTrace>,
    pub exchanges: Vec<ProductionExchangeTrace>,
    pub interaction_width: ProductionInteractionWidthTrace,
    pub reconstruction_verified: bool,
    pub environment_aggregation_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionAtomicStructureTrace {
    pub contract: ProductionAtomicStructureContract,
    pub q20: ProductionAtomicObjectiveTrace,
    pub q32: ProductionAtomicObjectiveTrace,
    pub aggregate_concordance: ProductionRepresentationConcordance,
    pub document_concordance: ProductionRepresentationConcordance,
    pub aggregate_discrepancy: ProductionRepresentationDiscrepancy,
    pub document_discrepancies: Vec<ProductionDocumentRepresentationDiscrepancy>,
    pub boundary_taxonomy: ProductionBoundaryTaxonomy,
    pub boundary_taxonomy_by_atom: [ProductionBoundaryTaxonomy; RANK],
    pub optimizer_change_authorized: bool,
}

#[derive(Clone)]
struct AtomicVertexEval {
    document_q20: Vec<u64>,
    document_q32: Vec<u64>,
    components: Vec<Base2SoftmaxNllQ47Components>,
}

pub fn freeze_production_atomic_structure_contract(
    model: &ProductionModelV1,
    token_stream_hash: u64,
    context_tokens: usize,
    moves: Vec<ProductionBooleanJetMove>,
    protocol_bindings: ProductionBooleanJetProtocolBindings,
    source_binding: ProductionAtomicSourceBinding,
    document_start: usize,
    documents: usize,
) -> Result<ProductionAtomicStructureContract, TrainError> {
    model.validate()?;
    validate_contract_inputs(model, context_tokens, &moves, protocol_bindings)?;
    if source_binding.source_index_hash == 0
        || source_binding.proposal_source_cluster_hash == 0
        || source_binding.proposal_source_clusters == 0
    {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_source_binding",
        ));
    }
    let (analysis_role, hard_stop_before_document) = atomic_surface(document_start, documents)?;
    let move_fingerprint = atomic_move_fingerprint(&moves);
    let manifest_hash = atomic_manifest_hash(
        protocol_bindings,
        model.model_hash(),
        model.tokenizer_hash,
        token_stream_hash,
        source_binding,
        context_tokens,
        document_start,
        documents,
        &moves,
    );
    Ok(ProductionAtomicStructureContract {
        analysis_role,
        protocol_bindings,
        model_hash: model.model_hash(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        source_index_hash: source_binding.source_index_hash,
        proposal_source_cluster_hash: source_binding.proposal_source_cluster_hash,
        proposal_source_clusters: source_binding.proposal_source_clusters,
        context_tokens,
        document_start,
        documents,
        windows_per_document: WINDOWS_PER_DOCUMENT,
        hard_stop_before_document,
        move_fingerprint,
        manifest_hash,
        moves,
    })
}

pub fn audit_production_atomic_structure(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    contract: ProductionAtomicStructureContract,
) -> Result<ProductionAtomicStructureTrace, TrainError> {
    model.validate()?;
    validate_contract_inputs(
        model,
        contract.context_tokens,
        &contract.moves,
        contract.protocol_bindings,
    )?;
    if contract.model_hash != model.model_hash()
        || contract.tokenizer_hash != model.tokenizer_hash
        || contract.token_stream_hash != token_stream_hash
        || contract.source_index_hash == 0
        || contract.proposal_source_cluster_hash == 0
        || contract.proposal_source_clusters == 0
        || atomic_surface(contract.document_start, contract.documents).ok()
            != Some((contract.analysis_role, contract.hard_stop_before_document))
        || contract.windows_per_document != WINDOWS_PER_DOCUMENT
        || contract.move_fingerprint != atomic_move_fingerprint(&contract.moves)
        || contract.manifest_hash
            != atomic_manifest_hash(
                contract.protocol_bindings,
                model.model_hash(),
                model.tokenizer_hash,
                token_stream_hash,
                ProductionAtomicSourceBinding {
                    source_index_hash: contract.source_index_hash,
                    proposal_source_cluster_hash: contract.proposal_source_cluster_hash,
                    proposal_source_clusters: contract.proposal_source_clusters,
                },
                contract.context_tokens,
                contract.document_start,
                contract.documents,
                &contract.moves,
            )
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_contract_mismatch",
        ));
    }

    let all_windows = document_windows_with_coordinates(tokens, contract.context_tokens);
    let windows = atomic_windows(
        &all_windows,
        contract.document_start,
        contract.documents,
        contract.windows_per_document,
    )?;
    if windows
        .iter()
        .any(|window| window.document >= contract.hard_stop_before_document)
    {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_hard_stop_crossed",
        ));
    }
    let mut vertices = Vec::with_capacity(VERTICES);
    for mask in 0..VERTICES {
        let mut candidate = model.clone();
        for bit in 0..RANK {
            if mask & (1 << bit) != 0 {
                let movement = &contract.moves[bit];
                set_parameter_delta(
                    &mut candidate,
                    movement.group_index,
                    movement.coordinate,
                    movement.parameter_delta,
                )?;
            }
        }
        vertices.push(evaluate_atomic_vertex(
            &candidate,
            &windows,
            contract.document_start,
            contract.documents,
        )?);
    }

    let q20 = build_objective_trace(&vertices, 20, contract.document_start, contract.documents)?;
    let q32 = build_objective_trace(&vertices, 32, contract.document_start, contract.documents)?;
    let aggregate_concordance = representation_concordance(&q20.coefficients, &q32.coefficients);
    let aggregate_discrepancy = representation_discrepancy(&q20.vertex_losses, &q32.vertex_losses)?;
    let mut document_concordance = ProductionRepresentationConcordance::default();
    let mut document_discrepancies = Vec::with_capacity(contract.documents);
    for (q20_document, q32_document) in q20.documents.iter().zip(&q32.documents) {
        if q20_document.document != q32_document.document {
            return Err(TrainError::CoreRejected(
                "production_atomic_structure_document_order",
            ));
        }
        add_concordance(
            &mut document_concordance,
            representation_concordance(&q20_document.coefficients, &q32_document.coefficients),
        );
        let discrepancy = representation_discrepancy(
            &reconstruct_losses(&q20_document.coefficients)?,
            &reconstruct_losses(&q32_document.coefficients)?,
        )?;
        document_discrepancies.push(ProductionDocumentRepresentationDiscrepancy {
            document: q20_document.document,
            residual_oscillation: discrepancy.residual_oscillation,
            shared_minimizer: !discrepancy.shared_minimizers.is_empty(),
            maximum_q32_regret_of_q20_minimizer: discrepancy.maximum_q32_regret_of_q20_minimizer,
            certificate_verified: discrepancy.certificate_verified,
        });
    }
    let (boundary_taxonomy, boundary_taxonomy_by_atom) = build_boundary_taxonomy(
        &vertices,
        &windows,
        contract.document_start,
        contract.documents,
        contract.windows_per_document,
    )?;

    Ok(ProductionAtomicStructureTrace {
        contract,
        q20,
        q32,
        aggregate_concordance,
        document_concordance,
        aggregate_discrepancy,
        document_discrepancies,
        boundary_taxonomy,
        boundary_taxonomy_by_atom,
        optimizer_change_authorized: false,
    })
}

fn validate_contract_inputs(
    model: &ProductionModelV1,
    context_tokens: usize,
    moves: &[ProductionBooleanJetMove],
    protocol_bindings: ProductionBooleanJetProtocolBindings,
) -> Result<(), TrainError> {
    if context_tokens == 0
        || context_tokens > model.config.context_tokens
        || moves.len() != RANK
        || protocol_bindings.source_fnv64 == 0
        || protocol_bindings.binary_fnv64 == 0
        || protocol_bindings.source_fnv64 != production_boolean_jet_source_fnv64()
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut coordinates = BTreeSet::new();
    for (order, movement) in moves.iter().enumerate() {
        let trunk = order < 4 && movement.block == "trunk" && movement.group_index == 3;
        let head = order >= 4 && movement.block == "head" && movement.group_index >= 11;
        if !(trunk || head)
            || movement.canonical_order != order
            || !matches!(movement.parameter_delta, -1 | 1)
            || movement.move_kind != "model_only_unit_sign_probe"
            || !coordinates.insert((movement.group_index, movement.coordinate))
            || !can_perturb_both(model, movement.group_index, movement.coordinate)
        {
            return Err(TrainError::CoreRejected(
                "production_atomic_structure_invalid_move_family",
            ));
        }
    }
    Ok(())
}

fn atomic_surface(
    document_start: usize,
    documents: usize,
) -> Result<(ProductionAtomicStructureRole, usize), TrainError> {
    match (document_start, documents) {
        (PROPOSAL_DOCUMENT_START, PROPOSAL_DOCUMENTS) => Ok((
            ProductionAtomicStructureRole::ProposalOnlyCalibration,
            TRANSFER_DOCUMENT_START,
        )),
        (CONFIRMATION_DOCUMENT_START, CONFIRMATION_DOCUMENTS) => Ok((
            ProductionAtomicStructureRole::UntouchedConfirmation,
            CONFIRMATION_HARD_STOP,
        )),
        _ => Err(TrainError::CoreRejected(
            "production_atomic_structure_unsupported_surface",
        )),
    }
}

fn atomic_windows(
    windows: &[DocumentWindow],
    document_start: usize,
    documents: usize,
    windows_per_document: usize,
) -> Result<Vec<DocumentWindow>, TrainError> {
    let mut selected = Vec::with_capacity(documents * windows_per_document);
    for document in document_start..document_start + documents {
        let before = selected.len();
        selected.extend(
            windows
                .iter()
                .filter(|window| window.document == document)
                .take(windows_per_document)
                .cloned(),
        );
        if selected.len() - before != windows_per_document {
            return Err(TrainError::CoreRejected(
                "production_atomic_structure_windows_missing",
            ));
        }
    }
    Ok(selected)
}

fn evaluate_atomic_vertex(
    model: &ProductionModelV1,
    windows: &[DocumentWindow],
    document_start: usize,
    documents: usize,
) -> Result<AtomicVertexEval, TrainError> {
    let mut document_q20 = vec![0_u64; documents];
    let mut document_q32 = vec![0_u64; documents];
    let mut components = Vec::with_capacity(windows.len());
    for window in windows {
        let forward = forward_production_model(model, &window.context)?;
        let component =
            base2_softmax_nll_q47_components(&forward.logits_q8, window.target as usize).ok_or(
                TrainError::CoreRejected("production_atomic_structure_objective_components"),
            )?;
        let document = window.document - document_start;
        let q20 = component
            .denominator_log2_q20
            .checked_sub(component.target_log2_q20)
            .ok_or(TrainError::CoreRejected(
                "production_atomic_structure_q20_underflow",
            ))?;
        let q32 = component
            .denominator_log2_q32
            .checked_sub(component.target_log2_q32)
            .ok_or(TrainError::CoreRejected(
                "production_atomic_structure_q32_underflow",
            ))?;
        document_q20[document] =
            document_q20[document]
                .checked_add(q20)
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_q20_overflow",
                ))?;
        document_q32[document] =
            document_q32[document]
                .checked_add(q32)
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_q32_overflow",
                ))?;
        components.push(component);
    }
    Ok(AtomicVertexEval {
        document_q20,
        document_q32,
        components,
    })
}

fn build_objective_trace(
    vertices: &[AtomicVertexEval],
    fractional_bits: u8,
    document_start: usize,
    documents_count: usize,
) -> Result<ProductionAtomicObjectiveTrace, TrainError> {
    if vertices.len() != VERTICES {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_vertex_shape",
        ));
    }
    let document_loss = |vertex: &AtomicVertexEval, document: usize| match fractional_bits {
        20 => vertex.document_q20[document],
        32 => vertex.document_q32[document],
        _ => unreachable!("atomic structure supports Q20 and Q32"),
    };
    let mut vertex_losses = vec![0_u64; VERTICES];
    for (mask, vertex) in vertices.iter().enumerate() {
        for document in 0..documents_count {
            vertex_losses[mask] = vertex_losses[mask]
                .checked_add(document_loss(vertex, document))
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_vertex_loss_overflow",
                ))?;
        }
    }
    let coefficients = mobius(&vertex_losses)?;
    let reconstruction_verified = reconstructs(&vertex_losses, &coefficients);
    if !reconstruction_verified {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_mobius_reconstruction",
        ));
    }
    let mut documents = Vec::with_capacity(documents_count);
    let mut coefficient_sums = vec![0_i128; VERTICES];
    let mut environmental_absolute_mass_by_order = [0_u128; RANK + 1];
    for document in 0..documents_count {
        let losses = vertices
            .iter()
            .map(|vertex| document_loss(vertex, document))
            .collect::<Vec<_>>();
        let document_coefficients = mobius(&losses)?;
        if !reconstructs(&losses, &document_coefficients) {
            return Err(TrainError::CoreRejected(
                "production_atomic_structure_document_reconstruction",
            ));
        }
        for (mask, &coefficient) in document_coefficients.iter().enumerate() {
            coefficient_sums[mask] =
                coefficient_sums[mask]
                    .checked_add(coefficient)
                    .ok_or(TrainError::CoreRejected(
                        "production_atomic_structure_coefficient_sum_overflow",
                    ))?;
            environmental_absolute_mass_by_order[mask.count_ones() as usize] =
                environmental_absolute_mass_by_order[mask.count_ones() as usize]
                    .checked_add(coefficient.unsigned_abs())
                    .ok_or(TrainError::CoreRejected(
                        "production_atomic_structure_environment_mass_overflow",
                    ))?;
        }
        documents.push(ProductionAtomicDocumentCoefficients {
            document: document_start + document,
            coefficients: document_coefficients,
        });
    }
    let environment_aggregation_verified = coefficient_sums == coefficients;
    if !environment_aggregation_verified {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_environment_aggregation",
        ));
    }
    let mut population_absolute_mass_by_order = [0_u128; RANK + 1];
    for (mask, coefficient) in coefficients.iter().enumerate() {
        population_absolute_mass_by_order[mask.count_ones() as usize] =
            population_absolute_mass_by_order[mask.count_ones() as usize]
                .checked_add(coefficient.unsigned_abs())
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_population_mass_overflow",
                ))?;
    }
    let tails = (1..RANK)
        .map(|retained_order| {
            interaction_tail(
                retained_order,
                &vertex_losses,
                &coefficients,
                &population_absolute_mass_by_order,
                &environmental_absolute_mass_by_order,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exchanges = (1..RANK)
        .map(|cardinality| exchange_trace(&vertex_losses, cardinality))
        .collect::<Result<Vec<_>, _>>()?;
    let interaction_width = interaction_width(&coefficients);
    Ok(ProductionAtomicObjectiveTrace {
        fractional_bits,
        vertex_losses,
        coefficients,
        documents,
        population_absolute_mass_by_order,
        environmental_absolute_mass_by_order,
        tails,
        exchanges,
        interaction_width,
        reconstruction_verified,
        environment_aggregation_verified,
    })
}

fn mobius(losses: &[u64]) -> Result<Vec<i128>, TrainError> {
    if losses.len() != VERTICES {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_mobius_shape",
        ));
    }
    let mut coefficients = losses.iter().copied().map(i128::from).collect::<Vec<_>>();
    for bit in 0..RANK {
        for mask in 0..VERTICES {
            if mask & (1 << bit) != 0 {
                coefficients[mask] = coefficients[mask]
                    .checked_sub(coefficients[mask ^ (1 << bit)])
                    .ok_or(TrainError::CoreRejected(
                        "production_atomic_structure_mobius_overflow",
                    ))?;
            }
        }
    }
    Ok(coefficients)
}

fn reconstructs(losses: &[u64], coefficients: &[i128]) -> bool {
    (0..VERTICES).all(|mask| {
        let mut sum = 0_i128;
        let mut subset = mask;
        loop {
            let Some(next) = sum.checked_add(coefficients[subset]) else {
                return false;
            };
            sum = next;
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & mask;
        }
        sum == i128::from(losses[mask])
    })
}

fn reconstruct_losses(coefficients: &[i128]) -> Result<Vec<u64>, TrainError> {
    if coefficients.len() != VERTICES {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_reconstruction_shape",
        ));
    }
    let mut losses = Vec::with_capacity(VERTICES);
    for mask in 0..VERTICES {
        let mut sum = 0_i128;
        let mut subset = mask;
        loop {
            sum = sum
                .checked_add(coefficients[subset])
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_reconstruction_overflow",
                ))?;
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & mask;
        }
        losses.push(u64::try_from(sum).map_err(|_| {
            TrainError::CoreRejected("production_atomic_structure_reconstruction_range")
        })?);
    }
    Ok(losses)
}

fn representation_discrepancy(
    q20: &[u64],
    q32: &[u64],
) -> Result<ProductionRepresentationDiscrepancy, TrainError> {
    const MULTIPLIER: u64 = 1 << 12;
    if q20.len() != VERTICES || q32.len() != VERTICES {
        return Err(TrainError::CoreRejected(
            "production_atomic_structure_representation_shape",
        ));
    }
    let residuals = q20
        .iter()
        .zip(q32)
        .map(|(&coarse, &fine)| {
            i128::from(fine)
                .checked_sub(i128::from(coarse) * i128::from(MULTIPLIER))
                .ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_representation_overflow",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let residual_minimum = *residuals
        .iter()
        .min()
        .expect("rank-six residual cube is nonempty");
    let residual_maximum = *residuals
        .iter()
        .max()
        .expect("rank-six residual cube is nonempty");
    let residual_oscillation = (residual_maximum - residual_minimum).unsigned_abs();
    let q20_minimum = *q20.iter().min().expect("rank-six cube is nonempty");
    let q32_minimum = *q32.iter().min().expect("rank-six cube is nonempty");
    let q20_minimizers = q20
        .iter()
        .enumerate()
        .filter_map(|(mask, &loss)| (loss == q20_minimum).then_some(mask))
        .collect::<Vec<_>>();
    let q32_minimizers = q32
        .iter()
        .enumerate()
        .filter_map(|(mask, &loss)| (loss == q32_minimum).then_some(mask))
        .collect::<Vec<_>>();
    let shared_minimizers = q20_minimizers
        .iter()
        .copied()
        .filter(|mask| q32[*mask] == q32_minimum)
        .collect::<Vec<_>>();
    let maximum_q32_regret_of_q20_minimizer = q20_minimizers
        .iter()
        .map(|&mask| q32[mask] - q32_minimum)
        .max()
        .expect("rank-six cube has a Q20 minimizer");
    Ok(ProductionRepresentationDiscrepancy {
        q20_to_q32_multiplier: MULTIPLIER,
        residual_minimum,
        residual_maximum,
        residual_oscillation,
        q20_minimizers,
        q32_minimizers,
        shared_minimizers,
        maximum_q32_regret_of_q20_minimizer,
        certificate_verified: u128::from(maximum_q32_regret_of_q20_minimizer)
            <= residual_oscillation,
    })
}

fn interaction_tail(
    retained_order: usize,
    losses: &[u64],
    coefficients: &[i128],
    population_mass: &[u128; RANK + 1],
    environmental_mass: &[u128; RANK + 1],
) -> Result<ProductionInteractionTailTrace, TrainError> {
    let population_absolute_tail = population_mass[retained_order + 1..]
        .iter()
        .try_fold(0_u128, |sum, &value| sum.checked_add(value))
        .ok_or(TrainError::CoreRejected(
            "production_atomic_structure_tail_overflow",
        ))?;
    let environmental_absolute_tail = environmental_mass[retained_order + 1..]
        .iter()
        .try_fold(0_u128, |sum, &value| sum.checked_add(value))
        .ok_or(TrainError::CoreRejected(
            "production_atomic_structure_tail_overflow",
        ))?;
    let mut truncated_minimizer = 0;
    let mut truncated_minimum = i128::MAX;
    for mask in 0..VERTICES {
        let mut value = 0_i128;
        let mut subset = mask;
        loop {
            if subset.count_ones() as usize <= retained_order {
                value = value
                    .checked_add(coefficients[subset])
                    .ok_or(TrainError::CoreRejected(
                        "production_atomic_structure_truncation_overflow",
                    ))?;
            }
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & mask;
        }
        if value < truncated_minimum {
            truncated_minimum = value;
            truncated_minimizer = mask;
        }
    }
    let exact_minimum = *losses.iter().min().expect("rank-six cube is nonempty");
    let exact_gap = u128::from(losses[truncated_minimizer] - exact_minimum);
    Ok(ProductionInteractionTailTrace {
        retained_order,
        population_absolute_tail,
        environmental_absolute_tail,
        cancellation_mass: environmental_absolute_tail.saturating_sub(population_absolute_tail),
        truncated_minimizer,
        exact_gap,
        tail_regret_bound: population_absolute_tail,
        certificate_verified: exact_gap <= population_absolute_tail
            && population_absolute_tail <= environmental_absolute_tail,
    })
}

fn exchange_trace(
    losses: &[u64],
    cardinality: usize,
) -> Result<ProductionExchangeTrace, TrainError> {
    let masks = (0..VERTICES)
        .filter(|mask| mask.count_ones() as usize == cardinality)
        .collect::<Vec<_>>();
    let mut uniform_defect = 0_u128;
    for &left in &masks {
        for &right in &masks {
            let left_only = left & !right;
            let right_only = right & !left;
            for source in 0..RANK {
                if left_only & (1 << source) == 0 {
                    continue;
                }
                let mut best = None;
                for target in 0..RANK {
                    if right_only & (1 << target) == 0 {
                        continue;
                    }
                    let exchanged_left = (left ^ (1 << source)) | (1 << target);
                    let exchanged_right = (right | (1 << source)) ^ (1 << target);
                    let defect = i128::from(losses[exchanged_left])
                        + i128::from(losses[exchanged_right])
                        - i128::from(losses[left])
                        - i128::from(losses[right]);
                    best = Some(best.map_or(defect, |current: i128| current.min(defect)));
                }
                let best = best.ok_or(TrainError::CoreRejected(
                    "production_atomic_structure_exchange_target_missing",
                ))?;
                if best > 0 {
                    uniform_defect = uniform_defect.max(best.unsigned_abs());
                }
            }
        }
    }
    let global_minimum = masks
        .iter()
        .map(|&mask| losses[mask])
        .min()
        .expect("fixed-cardinality slice is nonempty");
    let mut exchange_local_minima = 0;
    let mut maximum_local_gap = 0_u128;
    for &mask in &masks {
        let mut local = true;
        for source in 0..RANK {
            if mask & (1 << source) == 0 {
                continue;
            }
            for target in 0..RANK {
                if mask & (1 << target) != 0 {
                    continue;
                }
                let neighbor = (mask ^ (1 << source)) | (1 << target);
                if losses[neighbor] < losses[mask] {
                    local = false;
                }
            }
        }
        if local {
            exchange_local_minima += 1;
            maximum_local_gap = maximum_local_gap.max(u128::from(losses[mask] - global_minimum));
        }
    }
    let cardinality_defect_bound =
        uniform_defect
            .checked_mul(cardinality as u128)
            .ok_or(TrainError::CoreRejected(
                "production_atomic_structure_exchange_bound_overflow",
            ))?;
    Ok(ProductionExchangeTrace {
        cardinality,
        uniform_defect,
        exchange_local_minima,
        maximum_local_gap,
        cardinality_defect_bound,
        certificate_verified: maximum_local_gap <= cardinality_defect_bound,
    })
}

fn interaction_width(coefficients: &[i128]) -> ProductionInteractionWidthTrace {
    let support_hyperedges = coefficients
        .iter()
        .enumerate()
        .filter(|(mask, coefficient)| mask.count_ones() >= 2 && **coefficient != 0)
        .map(|(mask, _)| mask)
        .collect::<Vec<_>>();
    let mut adjacency = [0_u8; RANK];
    for &edge in &support_hyperedges {
        for left in 0..RANK {
            if edge & (1 << left) == 0 {
                continue;
            }
            for right in left + 1..RANK {
                if edge & (1 << right) != 0 {
                    adjacency[left] |= 1 << right;
                    adjacency[right] |= 1 << left;
                }
            }
        }
    }
    let mut orders = Vec::with_capacity(720);
    permutations(
        &mut Vec::with_capacity(RANK),
        &mut [false; RANK],
        &mut orders,
    );
    let mut best_induced_width = RANK;
    let mut best_order = [0; RANK];
    let mut width_histogram = [0; RANK];
    for order in &orders {
        let width = induced_width(adjacency, order);
        width_histogram[width] += 1;
        if width < best_induced_width {
            best_induced_width = width;
            best_order = *order;
        }
    }
    ProductionInteractionWidthTrace {
        support_hyperedges,
        elimination_orders_evaluated: orders.len(),
        best_induced_width,
        best_order,
        width_histogram,
    }
}

fn permutations(prefix: &mut Vec<usize>, used: &mut [bool; RANK], output: &mut Vec<[usize; RANK]>) {
    if prefix.len() == RANK {
        output.push(prefix.as_slice().try_into().expect("rank-six order"));
        return;
    }
    for variable in 0..RANK {
        if used[variable] {
            continue;
        }
        used[variable] = true;
        prefix.push(variable);
        permutations(prefix, used, output);
        prefix.pop();
        used[variable] = false;
    }
}

fn induced_width(mut adjacency: [u8; RANK], order: &[usize; RANK]) -> usize {
    let mut live = (1_u8 << RANK) - 1;
    let mut width = 0;
    for &variable in order {
        let neighbors = adjacency[variable] & live & !(1 << variable);
        width = width.max(neighbors.count_ones() as usize);
        let variables = (0..RANK)
            .filter(|candidate| neighbors & (1 << candidate) != 0)
            .collect::<Vec<_>>();
        for &left in &variables {
            for &right in &variables {
                if left != right {
                    adjacency[left] |= 1 << right;
                }
            }
            adjacency[left] &= !(1 << variable);
        }
        live &= !(1 << variable);
    }
    width
}

fn representation_concordance(q20: &[i128], q32: &[i128]) -> ProductionRepresentationConcordance {
    let mut trace = ProductionRepresentationConcordance::default();
    for (&coarse, &fine) in q20.iter().skip(1).zip(q32.iter().skip(1)) {
        match (coarse.signum(), fine.signum()) {
            (0, 0) => trace.both_zero += 1,
            (0, _) => trace.q20_zero_q32_nonzero += 1,
            (_, 0) => trace.q20_nonzero_q32_zero += 1,
            (left, right) if left == right => trace.both_nonzero_sign_agree += 1,
            _ => trace.both_nonzero_sign_disagree += 1,
        }
    }
    trace
}

fn add_concordance(
    total: &mut ProductionRepresentationConcordance,
    value: ProductionRepresentationConcordance,
) {
    total.both_zero += value.both_zero;
    total.q20_zero_q32_nonzero += value.q20_zero_q32_nonzero;
    total.q20_nonzero_q32_zero += value.q20_nonzero_q32_zero;
    total.both_nonzero_sign_agree += value.both_nonzero_sign_agree;
    total.both_nonzero_sign_disagree += value.both_nonzero_sign_disagree;
}

fn build_boundary_taxonomy(
    vertices: &[AtomicVertexEval],
    windows: &[DocumentWindow],
    document_start: usize,
    documents: usize,
    windows_per_document: usize,
) -> Result<
    (
        ProductionBoundaryTaxonomy,
        [ProductionBoundaryTaxonomy; RANK],
    ),
    TrainError,
> {
    let mut total = ProductionBoundaryTaxonomy::default();
    let mut by_atom = [ProductionBoundaryTaxonomy::default(); RANK];
    for atom in 0..RANK {
        for base_mask in 0..VERTICES {
            if base_mask & (1 << atom) != 0 {
                continue;
            }
            let candidate_mask = base_mask | (1 << atom);
            for document in 0..documents {
                let mut fine_active = false;
                let mut component_activity = 0_u128;
                let mut q20_contrast = 0_i128;
                for window_offset in 0..windows_per_document {
                    let index = document * windows_per_document + window_offset;
                    if windows[index].document != document_start + document {
                        return Err(TrainError::CoreRejected(
                            "production_atomic_structure_window_order",
                        ));
                    }
                    let base = vertices[base_mask].components[index];
                    let candidate = vertices[candidate_mask].components[index];
                    let denominator_displacement = i128::from(candidate.denominator_log2_q32)
                        - i128::from(base.denominator_log2_q32);
                    let target_displacement =
                        i128::from(candidate.target_log2_q32) - i128::from(base.target_log2_q32);
                    fine_active |= denominator_displacement != 0 || target_displacement != 0;
                    let denominator_crossing = i128::from(candidate.denominator_log2_q20)
                        - i128::from(base.denominator_log2_q20);
                    let target_crossing =
                        i128::from(candidate.target_log2_q20) - i128::from(base.target_log2_q20);
                    component_activity = component_activity
                        .checked_add(denominator_crossing.unsigned_abs())
                        .and_then(|sum| sum.checked_add(target_crossing.unsigned_abs()))
                        .ok_or(TrainError::CoreRejected(
                            "production_atomic_structure_boundary_activity_overflow",
                        ))?;
                    q20_contrast = q20_contrast
                        .checked_add(denominator_crossing - target_crossing)
                        .ok_or(TrainError::CoreRejected(
                            "production_atomic_structure_boundary_contrast_overflow",
                        ))?;
                }
                let direct = i128::from(vertices[candidate_mask].document_q20[document])
                    - i128::from(vertices[base_mask].document_q20[document]);
                if q20_contrast != direct {
                    return Err(TrainError::CoreRejected(
                        "production_atomic_structure_boundary_decomposition",
                    ));
                }
                let category = if !fine_active {
                    0
                } else if component_activity == 0 {
                    1
                } else if q20_contrast == 0 {
                    2
                } else {
                    3
                };
                increment_taxonomy(&mut total, category);
                increment_taxonomy(&mut by_atom[atom], category);
            }
        }
    }
    Ok((total, by_atom))
}

fn increment_taxonomy(trace: &mut ProductionBoundaryTaxonomy, category: usize) {
    match category {
        0 => trace.fine_grid_inactive += 1,
        1 => trace.phase_masked += 1,
        2 => trace.component_cancelled += 1,
        3 => trace.objective_visible += 1,
        _ => unreachable!(),
    }
}

fn atomic_move_fingerprint(moves: &[ProductionBooleanJetMove]) -> u64 {
    let mut hash = FNV_OFFSET;
    for movement in moves {
        for value in [movement.group_index as u64, movement.coordinate as u64] {
            for byte in value.to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
        }
        hash = fnv_byte(hash, movement.parameter_delta as u8);
    }
    hash
}

fn atomic_manifest_hash(
    bindings: ProductionBooleanJetProtocolBindings,
    model_hash: u64,
    tokenizer_hash: u64,
    token_stream_hash: u64,
    source_binding: ProductionAtomicSourceBinding,
    context_tokens: usize,
    document_start: usize,
    documents: usize,
    moves: &[ProductionBooleanJetMove],
) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in b"nsrl.production_atomic_structure.v1" {
        hash = fnv_byte(hash, *byte);
    }
    for value in [
        bindings.source_fnv64,
        bindings.binary_fnv64,
        model_hash,
        tokenizer_hash,
        token_stream_hash,
        source_binding.source_index_hash,
        source_binding.proposal_source_cluster_hash,
        source_binding.proposal_source_clusters as u64,
        context_tokens as u64,
        document_start as u64,
        documents as u64,
        WINDOWS_PER_DOCUMENT as u64,
        atomic_move_fingerprint(moves),
    ] {
        for byte in value.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    hash
}

const fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

impl ProductionAtomicStructureContract {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_atomic_structure_contract.v1\",",
                "\"analysis_role\":\"{}\",\"rank\":6,",
                "\"bindings\":{{\"source_fnv64\":\"0x{:016x}\",",
                "\"binary_fnv64\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\",",
                "\"source_index_hash\":\"0x{:016x}\"}},",
                "\"surface\":{{\"document_start\":{},\"documents\":{},",
                "\"windows_per_document\":{},\"context_tokens\":{},",
                "\"hard_stop_before_document\":{}}},",
                "\"source_population\":{{\"proposal_source_cluster_hash\":\"0x{:016x}\",",
                "\"proposal_source_clusters\":{},",
                "\"source_clustered_fold_estimation_available\":{}}},",
                "\"objectives\":[{{\"algorithm\":\"q47_logit_anchored\",",
                "\"fractional_bits\":20}},{{\"algorithm\":\"q47_logit_anchored\",",
                "\"fractional_bits\":32}}],",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\",\"moves\":"
            ),
            self.analysis_role.as_str(),
            self.protocol_bindings.source_fnv64,
            self.protocol_bindings.binary_fnv64,
            self.model_hash,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.source_index_hash,
            self.document_start,
            self.documents,
            self.windows_per_document,
            self.context_tokens,
            self.hard_stop_before_document,
            self.proposal_source_cluster_hash,
            self.proposal_source_clusters,
            self.proposal_source_clusters >= 2,
            self.move_fingerprint,
            self.manifest_hash,
        )
        .expect("writing atomic structure contract cannot fail");
        push_moves(&mut json, &self.moves);
        json.push_str(",\"authorization\":{\"optimizer_change\":false,\"paid_scaling\":false}}\n");
        json
    }
}

impl ProductionAtomicStructureTrace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        let reserved_documents_read = match self.contract.analysis_role {
            ProductionAtomicStructureRole::ProposalOnlyCalibration => 0,
            ProductionAtomicStructureRole::UntouchedConfirmation => self.contract.documents,
        };
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_atomic_structure.v1\",",
                "\"analysis_role\":\"{}\",\"rank\":6,",
                "\"vertices_evaluated\":64,\"transfer_documents_read\":0,",
                "\"reserved_documents_read\":{},",
                "\"bindings\":{{\"source_fnv64\":\"0x{:016x}\",",
                "\"binary_fnv64\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\",",
                "\"source_index_hash\":\"0x{:016x}\",",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\"}},",
                "\"surface\":{{\"document_start\":{},\"documents\":{},",
                "\"windows_per_document\":{},\"context_tokens\":{},",
                "\"hard_stop_before_document\":{}}},",
                "\"source_population\":{{\"proposal_source_cluster_hash\":\"0x{:016x}\",",
                "\"proposal_source_clusters\":{},",
                "\"source_clustered_fold_estimation_available\":{}}},\"moves\":"
            ),
            self.contract.analysis_role.as_str(),
            reserved_documents_read,
            self.contract.protocol_bindings.source_fnv64,
            self.contract.protocol_bindings.binary_fnv64,
            self.contract.model_hash,
            self.contract.tokenizer_hash,
            self.contract.token_stream_hash,
            self.contract.source_index_hash,
            self.contract.move_fingerprint,
            self.contract.manifest_hash,
            self.contract.document_start,
            self.contract.documents,
            self.contract.windows_per_document,
            self.contract.context_tokens,
            self.contract.hard_stop_before_document,
            self.contract.proposal_source_cluster_hash,
            self.contract.proposal_source_clusters,
            self.contract.proposal_source_clusters >= 2,
        )
        .expect("writing atomic structure header cannot fail");
        push_moves(&mut json, &self.contract.moves);
        json.push_str(",\"q20\":");
        push_objective(&mut json, &self.q20);
        json.push_str(",\"q32\":");
        push_objective(&mut json, &self.q32);
        json.push_str(",\"representation_concordance\":{\"aggregate\":");
        push_concordance(&mut json, self.aggregate_concordance);
        json.push_str(",\"document_coefficients\":");
        push_concordance(&mut json, self.document_concordance);
        json.push_str("},\"representation_discrepancy\":{\"aggregate\":");
        push_discrepancy(&mut json, &self.aggregate_discrepancy);
        json.push_str(",\"documents\":[");
        for (index, document) in self.document_discrepancies.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            write!(
                json,
                concat!(
                    "{{\"document\":{},\"residual_oscillation\":\"{}\",",
                    "\"shared_minimizer\":{},",
                    "\"maximum_q32_regret_of_q20_minimizer\":{},",
                    "\"certificate_verified\":{}}}"
                ),
                document.document,
                document.residual_oscillation,
                document.shared_minimizer,
                document.maximum_q32_regret_of_q20_minimizer,
                document.certificate_verified,
            )
            .expect("writing document representation discrepancy cannot fail");
        }
        json.push_str("]},\"boundary_taxonomy\":{\"all_edges\":");
        push_taxonomy(&mut json, self.boundary_taxonomy);
        json.push_str(",\"by_atom\":[");
        for (atom, taxonomy) in self.boundary_taxonomy_by_atom.iter().enumerate() {
            if atom != 0 {
                json.push(',');
            }
            write!(json, "{{\"atom\":{atom},\"counts\":")
                .expect("writing atom taxonomy cannot fail");
            push_taxonomy(&mut json, *taxonomy);
            json.push('}');
        }
        write!(
            json,
            concat!(
                "]}},\"decision\":{{\"structure_certificate_selected\":false,",
                "\"optimizer_change_authorized\":{},",
                "\"paid_scaling_authorized\":false}}}}\n"
            ),
            self.optimizer_change_authorized,
        )
        .expect("writing atomic structure decision cannot fail");
        json
    }
}

fn push_objective(json: &mut String, trace: &ProductionAtomicObjectiveTrace) {
    write!(
        json,
        concat!(
            "{{\"algorithm\":\"q47_logit_anchored\",\"fractional_bits\":{},",
            "\"reconstruction_verified\":{},\"environment_aggregation_verified\":{},",
            "\"vertex_losses\":["
        ),
        trace.fractional_bits,
        trace.reconstruction_verified,
        trace.environment_aggregation_verified,
    )
    .expect("writing atomic objective header cannot fail");
    push_u64(json, &trace.vertex_losses);
    json.push_str("],\"coefficients\":[");
    push_i128(json, &trace.coefficients);
    json.push_str("],\"documents\":[");
    for (index, document) in trace.documents.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"document\":{},\"coefficients\":[",
            document.document
        )
        .expect("writing atomic document cannot fail");
        push_i128(json, &document.coefficients);
        json.push_str("]}");
    }
    json.push_str("],\"absolute_mass_by_order\":{\"population\":[");
    push_u128(json, &trace.population_absolute_mass_by_order);
    json.push_str("],\"environmental\":[");
    push_u128(json, &trace.environmental_absolute_mass_by_order);
    json.push_str("]},\"tails\":[");
    for (index, tail) in trace.tails.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"retained_order\":{},\"population_absolute_tail\":\"{}\",",
                "\"environmental_absolute_tail\":\"{}\",",
                "\"cancellation_mass\":\"{}\",\"truncated_minimizer\":{},",
                "\"exact_gap\":\"{}\",\"tail_regret_bound\":\"{}\",",
                "\"certificate_verified\":{}}}"
            ),
            tail.retained_order,
            tail.population_absolute_tail,
            tail.environmental_absolute_tail,
            tail.cancellation_mass,
            tail.truncated_minimizer,
            tail.exact_gap,
            tail.tail_regret_bound,
            tail.certificate_verified,
        )
        .expect("writing interaction tail cannot fail");
    }
    json.push_str("],\"exchange_slices\":[");
    for (index, exchange) in trace.exchanges.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"cardinality\":{},\"uniform_defect\":\"{}\",",
                "\"exchange_local_minima\":{},\"maximum_local_gap\":\"{}\",",
                "\"cardinality_defect_bound\":\"{}\",",
                "\"certificate_verified\":{}}}"
            ),
            exchange.cardinality,
            exchange.uniform_defect,
            exchange.exchange_local_minima,
            exchange.maximum_local_gap,
            exchange.cardinality_defect_bound,
            exchange.certificate_verified,
        )
        .expect("writing exchange trace cannot fail");
    }
    let width = &trace.interaction_width;
    json.push_str("],\"interaction_width\":{\"support_hyperedges\":[");
    for (index, edge) in width.support_hyperedges.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{edge}").expect("writing support edge cannot fail");
    }
    write!(
        json,
        concat!(
            "],\"elimination_orders_evaluated\":{},\"best_induced_width\":{},",
            "\"best_order\":[{},{},{},{},{},{}],\"width_histogram\":["
        ),
        width.elimination_orders_evaluated,
        width.best_induced_width,
        width.best_order[0],
        width.best_order[1],
        width.best_order[2],
        width.best_order[3],
        width.best_order[4],
        width.best_order[5],
    )
    .expect("writing interaction width cannot fail");
    for (index, count) in width.width_histogram.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{count}").expect("writing width histogram cannot fail");
    }
    json.push_str("]}}");
}

fn push_moves(json: &mut String, moves: &[ProductionBooleanJetMove]) {
    json.push('[');
    for (index, movement) in moves.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"atom\":{},\"block\":\"{}\",\"group\":\"{}\",",
                "\"group_index\":{},\"coordinate\":{},\"parameter_delta\":{}}}"
            ),
            movement.canonical_order,
            movement.block,
            movement.group,
            movement.group_index,
            movement.coordinate,
            movement.parameter_delta,
        )
        .expect("writing atomic move cannot fail");
    }
    json.push(']');
}

fn push_u64(json: &mut String, values: &[u64]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{value}").expect("writing u64 cannot fail");
    }
}

fn push_i128(json: &mut String, values: &[i128]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"{value}\"").expect("writing i128 cannot fail");
    }
}

fn push_u128(json: &mut String, values: &[u128]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"{value}\"").expect("writing u128 cannot fail");
    }
}

fn push_concordance(json: &mut String, trace: ProductionRepresentationConcordance) {
    write!(
        json,
        concat!(
            "{{\"both_zero\":{},\"q20_zero_q32_nonzero\":{},",
            "\"q20_nonzero_q32_zero\":{},\"both_nonzero_sign_agree\":{},",
            "\"both_nonzero_sign_disagree\":{}}}"
        ),
        trace.both_zero,
        trace.q20_zero_q32_nonzero,
        trace.q20_nonzero_q32_zero,
        trace.both_nonzero_sign_agree,
        trace.both_nonzero_sign_disagree,
    )
    .expect("writing concordance cannot fail");
}

fn push_discrepancy(json: &mut String, trace: &ProductionRepresentationDiscrepancy) {
    write!(
        json,
        concat!(
            "{{\"q20_to_q32_multiplier\":{},\"residual_minimum\":\"{}\",",
            "\"residual_maximum\":\"{}\",\"residual_oscillation\":\"{}\",",
            "\"q20_minimizers\":["
        ),
        trace.q20_to_q32_multiplier,
        trace.residual_minimum,
        trace.residual_maximum,
        trace.residual_oscillation,
    )
    .expect("writing representation discrepancy cannot fail");
    push_usize(json, &trace.q20_minimizers);
    json.push_str("],\"q32_minimizers\":[");
    push_usize(json, &trace.q32_minimizers);
    json.push_str("],\"shared_minimizers\":[");
    push_usize(json, &trace.shared_minimizers);
    write!(
        json,
        concat!(
            "],\"maximum_q32_regret_of_q20_minimizer\":{},",
            "\"certificate_verified\":{}}}"
        ),
        trace.maximum_q32_regret_of_q20_minimizer, trace.certificate_verified,
    )
    .expect("writing representation discrepancy cannot fail");
}

fn push_usize(json: &mut String, values: &[usize]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "{value}").expect("writing usize cannot fail");
    }
}

fn push_taxonomy(json: &mut String, trace: ProductionBoundaryTaxonomy) {
    write!(
        json,
        concat!(
            "{{\"fine_grid_inactive\":{},\"phase_masked\":{},",
            "\"component_cancelled\":{},\"objective_visible\":{}}}"
        ),
        trace.fine_grid_inactive,
        trace.phase_masked,
        trace.component_cancelled,
        trace.objective_visible,
    )
    .expect("writing boundary taxonomy cannot fail");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_six_mobius_transform_reconstructs_exactly() {
        let losses = (0..VERTICES)
            .map(|mask| 10_000_u64 + (mask as u64 * 17) ^ (mask.count_ones() as u64 * 31))
            .collect::<Vec<_>>();
        let coefficients = mobius(&losses).expect("rank-six Möbius transform");
        assert!(reconstructs(&losses, &coefficients));
    }

    #[test]
    fn all_elimination_orders_are_audited_and_chain_width_is_one() {
        let mut coefficients = vec![0_i128; VERTICES];
        for variable in 0..RANK {
            coefficients[1 << variable] = variable as i128 + 1;
            if variable + 1 < RANK {
                coefficients[(1 << variable) | (1 << (variable + 1))] = -1;
            }
        }
        let width = interaction_width(&coefficients);
        assert_eq!(width.elimination_orders_evaluated, 720);
        assert_eq!(width.best_induced_width, 1);
        assert_eq!(width.width_histogram.iter().sum::<usize>(), 720);
    }

    #[test]
    fn modular_fixed_budget_slices_have_zero_exchange_defect() {
        let costs = [3_i64, -5, 2, 7, -1, 4];
        let losses = (0..VERTICES)
            .map(|mask| {
                let value = costs
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| mask & (1 << bit) != 0)
                    .map(|(_, value)| value)
                    .sum::<i64>();
                u64::try_from(value + 100).expect("shifted modular loss")
            })
            .collect::<Vec<_>>();
        for cardinality in 1..RANK {
            let trace = exchange_trace(&losses, cardinality).expect("exchange trace");
            assert_eq!(trace.uniform_defect, 0);
            assert_eq!(trace.maximum_local_gap, 0);
            assert!(trace.certificate_verified);
        }
    }

    #[test]
    fn concordance_separates_support_gain_from_sign_disagreement() {
        let mut q20 = vec![0_i128; VERTICES];
        let mut q32 = vec![0_i128; VERTICES];
        q20[1] = 1;
        q32[1] = 4096;
        q32[2] = -1;
        q20[3] = 1;
        q32[3] = -1;
        let trace = representation_concordance(&q20, &q32);
        assert_eq!(trace.q20_zero_q32_nonzero, 1);
        assert_eq!(trace.both_nonzero_sign_agree, 1);
        assert_eq!(trace.both_nonzero_sign_disagree, 1);
        assert_eq!(
            trace.both_zero
                + trace.q20_zero_q32_nonzero
                + trace.q20_nonzero_q32_zero
                + trace.both_nonzero_sign_agree
                + trace.both_nonzero_sign_disagree,
            63
        );
    }

    #[test]
    fn representation_discrepancy_uses_shared_scale_and_exact_oscillation() {
        let q20 = (0..VERTICES)
            .map(|mask| 100_u64 + (mask % 3) as u64)
            .collect::<Vec<_>>();
        let q32 = q20
            .iter()
            .enumerate()
            .map(|(mask, &loss)| loss * 4096 + (mask % 5) as u64)
            .collect::<Vec<_>>();
        let trace = representation_discrepancy(&q20, &q32).expect("representation discrepancy");
        assert_eq!(trace.residual_minimum, 0);
        assert_eq!(trace.residual_maximum, 4);
        assert_eq!(trace.residual_oscillation, 4);
        assert_eq!(trace.maximum_q32_regret_of_q20_minimizer, 4);
        assert!(trace.certificate_verified);
    }
}
