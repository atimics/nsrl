#![deny(unsafe_code)]

use core::cmp::Ordering;

pub const Q15_ONE: i32 = 32_768;
pub const DEFAULT_TRACE_DIM: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    TrainShard,
    GradientReduce,
    CheckpointPublish,
    DashboardSync,
    GenerationProbe,
    EvaluateCheckpoint,
    TerminateInstance,
}

impl TaskKind {
    pub const fn base_priority_q15(self) -> i32 {
        match self {
            Self::TerminateInstance => 50_000,
            Self::DashboardSync => 32_000,
            Self::CheckpointPublish => 29_000,
            Self::GradientReduce => 27_000,
            Self::EvaluateCheckpoint => 24_000,
            Self::GenerationProbe => 18_000,
            Self::TrainShard => 16_000,
        }
    }

    const fn action_seed(self) -> u64 {
        match self {
            Self::TrainShard => 0x9e37_0001,
            Self::GradientReduce => 0x9e37_0002,
            Self::CheckpointPublish => 0x9e37_0003,
            Self::DashboardSync => 0x9e37_0004,
            Self::GenerationProbe => 0x9e37_0005,
            Self::EvaluateCheckpoint => 0x9e37_0006,
            Self::TerminateInstance => 0x9e37_0007,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Component {
    Output,
    Mlp,
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
}

impl Component {
    const fn seed(self) -> u64 {
        match self {
            Self::Output => 0xa511_0001,
            Self::Mlp => 0xa511_0002,
            Self::Embedding => 0xa511_0003,
            Self::AttentionQ => 0xa511_0004,
            Self::AttentionK => 0xa511_0005,
            Self::AttentionV => 0xa511_0006,
            Self::AttentionO => 0xa511_0007,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentDeltas {
    pub output: u64,
    pub mlp: u64,
    pub embedding: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
}

impl ComponentDeltas {
    pub fn get(self, component: Component) -> u64 {
        match component {
            Component::Output => self.output,
            Component::Mlp => self.mlp,
            Component::Embedding => self.embedding,
            Component::AttentionQ => self.q,
            Component::AttentionK => self.k,
            Component::AttentionV => self.v,
            Component::AttentionO => self.o,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRef {
    pub name: String,
    pub bytes: u64,
}

impl ArtifactRef {
    pub fn new(name: impl Into<String>, bytes: u64) -> Self {
        Self {
            name: name.into(),
            bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceEstimate {
    pub min_cores: u16,
    pub memory_bytes: u64,
    pub cache_bytes: u64,
    pub expected_micros: u64,
}

impl ResourceEstimate {
    pub const fn new(
        min_cores: u16,
        memory_bytes: u64,
        cache_bytes: u64,
        expected_micros: u64,
    ) -> Self {
        Self {
            min_cores,
            memory_bytes,
            cache_bytes,
            expected_micros,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroTask {
    pub id: String,
    pub kind: TaskKind,
    pub component: Option<Component>,
    pub resources: ResourceEstimate,
    pub inputs: Vec<ArtifactRef>,
    pub outputs: Vec<ArtifactRef>,
    pub priority_q15: i32,
    pub attempts: u16,
    pub status: TaskStatus,
}

impl MicroTask {
    pub fn new(id: impl Into<String>, kind: TaskKind, resources: ResourceEstimate) -> Self {
        Self {
            id: id.into(),
            kind,
            component: None,
            resources,
            inputs: Vec::new(),
            outputs: Vec::new(),
            priority_q15: 0,
            attempts: 0,
            status: TaskStatus::Pending,
        }
    }

    pub fn with_component(mut self, component: Component) -> Self {
        self.component = Some(component);
        self
    }

    pub fn with_priority_q15(mut self, priority_q15: i32) -> Self {
        self.priority_q15 = priority_q15;
        self
    }

    pub fn with_input(mut self, artifact: ArtifactRef) -> Self {
        self.inputs.push(artifact);
        self
    }

    pub fn with_output(mut self, artifact: ArtifactRef) -> Self {
        self.outputs.push(artifact);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arm64Machine {
    pub label: String,
    pub cores: u16,
    pub l1d_per_core_bytes: u64,
    pub shared_cache_bytes: u64,
    pub memory_bytes: u64,
    pub hourly_micro_usd: u64,
}

impl Arm64Machine {
    pub fn c8g_xlarge() -> Self {
        Self {
            label: "c8g.xlarge".to_owned(),
            cores: 4,
            l1d_per_core_bytes: 64 * 1024,
            shared_cache_bytes: 4 * 1024 * 1024,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            hourly_micro_usd: 159_520,
        }
    }

    pub fn c8g_4xlarge() -> Self {
        Self {
            label: "c8g.4xlarge".to_owned(),
            cores: 16,
            l1d_per_core_bytes: 64 * 1024,
            shared_cache_bytes: 16 * 1024 * 1024,
            memory_bytes: 32 * 1024 * 1024 * 1024,
            hourly_micro_usd: 638_080,
        }
    }

    pub fn c8g_16xlarge() -> Self {
        Self {
            label: "c8g.16xlarge".to_owned(),
            cores: 64,
            l1d_per_core_bytes: 64 * 1024,
            shared_cache_bytes: 64 * 1024 * 1024,
            memory_bytes: 128 * 1024 * 1024 * 1024,
            hourly_micro_usd: 2_552_320,
        }
    }

    pub fn suggested_parallelism(&self, task: &MicroTask) -> u16 {
        let core_limit = self.cores.max(1);
        let cache_per_task = task.resources.cache_bytes.max(1);
        let cache_limit = (self.shared_cache_bytes / cache_per_task).clamp(1, u64::from(u16::MAX));
        core_limit.min(cache_limit as u16).max(1)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerObservation {
    pub active_workers: u16,
    pub pending_tasks: u16,
    pub artifact_backlog: u16,
    pub dashboard_staleness_ms: u32,
    pub rollback_rate_q15: i16,
    pub invalid_forward_count: u16,
    pub zero_delta_ratio_q15: i16,
    pub phase_per_mille: u16,
    pub cost_spent_micro_usd: u64,
    pub component_delta_l1: ComponentDeltas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub max_attempts: u16,
    pub dashboard_stale_after_ms: u32,
    pub cache_soft_limit_q15: i32,
    pub residual_trace_weight_q15: i32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            dashboard_stale_after_ms: 30_000,
            cache_soft_limit_q15: Q15_ONE,
            residual_trace_weight_q15: 8_192,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleDecision {
    pub task_index: usize,
    pub task_id: String,
    pub score_q15: i64,
    pub baseline_score_q15: i64,
    pub residual_score_q15: i64,
    pub suggested_parallelism: u16,
    pub reason: ScheduleReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleReason {
    Baseline,
    ResidualTraceBoost,
    DashboardStale,
    TerminationSafety,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfeasibleReason {
    NotPending,
    AttemptsExceeded,
    NotEnoughCores,
    NotEnoughMemory,
    CacheOversubscribed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskScore {
    pub task_index: usize,
    pub feasible: bool,
    pub infeasible_reason: Option<InfeasibleReason>,
    pub baseline_score_q15: i64,
    pub residual_score_q15: i64,
    pub total_score_q15: i64,
    pub suggested_parallelism: u16,
    pub reason: ScheduleReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scheduler<const D: usize = DEFAULT_TRACE_DIM> {
    pub config: SchedulerConfig,
    pub trace: ResidualTrace<D>,
}

impl<const D: usize> Scheduler<D> {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            trace: ResidualTrace::default(),
        }
    }

    pub fn score_tasks(
        &self,
        machine: &Arm64Machine,
        observation: SchedulerObservation,
        tasks: &[MicroTask],
    ) -> Vec<TaskScore> {
        tasks
            .iter()
            .enumerate()
            .map(|(task_index, task)| self.score_task(machine, observation, task_index, task))
            .collect()
    }

    pub fn select(
        &self,
        machine: &Arm64Machine,
        observation: SchedulerObservation,
        tasks: &[MicroTask],
    ) -> Option<ScheduleDecision> {
        self.score_tasks(machine, observation, tasks)
            .into_iter()
            .filter(|score| score.feasible)
            .max_by(compare_task_score)
            .map(|score| ScheduleDecision {
                task_index: score.task_index,
                task_id: tasks[score.task_index].id.clone(),
                score_q15: score.total_score_q15,
                baseline_score_q15: score.baseline_score_q15,
                residual_score_q15: score.residual_score_q15,
                suggested_parallelism: score.suggested_parallelism,
                reason: score.reason,
            })
    }

    pub fn observe_outcome(
        &mut self,
        observation: SchedulerObservation,
        task: &MicroTask,
        reward_q15: i16,
    ) {
        self.trace.remember(observation, task, reward_q15);
    }

    fn score_task(
        &self,
        machine: &Arm64Machine,
        observation: SchedulerObservation,
        task_index: usize,
        task: &MicroTask,
    ) -> TaskScore {
        let infeasible_reason = self.infeasible_reason(machine, task);
        let feasible = infeasible_reason.is_none();
        let suggested_parallelism = machine.suggested_parallelism(task);
        let baseline_score_q15 = self.baseline_score_q15(machine, observation, task);
        let raw_residual = self.trace.score(observation, task);
        let residual_score_q15 =
            (raw_residual.saturating_mul(i64::from(self.config.residual_trace_weight_q15))) >> 15;
        let total_score_q15 = if feasible {
            baseline_score_q15.saturating_add(residual_score_q15)
        } else {
            i64::MIN / 4
        };
        let reason = if task.kind == TaskKind::TerminateInstance {
            ScheduleReason::TerminationSafety
        } else if task.kind == TaskKind::DashboardSync
            && observation.dashboard_staleness_ms >= self.config.dashboard_stale_after_ms
        {
            ScheduleReason::DashboardStale
        } else if residual_score_q15 > Q15_ONE as i64 {
            ScheduleReason::ResidualTraceBoost
        } else {
            ScheduleReason::Baseline
        };
        TaskScore {
            task_index,
            feasible,
            infeasible_reason,
            baseline_score_q15,
            residual_score_q15,
            total_score_q15,
            suggested_parallelism,
            reason,
        }
    }

    fn infeasible_reason(
        &self,
        machine: &Arm64Machine,
        task: &MicroTask,
    ) -> Option<InfeasibleReason> {
        if task.status != TaskStatus::Pending {
            return Some(InfeasibleReason::NotPending);
        }
        if task.attempts >= self.config.max_attempts {
            return Some(InfeasibleReason::AttemptsExceeded);
        }
        if task.resources.min_cores > machine.cores {
            return Some(InfeasibleReason::NotEnoughCores);
        }
        if task.resources.memory_bytes > machine.memory_bytes {
            return Some(InfeasibleReason::NotEnoughMemory);
        }
        let cache_limit = (u128::from(machine.shared_cache_bytes)
            * self.config.cache_soft_limit_q15.max(1) as u128)
            / Q15_ONE as u128;
        if u128::from(task.resources.cache_bytes) > cache_limit {
            return Some(InfeasibleReason::CacheOversubscribed);
        }
        None
    }

    fn baseline_score_q15(
        &self,
        machine: &Arm64Machine,
        observation: SchedulerObservation,
        task: &MicroTask,
    ) -> i64 {
        let mut score = i64::from(task.kind.base_priority_q15() + task.priority_q15);
        score -= i64::from(task.attempts) * 4_096;

        if let Some(component) = task.component {
            score += component_pressure_q15(component, observation.component_delta_l1);
        }

        match task.kind {
            TaskKind::DashboardSync => {
                let stale = observation.dashboard_staleness_ms;
                if stale >= self.config.dashboard_stale_after_ms {
                    score += 36_000;
                } else {
                    score += i64::from(stale / 4);
                }
            }
            TaskKind::GradientReduce | TaskKind::CheckpointPublish => {
                score += i64::from(observation.artifact_backlog) * 2_048;
            }
            TaskKind::TrainShard => {
                let suggested = machine.suggested_parallelism(task);
                if observation.active_workers >= suggested {
                    score -= 24_000;
                }
                score -= i64::from(observation.rollback_rate_q15.max(0)) / 2;
                score -= i64::from(observation.invalid_forward_count) * 8_192;
                score += i64::from(observation.zero_delta_ratio_q15.max(0)) / 4;
                if observation.phase_per_mille < 250 {
                    score += 3_000;
                }
            }
            TaskKind::TerminateInstance => {
                if observation.pending_tasks == 0 && observation.active_workers == 0 {
                    score += 64_000;
                } else {
                    score -= 64_000;
                }
            }
            TaskKind::EvaluateCheckpoint | TaskKind::GenerationProbe => {
                if observation.artifact_backlog > 0 {
                    score += 4_096;
                }
            }
        }

        let cache_penalty = cache_penalty_q15(machine, task);
        score.saturating_sub(cache_penalty)
    }
}

impl<const D: usize> Default for Scheduler<D> {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

fn compare_task_score(left: &TaskScore, right: &TaskScore) -> Ordering {
    left.total_score_q15
        .cmp(&right.total_score_q15)
        .then_with(|| right.task_index.cmp(&left.task_index))
}

fn component_pressure_q15(component: Component, deltas: ComponentDeltas) -> i64 {
    let value = deltas.get(component);
    if value == 0 {
        return 0;
    }
    let leading = value.leading_zeros() as i64;
    let magnitude = 64_i64.saturating_sub(leading);
    (magnitude * 384).min(8_192)
}

fn cache_penalty_q15(machine: &Arm64Machine, task: &MicroTask) -> i64 {
    if task.resources.cache_bytes <= machine.l1d_per_core_bytes {
        return 0;
    }
    let excess = task
        .resources
        .cache_bytes
        .saturating_sub(machine.l1d_per_core_bytes);
    let units = excess / machine.l1d_per_core_bytes.max(1);
    (units as i64).saturating_mul(512).min(16_384)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualTrace<const D: usize = DEFAULT_TRACE_DIM> {
    memory: [i64; D],
    stores: u64,
}

impl<const D: usize> ResidualTrace<D> {
    pub fn remember(
        &mut self,
        observation: SchedulerObservation,
        task: &MicroTask,
        reward_q15: i16,
    ) {
        if D == 0 || reward_q15 == 0 {
            return;
        }
        for dim in 0..D {
            let feature = trace_feature_q15(observation, task, dim as u64);
            let delta = (i64::from(feature) * i64::from(reward_q15)) >> 8;
            self.memory[dim] = self.memory[dim].saturating_add(delta);
        }
        self.stores = self.stores.saturating_add(1);
    }

    pub fn score(&self, observation: SchedulerObservation, task: &MicroTask) -> i64 {
        if D == 0 || self.stores == 0 {
            return 0;
        }
        let mut acc = 0_i128;
        for dim in 0..D {
            let feature = trace_feature_q15(observation, task, dim as u64);
            acc += i128::from(self.memory[dim]) * i128::from(feature);
        }
        let normalized = acc / i128::from(D as i64);
        clamp_i128_to_i64(normalized >> 22)
    }

    pub fn stores(&self) -> u64 {
        self.stores
    }

    pub fn memory(&self) -> &[i64; D] {
        &self.memory
    }
}

impl<const D: usize> Default for ResidualTrace<D> {
    fn default() -> Self {
        Self {
            memory: [0; D],
            stores: 0,
        }
    }
}

fn trace_feature_q15(observation: SchedulerObservation, task: &MicroTask, dim: u64) -> i16 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash = mix_u64(hash ^ task.kind.action_seed());
    hash = mix_u64(hash ^ (dim.wrapping_mul(0x9e37_79b9_7f4a_7c15)));
    hash = mix_u64(hash ^ u64::from(observation.active_workers));
    hash = mix_u64(hash ^ (u64::from(observation.pending_tasks) << 7));
    hash = mix_u64(hash ^ (u64::from(observation.artifact_backlog) << 13));
    hash = mix_u64(hash ^ (u64::from(observation.dashboard_staleness_ms / 1_000) << 19));
    hash = mix_u64(hash ^ (observation.rollback_rate_q15 as i64 as u64).rotate_left(11));
    hash = mix_u64(hash ^ (u64::from(observation.invalid_forward_count) << 23));
    hash = mix_u64(hash ^ (observation.zero_delta_ratio_q15 as i64 as u64).rotate_left(29));
    hash = mix_u64(hash ^ (u64::from(observation.phase_per_mille) << 31));
    if let Some(component) = task.component {
        hash = mix_u64(hash ^ component.seed());
        hash = mix_u64(hash ^ observation.component_delta_l1.get(component));
    }
    let magnitude = 8_192_i16 + ((hash >> 48) as i16 & 0x1fff);
    if hash & 1 == 0 { magnitude } else { -magnitude }
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn clamp_i128_to_i64(value: i128) -> i64 {
    if value > i128::from(i64::MAX) {
        i64::MAX
    } else if value < i128::from(i64::MIN) {
        i64::MIN
    } else {
        value as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_task(id: &str, component: Component) -> MicroTask {
        MicroTask::new(
            id,
            TaskKind::TrainShard,
            ResourceEstimate::new(1, 512 * 1024 * 1024, 2 * 1024 * 1024, 2_000_000),
        )
        .with_component(component)
    }

    #[test]
    fn stale_dashboard_preempts_train_shards() {
        let scheduler = Scheduler::<16>::default();
        let machine = Arm64Machine::c8g_xlarge();
        let observation = SchedulerObservation {
            pending_tasks: 2,
            dashboard_staleness_ms: 60_000,
            ..SchedulerObservation::default()
        };
        let tasks = vec![
            train_task("train-q", Component::AttentionQ),
            MicroTask::new(
                "sync-dashboard",
                TaskKind::DashboardSync,
                ResourceEstimate::new(1, 64 * 1024 * 1024, 256 * 1024, 100_000),
            ),
        ];

        let decision = scheduler
            .select(&machine, observation, &tasks)
            .expect("decision");

        assert_eq!(decision.task_id, "sync-dashboard");
        assert_eq!(decision.reason, ScheduleReason::DashboardStale);
    }

    #[test]
    fn infeasible_tasks_are_skipped() {
        let scheduler = Scheduler::<16>::default();
        let machine = Arm64Machine::c8g_xlarge();
        let observation = SchedulerObservation::default();
        let tasks = vec![
            MicroTask::new(
                "too-big",
                TaskKind::TrainShard,
                ResourceEstimate::new(1, machine.memory_bytes + 1, 1024, 100),
            ),
            MicroTask::new(
                "fits",
                TaskKind::GenerationProbe,
                ResourceEstimate::new(1, 64 * 1024 * 1024, 1024, 100),
            ),
        ];

        let scores = scheduler.score_tasks(&machine, observation, &tasks);
        assert_eq!(
            scores[0].infeasible_reason,
            Some(InfeasibleReason::NotEnoughMemory)
        );
        assert_eq!(
            scheduler
                .select(&machine, observation, &tasks)
                .expect("decision")
                .task_id,
            "fits"
        );
    }

    #[test]
    fn arm64_profile_caps_parallelism_by_cache_footprint() {
        let machine = Arm64Machine::c8g_4xlarge();
        let small = MicroTask::new(
            "small",
            TaskKind::TrainShard,
            ResourceEstimate::new(1, 1_000, 512 * 1024, 1_000),
        );
        let large = MicroTask::new(
            "large",
            TaskKind::TrainShard,
            ResourceEstimate::new(1, 1_000, 8 * 1024 * 1024, 1_000),
        );

        assert_eq!(machine.suggested_parallelism(&small), 16);
        assert_eq!(machine.suggested_parallelism(&large), 2);
    }

    #[test]
    fn residual_trace_can_boost_a_previously_successful_task() {
        let mut scheduler = Scheduler::<32>::new(SchedulerConfig {
            residual_trace_weight_q15: Q15_ONE,
            ..SchedulerConfig::default()
        });
        let machine = Arm64Machine::c8g_xlarge();
        let observation = SchedulerObservation {
            active_workers: 2,
            pending_tasks: 2,
            phase_per_mille: 500,
            component_delta_l1: ComponentDeltas {
                q: 10,
                k: 10,
                ..ComponentDeltas::default()
            },
            ..SchedulerObservation::default()
        };
        let q = train_task("q-shard", Component::AttentionQ).with_priority_q15(-8_000);
        let k = train_task("k-shard", Component::AttentionK);
        let tasks = vec![q.clone(), k];

        assert_eq!(
            scheduler
                .select(&machine, observation, &tasks)
                .expect("initial decision")
                .task_id,
            "k-shard"
        );

        for _ in 0..16 {
            scheduler.observe_outcome(observation, &q, 24_000);
        }

        let decision = scheduler
            .select(&machine, observation, &tasks)
            .expect("boosted decision");
        assert_eq!(decision.task_id, "q-shard");
        assert_eq!(decision.reason, ScheduleReason::ResidualTraceBoost);
    }

    #[test]
    fn negative_residual_trace_suppresses_a_noisy_task() {
        let mut scheduler = Scheduler::<32>::new(SchedulerConfig {
            residual_trace_weight_q15: Q15_ONE,
            ..SchedulerConfig::default()
        });
        let machine = Arm64Machine::c8g_xlarge();
        let observation = SchedulerObservation {
            pending_tasks: 2,
            artifact_backlog: 1,
            ..SchedulerObservation::default()
        };
        let noisy = MicroTask::new(
            "probe-noisy",
            TaskKind::GenerationProbe,
            ResourceEstimate::new(1, 64 * 1024 * 1024, 128 * 1024, 100),
        )
        .with_priority_q15(18_000);
        let publish = MicroTask::new(
            "publish",
            TaskKind::CheckpointPublish,
            ResourceEstimate::new(1, 64 * 1024 * 1024, 128 * 1024, 100),
        );
        let tasks = vec![noisy.clone(), publish];

        assert_eq!(
            scheduler
                .select(&machine, observation, &tasks)
                .expect("initial decision")
                .task_id,
            "probe-noisy"
        );
        for _ in 0..24 {
            scheduler.observe_outcome(observation, &noisy, -24_000);
        }

        assert_eq!(
            scheduler
                .select(&machine, observation, &tasks)
                .expect("suppressed decision")
                .task_id,
            "publish"
        );
    }
}
