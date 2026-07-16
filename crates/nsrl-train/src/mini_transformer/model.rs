//! Model, optimizer, worker, and swarm artifact serialization.

use super::*;

impl MiniTransformerMlpModel {
    pub fn new_initial() -> Self {
        Self::new_initial_with_seq_len(DEFAULT_MINI_TRANSFORMER_SEQ_LEN)
    }

    pub fn new_initial_with_seq_len(context_seq_len: usize) -> Self {
        Self::new_initial_with_seq_len_and_layers(context_seq_len, DEFAULT_MINI_TRANSFORMER_LAYERS)
            .expect("default mini transformer layer count should be valid")
    }

    pub fn new_initial_with_seq_len_and_layers(
        context_seq_len: usize,
        layers: usize,
    ) -> Result<Self, TrainError> {
        if layers == 0 {
            return Err(TrainError::InvalidModel("bad mini transformer layer count"));
        }
        Self {
            context_seq_len,
            embeddings: initial_mini_transformer_embeddings(),
            position_embeddings: initial_mini_transformer_position_embeddings(context_seq_len),
            attention_rms_weights: Vec::new(),
            mlp_rms_weights: Vec::new(),
            q_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            k_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            v_weights: stack_i8_layers_with_active_final(
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            o_weights: stack_i8_layers_with_active_final(
                vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL],
                identity_i8_matrix(MINI_TRANSFORMER_D_MODEL),
                layers,
            ),
            up_weights: stack_i8_layers_with_active_final(
                initial_mini_transformer_mlp_up_weights(),
                initial_mini_transformer_mlp_up_weights(),
                layers,
            ),
            gate_weights: stack_i8_layers_with_active_final(
                initial_mini_transformer_mlp_gate_weights(),
                initial_mini_transformer_mlp_gate_weights(),
                layers,
            ),
            down_weights: stack_i8_layers_with_active_final(
                vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL],
                initial_mini_transformer_mlp_down_weights(),
                layers,
            ),
            output_weights: initial_mini_transformer_output_weights(),
        }
        .validate()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_seq_len: usize,
        embeddings: Vec<i16>,
        position_embeddings: Vec<i16>,
        q_weights: Vec<i8>,
        k_weights: Vec<i8>,
        v_weights: Vec<i8>,
        o_weights: Vec<i8>,
        up_weights: Vec<i8>,
        gate_weights: Vec<i8>,
        down_weights: Vec<i8>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        Self::new_with_rms_weights(
            context_seq_len,
            embeddings,
            position_embeddings,
            Vec::new(),
            Vec::new(),
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_rms_weights(
        context_seq_len: usize,
        embeddings: Vec<i16>,
        position_embeddings: Vec<i16>,
        attention_rms_weights: Vec<i16>,
        mlp_rms_weights: Vec<i16>,
        q_weights: Vec<i8>,
        k_weights: Vec<i8>,
        v_weights: Vec<i8>,
        o_weights: Vec<i8>,
        up_weights: Vec<i8>,
        gate_weights: Vec<i8>,
        down_weights: Vec<i8>,
        output_weights: Vec<i8>,
    ) -> Result<Self, TrainError> {
        if context_seq_len == 0 {
            return Err(TrainError::InvalidModel(
                "bad mini transformer context seq_len",
            ));
        }
        if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer embedding count",
            ));
        }
        if position_embeddings.len()
            != context_seq_len
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidModel(
                    "mini transformer position count overflow",
                ))?
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer position embedding count",
            ));
        }
        let attention_weight_count = mini_transformer_attention_weight_count()?;
        let mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        let attention_layers = infer_layer_count(q_weights.len(), attention_weight_count).ok_or(
            TrainError::InvalidModel("wrong mini transformer attention weight count"),
        )?;
        let mlp_layers = infer_layer_count(up_weights.len(), mlp_up_or_gate_count).ok_or(
            TrainError::InvalidModel("wrong mini transformer up/gate weight count"),
        )?;
        if attention_layers == 0 || mlp_layers == 0 || attention_layers != mlp_layers {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer layer count",
            ));
        }
        let expected_rms_weight_count = attention_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm weight count overflow"))?;
        let rms_disabled = attention_rms_weights.is_empty() && mlp_rms_weights.is_empty();
        let rms_enabled = attention_rms_weights.len() == expected_rms_weight_count
            && mlp_rms_weights.len() == expected_rms_weight_count;
        if !rms_disabled && !rms_enabled {
            return Err(TrainError::InvalidModel("wrong RMSNorm weight count"));
        }
        if k_weights.len() != q_weights.len()
            || v_weights.len() != q_weights.len()
            || o_weights.len() != q_weights.len()
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer attention weight count",
            ));
        }
        if gate_weights.len() != up_weights.len() {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer up/gate weight count",
            ));
        }
        if down_weights.len() != mlp_layers * mlp_down_count {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer down weight count",
            ));
        }
        if output_weights.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer output weight count",
            ));
        }

        Ok(Self {
            context_seq_len,
            embeddings,
            position_embeddings,
            attention_rms_weights,
            mlp_rms_weights,
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        })
    }

    fn validate(self) -> Result<Self, TrainError> {
        Self::new_with_rms_weights(
            self.context_seq_len,
            self.embeddings,
            self.position_embeddings,
            self.attention_rms_weights,
            self.mlp_rms_weights,
            self.q_weights,
            self.k_weights,
            self.v_weights,
            self.o_weights,
            self.up_weights,
            self.gate_weights,
            self.down_weights,
            self.output_weights,
        )
    }

    pub fn transformer_layers(&self) -> usize {
        let Ok(attention_weight_count) = mini_transformer_attention_weight_count() else {
            return 0;
        };
        let Some(attention_layers) =
            infer_layer_count(self.q_weights.len(), attention_weight_count)
        else {
            return 0;
        };
        attention_layers
    }

    pub(super) fn checked_transformer_layers(&self) -> Result<usize, TrainError> {
        let layers = self.transformer_layers();
        if layers == 0 {
            return Err(TrainError::InvalidModel("bad mini transformer layer count"));
        }
        let attention_weight_count = mini_transformer_attention_weight_count()?;
        let mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        if self.k_weights.len() != self.q_weights.len()
            || self.v_weights.len() != self.q_weights.len()
            || self.o_weights.len() != self.q_weights.len()
            || self.up_weights.len() != layers * mlp_up_or_gate_count
            || self.gate_weights.len() != self.up_weights.len()
            || self.down_weights.len() != layers * mlp_down_count
            || self.q_weights.len() != layers * attention_weight_count
            || (!self.attention_rms_weights.is_empty()
                && (self.attention_rms_weights.len() != layers * MINI_TRANSFORMER_D_MODEL
                    || self.mlp_rms_weights.len() != layers * MINI_TRANSFORMER_D_MODEL))
            || (self.attention_rms_weights.is_empty() != self.mlp_rms_weights.is_empty())
        {
            return Err(TrainError::InvalidModel(
                "wrong mini transformer layer tensor count",
            ));
        }
        Ok(layers)
    }

    pub fn rms_norm_enabled(&self) -> bool {
        !self.attention_rms_weights.is_empty()
    }

    pub fn enable_rms_norm(&mut self) -> Result<(), TrainError> {
        if self.rms_norm_enabled() {
            return Ok(());
        }
        self.enable_rms_norm_with_gamma(DEFAULT_MINI_TRANSFORMER_RMS_GAMMA_Q15)
    }

    pub fn enable_rms_norm_with_gamma(&mut self, gamma_q15: i16) -> Result<(), TrainError> {
        if gamma_q15 <= 0 {
            return Err(TrainError::InvalidConfig);
        }
        let layers = self.checked_transformer_layers()?;
        if self.rms_norm_enabled() {
            if self
                .attention_rms_weights
                .iter()
                .chain(self.mlp_rms_weights.iter())
                .all(|&weight| weight == gamma_q15)
            {
                return Ok(());
            }
            return Err(TrainError::InvalidConfig);
        }
        let count = layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm weight count overflow"))?;
        self.attention_rms_weights = vec![gamma_q15; count];
        self.mlp_rms_weights = vec![gamma_q15; count];
        Ok(())
    }

    pub(super) fn rms_weight_range(&self, layer_index: usize) -> Result<Range<usize>, TrainError> {
        if !self.rms_norm_enabled() || layer_index >= self.checked_transformer_layers()? {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, MINI_TRANSFORMER_D_MODEL)
    }

    pub(super) fn attention_weight_range(
        &self,
        layer_index: usize,
    ) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_attention_weight_count()?)
    }

    pub(super) fn mlp_up_or_gate_weight_range(
        &self,
        layer_index: usize,
    ) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_mlp_up_or_gate_weight_count()?)
    }

    pub(super) fn mlp_down_weight_range(
        &self,
        layer_index: usize,
    ) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        if layer_index >= layers {
            return Err(TrainError::InvalidConfig);
        }
        mini_transformer_layer_range(layer_index, mini_transformer_mlp_down_weight_count()?)
    }

    pub(super) fn final_attention_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.attention_weight_range(layers - 1)
    }

    pub(super) fn final_mlp_up_or_gate_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.mlp_up_or_gate_weight_range(layers - 1)
    }

    pub(super) fn final_mlp_down_weight_range(&self) -> Result<Range<usize>, TrainError> {
        let layers = self.checked_transformer_layers()?;
        self.mlp_down_weight_range(layers - 1)
    }

    pub fn embedding_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i16_slice(&self.position_embeddings);
        hasher.finish()
    }

    pub fn attention_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_i8_slice(&self.q_weights);
        hasher.update_i8_slice(&self.k_weights);
        hasher.update_i8_slice(&self.v_weights);
        hasher.update_i8_slice(&self.o_weights);
        hasher.finish()
    }

    pub fn attention_q_hash(&self) -> u64 {
        hash_i8_slice(&self.q_weights)
    }

    pub fn attention_k_hash(&self) -> u64 {
        hash_i8_slice(&self.k_weights)
    }

    pub fn attention_v_hash(&self) -> u64 {
        hash_i8_slice(&self.v_weights)
    }

    pub fn attention_o_hash(&self) -> u64 {
        hash_i8_slice(&self.o_weights)
    }

    pub fn mlp_hash(&self) -> u64 {
        hash_three_i8_slices(&self.up_weights, &self.gate_weights, &self.down_weights)
    }

    pub fn output_head_hash(&self) -> u64 {
        hash_i8_slice(&self.output_weights)
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.context_seq_len);
        hasher.update_i16_slice(&self.embeddings);
        hasher.update_i16_slice(&self.position_embeddings);
        if self.rms_norm_enabled() {
            hasher.update_i16_slice(&self.attention_rms_weights);
            hasher.update_i16_slice(&self.mlp_rms_weights);
        }
        hasher.update_i8_slice(&self.q_weights);
        hasher.update_i8_slice(&self.k_weights);
        hasher.update_i8_slice(&self.v_weights);
        hasher.update_i8_slice(&self.o_weights);
        hasher.update_i8_slice(&self.up_weights);
        hasher.update_i8_slice(&self.gate_weights);
        hasher.update_i8_slice(&self.down_weights);
        hasher.update_i8_slice(&self.output_weights);
        hasher.finish()
    }

    pub fn optimizer_parameter_count(&self) -> Result<usize, TrainError> {
        [
            self.embeddings.len(),
            self.position_embeddings.len(),
            self.attention_rms_weights.len(),
            self.mlp_rms_weights.len(),
            self.q_weights.len(),
            self.k_weights.len(),
            self.v_weights.len(),
            self.o_weights.len(),
            self.up_weights.len(),
            self.gate_weights.len(),
            self.down_weights.len(),
            self.output_weights.len(),
        ]
        .into_iter()
        .try_fold(0_usize, |total, count| {
            total.checked_add(count).ok_or(TrainError::InvalidModel(
                "optimizer parameter count overflow",
            ))
        })
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        let embedding_bytes = checked_i16_tensor_bytes(
            self.embeddings.len(),
            "mini transformer embedding bytes overflow",
        )?;
        let position_embedding_bytes = checked_i16_tensor_bytes(
            self.position_embeddings.len(),
            "mini transformer position embedding bytes overflow",
        )?;
        let attention_rms_bytes = checked_i16_tensor_bytes(
            self.attention_rms_weights.len(),
            "mini transformer attention RMS bytes overflow",
        )?;
        let mlp_rms_bytes = checked_i16_tensor_bytes(
            self.mlp_rms_weights.len(),
            "mini transformer MLP RMS bytes overflow",
        )?;
        let weight_bytes = checked_model_capacity(
            0,
            &[
                self.q_weights.len(),
                self.k_weights.len(),
                self.v_weights.len(),
                self.o_weights.len(),
                self.up_weights.len(),
                self.gate_weights.len(),
                self.down_weights.len(),
                self.output_weights.len(),
            ],
        )?;
        let mut out = Vec::with_capacity(checked_model_capacity(
            136,
            &[
                embedding_bytes,
                position_embedding_bytes,
                weight_bytes,
                attention_rms_bytes,
                mlp_rms_bytes,
            ],
        )?);
        out.extend_from_slice(MINI_TRANSFORMER_MODEL_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(
                MINI_TRANSFORMER_D_MODEL,
                "mini transformer d_model exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "mini transformer heads exceeds u32")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                MINI_TRANSFORMER_HIDDEN_DIM,
                "mini transformer hidden_dim exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.context_seq_len,
                "mini transformer context_seq_len exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.embeddings.len(),
                "mini transformer embedding count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.position_embeddings.len(),
                "mini transformer position embedding count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.q_weights.len(), "mini transformer q count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.k_weights.len(), "mini transformer k count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.v_weights.len(), "mini transformer v count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(self.o_weights.len(), "mini transformer o count exceeds u64")?
                .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.up_weights.len(),
                "mini transformer up count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.gate_weights.len(),
                "mini transformer gate count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.down_weights.len(),
                "mini transformer down count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u64(
                self.output_weights.len(),
                "mini transformer output count exceeds u64",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(&self.embedding_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_q_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_k_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_v_hash().to_le_bytes());
        out.extend_from_slice(&self.attention_o_hash().to_le_bytes());
        out.extend_from_slice(&self.mlp_hash().to_le_bytes());
        out.extend_from_slice(&self.output_head_hash().to_le_bytes());
        out.extend_from_slice(&self.model_hash().to_le_bytes());
        for &embedding in self.embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        for &embedding in self.position_embeddings.iter() {
            out.extend_from_slice(&embedding.to_le_bytes());
        }
        out.extend(self.q_weights.iter().map(|&weight| weight as u8));
        out.extend(self.k_weights.iter().map(|&weight| weight as u8));
        out.extend(self.v_weights.iter().map(|&weight| weight as u8));
        out.extend(self.o_weights.iter().map(|&weight| weight as u8));
        out.extend(self.up_weights.iter().map(|&weight| weight as u8));
        out.extend(self.gate_weights.iter().map(|&weight| weight as u8));
        out.extend(self.down_weights.iter().map(|&weight| weight as u8));
        out.extend(self.output_weights.iter().map(|&weight| weight as u8));
        for &weight in &self.attention_rms_weights {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        for &weight in &self.mlp_rms_weights {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer model should fit on-disk format")
    }

    /// Returns the model hash recorded in a serialized V4 or V5 artifact.
    ///
    /// This is intentionally separate from [`Self::model_hash`]: loading a
    /// historical 32-wide V4 artifact upgrades it to the current geometry, so
    /// the in-memory model has a new hash after the source artifact has been
    /// authenticated.
    pub fn serialized_model_hash(bytes: &[u8]) -> Result<u64, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        let magic = &bytes[..MINI_TRANSFORMER_MODEL_MAGIC.len()];
        if magic != MINI_TRANSFORMER_MODEL_MAGIC && magic != MINI_TRANSFORMER_LEGACY_MODEL_MAGIC {
            return Err(TrainError::InvalidModel("bad magic"));
        }
        let hash_offset = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 7;
        let hash_bytes = bytes
            .get(hash_offset..hash_offset + 8)
            .ok_or(TrainError::InvalidModel("artifact too short"))?;
        Ok(u64::from_le_bytes(
            hash_bytes
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad model hash"))?,
        ))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_MODEL_MAGIC.len() + 4 * 5 + 8 * 10 + 8 * 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("artifact too short"));
        }
        let magic = &bytes[..MINI_TRANSFORMER_MODEL_MAGIC.len()];
        let legacy = magic == MINI_TRANSFORMER_LEGACY_MODEL_MAGIC;
        if magic != MINI_TRANSFORMER_MODEL_MAGIC && !legacy {
            return Err(TrainError::InvalidModel("bad magic"));
        }

        let mut offset = MINI_TRANSFORMER_MODEL_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let context_seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let position_embedding_count = read_u64_le(bytes, &mut offset)? as usize;
        let q_count = read_u64_le(bytes, &mut offset)? as usize;
        let k_count = read_u64_le(bytes, &mut offset)? as usize;
        let v_count = read_u64_le(bytes, &mut offset)? as usize;
        let o_count = read_u64_le(bytes, &mut offset)? as usize;
        let up_count = read_u64_le(bytes, &mut offset)? as usize;
        let gate_count = read_u64_le(bytes, &mut offset)? as usize;
        let down_count = read_u64_le(bytes, &mut offset)? as usize;
        let output_count = read_u64_le(bytes, &mut offset)? as usize;
        let expected_embedding_hash = read_u64_le(bytes, &mut offset)?;
        let expected_q_hash = read_u64_le(bytes, &mut offset)?;
        let expected_k_hash = read_u64_le(bytes, &mut offset)?;
        let expected_v_hash = read_u64_le(bytes, &mut offset)?;
        let expected_o_hash = read_u64_le(bytes, &mut offset)?;
        let expected_mlp_hash = read_u64_le(bytes, &mut offset)?;
        let expected_output_hash = read_u64_le(bytes, &mut offset)?;
        let expected_model_hash = read_u64_le(bytes, &mut offset)?;

        if legacy
            && d_model == MINI_TRANSFORMER_LEGACY_V4_D_MODEL
            && heads == MINI_TRANSFORMER_LEGACY_V4_HEADS
            && hidden_dim == MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM
        {
            return decode_and_upgrade_legacy_v4_model(
                bytes,
                offset,
                vocab,
                context_seq_len,
                embedding_count,
                position_embedding_count,
                q_count,
                k_count,
                v_count,
                o_count,
                up_count,
                gate_count,
                down_count,
                output_count,
                expected_embedding_hash,
                expected_q_hash,
                expected_k_hash,
                expected_v_hash,
                expected_o_hash,
                expected_mlp_hash,
                expected_output_hash,
                expected_model_hash,
            );
        }

        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("shape mismatch"));
        }

        let expected_attention_count = mini_transformer_attention_weight_count()?;
        let expected_mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
        let expected_mlp_down_count = mini_transformer_mlp_down_weight_count()?;
        let inferred_attention_layers = infer_layer_count(q_count, expected_attention_count)
            .ok_or(TrainError::InvalidModel("attention tensor count mismatch"))?;
        let inferred_mlp_layers = infer_layer_count(up_count, expected_mlp_up_or_gate_count)
            .ok_or(TrainError::InvalidModel("mlp tensor count mismatch"))?;
        let expected_position_embedding_count = context_seq_len
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel(
                "position embedding count overflow",
            ))?;
        if embedding_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
            || position_embedding_count != expected_position_embedding_count
            || inferred_attention_layers == 0
            || inferred_mlp_layers == 0
            || inferred_attention_layers != inferred_mlp_layers
            || k_count != q_count
            || v_count != q_count
            || o_count != q_count
            || gate_count != up_count
            || down_count != inferred_mlp_layers * expected_mlp_down_count
            || output_count != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        {
            return Err(TrainError::InvalidModel("tensor count mismatch"));
        }

        let embedding_bytes = embedding_count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("embedding length overflow"))?;
        let position_embedding_bytes =
            position_embedding_count
                .checked_mul(2)
                .ok_or(TrainError::InvalidModel(
                    "position embedding length overflow",
                ))?;
        let weight_bytes = q_count
            .checked_add(k_count)
            .and_then(|value| value.checked_add(v_count))
            .and_then(|value| value.checked_add(o_count))
            .and_then(|value| value.checked_add(up_count))
            .and_then(|value| value.checked_add(gate_count))
            .and_then(|value| value.checked_add(down_count))
            .and_then(|value| value.checked_add(output_count))
            .ok_or(TrainError::InvalidModel("weight length overflow"))?;
        let base_expected_len = offset
            .checked_add(embedding_bytes)
            .and_then(|value| value.checked_add(position_embedding_bytes))
            .and_then(|value| value.checked_add(weight_bytes))
            .ok_or(TrainError::InvalidModel("artifact length overflow"))?;
        let rms_count = inferred_attention_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("RMSNorm count overflow"))?;
        let rms_bytes = rms_count
            .checked_mul(4)
            .ok_or(TrainError::InvalidModel("RMSNorm bytes overflow"))?;
        let rms_expected_len = base_expected_len
            .checked_add(rms_bytes)
            .ok_or(TrainError::InvalidModel("artifact length overflow"))?;
        let rms_enabled = !legacy && bytes.len() == rms_expected_len;
        if (legacy && bytes.len() != base_expected_len)
            || (!legacy && bytes.len() != base_expected_len && !rms_enabled)
        {
            return Err(TrainError::InvalidModel("artifact length mismatch"));
        }

        let embedding_end = offset + embedding_bytes;
        let mut embeddings = Vec::with_capacity(embedding_count);
        for chunk in bytes[offset..embedding_end].chunks_exact(2) {
            embeddings.push(i16::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TrainError::InvalidModel("bad embedding"))?,
            ));
        }
        offset = embedding_end;

        let position_embedding_end = offset + position_embedding_bytes;
        let mut position_embeddings = Vec::with_capacity(position_embedding_count);
        for chunk in bytes[offset..position_embedding_end].chunks_exact(2) {
            position_embeddings.push(i16::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| TrainError::InvalidModel("bad position embedding"))?,
            ));
        }
        offset = position_embedding_end;

        let q_weights = read_i8_vec(bytes, &mut offset, q_count)?;
        let k_weights = read_i8_vec(bytes, &mut offset, k_count)?;
        let v_weights = read_i8_vec(bytes, &mut offset, v_count)?;
        let o_weights = read_i8_vec(bytes, &mut offset, o_count)?;
        let up_weights = read_i8_vec(bytes, &mut offset, up_count)?;
        let gate_weights = read_i8_vec(bytes, &mut offset, gate_count)?;
        let down_weights = read_i8_vec(bytes, &mut offset, down_count)?;
        let output_weights = read_i8_vec(bytes, &mut offset, output_count)?;
        let (attention_rms_weights, mlp_rms_weights) = if rms_enabled {
            (
                read_i16_vec(bytes, &mut offset, rms_count)?,
                read_i16_vec(bytes, &mut offset, rms_count)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let model = Self::new_with_rms_weights(
            context_seq_len,
            embeddings,
            position_embeddings,
            attention_rms_weights,
            mlp_rms_weights,
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            up_weights,
            gate_weights,
            down_weights,
            output_weights,
        )?;
        if model.embedding_hash() != expected_embedding_hash {
            return Err(TrainError::InvalidModel("embedding hash mismatch"));
        }
        if model.attention_q_hash() != expected_q_hash
            || model.attention_k_hash() != expected_k_hash
            || model.attention_v_hash() != expected_v_hash
            || model.attention_o_hash() != expected_o_hash
        {
            return Err(TrainError::InvalidModel("attention hash mismatch"));
        }
        if model.mlp_hash() != expected_mlp_hash {
            return Err(TrainError::InvalidModel("mlp hash mismatch"));
        }
        if model.output_head_hash() != expected_output_hash {
            return Err(TrainError::InvalidModel("output hash mismatch"));
        }
        if model.model_hash() != expected_model_hash {
            return Err(TrainError::InvalidModel("model hash mismatch"));
        }
        Ok(model)
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_and_upgrade_legacy_v4_model(
    bytes: &[u8],
    mut offset: usize,
    vocab: usize,
    context_seq_len: usize,
    embedding_count: usize,
    position_embedding_count: usize,
    q_count: usize,
    k_count: usize,
    v_count: usize,
    o_count: usize,
    up_count: usize,
    gate_count: usize,
    down_count: usize,
    output_count: usize,
    expected_embedding_hash: u64,
    expected_q_hash: u64,
    expected_k_hash: u64,
    expected_v_hash: u64,
    expected_o_hash: u64,
    expected_mlp_hash: u64,
    expected_output_hash: u64,
    expected_model_hash: u64,
) -> Result<MiniTransformerMlpModel, TrainError> {
    let expected_embedding_count = BYTE_VOCAB
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy embedding count overflow"))?;
    let expected_position_count = context_seq_len
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy position count overflow"))?;
    let expected_attention_count = MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy attention count overflow"))?;
    let expected_up_count = MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM)
        .ok_or(TrainError::InvalidModel("legacy MLP count overflow"))?;
    let expected_down_count = MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy MLP count overflow"))?;
    let expected_output_count = BYTE_VOCAB
        .checked_mul(MINI_TRANSFORMER_LEGACY_V4_D_MODEL)
        .ok_or(TrainError::InvalidModel("legacy output count overflow"))?;
    if vocab != BYTE_VOCAB
        || context_seq_len == 0
        || embedding_count != expected_embedding_count
        || position_embedding_count != expected_position_count
        || q_count != expected_attention_count
        || k_count != expected_attention_count
        || v_count != expected_attention_count
        || o_count != expected_attention_count
        || up_count != expected_up_count
        || gate_count != expected_up_count
        || down_count != expected_down_count
        || output_count != expected_output_count
    {
        return Err(TrainError::InvalidModel("legacy V4 tensor count mismatch"));
    }

    let expected_len = offset
        .checked_add(
            embedding_count
                .checked_mul(2)
                .ok_or(TrainError::InvalidModel("legacy embedding length overflow"))?,
        )
        .and_then(|value| value.checked_add(position_embedding_count.checked_mul(2)?))
        .and_then(|value| value.checked_add(q_count))
        .and_then(|value| value.checked_add(k_count))
        .and_then(|value| value.checked_add(v_count))
        .and_then(|value| value.checked_add(o_count))
        .and_then(|value| value.checked_add(up_count))
        .and_then(|value| value.checked_add(gate_count))
        .and_then(|value| value.checked_add(down_count))
        .and_then(|value| value.checked_add(output_count))
        .ok_or(TrainError::InvalidModel("legacy artifact length overflow"))?;
    if bytes.len() != expected_len {
        return Err(TrainError::InvalidModel(
            "legacy V4 artifact length mismatch",
        ));
    }

    let embeddings = read_i16_vec(bytes, &mut offset, embedding_count)?;
    let position_embeddings = read_i16_vec(bytes, &mut offset, position_embedding_count)?;
    let q_weights = read_i8_vec(bytes, &mut offset, q_count)?;
    let k_weights = read_i8_vec(bytes, &mut offset, k_count)?;
    let v_weights = read_i8_vec(bytes, &mut offset, v_count)?;
    let o_weights = read_i8_vec(bytes, &mut offset, o_count)?;
    let up_weights = read_i8_vec(bytes, &mut offset, up_count)?;
    let gate_weights = read_i8_vec(bytes, &mut offset, gate_count)?;
    let down_weights = read_i8_vec(bytes, &mut offset, down_count)?;
    let output_weights = read_i8_vec(bytes, &mut offset, output_count)?;

    let mut embedding_hasher = StableHasher::new();
    embedding_hasher.update_i16_slice(&embeddings);
    embedding_hasher.update_i16_slice(&position_embeddings);
    if embedding_hasher.finish() != expected_embedding_hash {
        return Err(TrainError::InvalidModel("embedding hash mismatch"));
    }
    if hash_i8_slice(&q_weights) != expected_q_hash
        || hash_i8_slice(&k_weights) != expected_k_hash
        || hash_i8_slice(&v_weights) != expected_v_hash
        || hash_i8_slice(&o_weights) != expected_o_hash
    {
        return Err(TrainError::InvalidModel("attention hash mismatch"));
    }
    if hash_three_i8_slices(&up_weights, &gate_weights, &down_weights) != expected_mlp_hash {
        return Err(TrainError::InvalidModel("mlp hash mismatch"));
    }
    if hash_i8_slice(&output_weights) != expected_output_hash {
        return Err(TrainError::InvalidModel("output hash mismatch"));
    }
    let mut model_hasher = StableHasher::new();
    model_hasher.update_usize(context_seq_len);
    model_hasher.update_i16_slice(&embeddings);
    model_hasher.update_i16_slice(&position_embeddings);
    model_hasher.update_i8_slice(&q_weights);
    model_hasher.update_i8_slice(&k_weights);
    model_hasher.update_i8_slice(&v_weights);
    model_hasher.update_i8_slice(&o_weights);
    model_hasher.update_i8_slice(&up_weights);
    model_hasher.update_i8_slice(&gate_weights);
    model_hasher.update_i8_slice(&down_weights);
    model_hasher.update_i8_slice(&output_weights);
    if model_hasher.finish() != expected_model_hash {
        return Err(TrainError::InvalidModel("model hash mismatch"));
    }

    upgrade_legacy_v4_model(
        context_seq_len,
        embeddings,
        position_embeddings,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        up_weights,
        gate_weights,
        down_weights,
        output_weights,
    )
}

#[allow(clippy::too_many_arguments)]
fn upgrade_legacy_v4_model(
    context_seq_len: usize,
    embeddings: Vec<i16>,
    position_embeddings: Vec<i16>,
    q_weights: Vec<i8>,
    k_weights: Vec<i8>,
    v_weights: Vec<i8>,
    o_weights: Vec<i8>,
    up_weights: Vec<i8>,
    gate_weights: Vec<i8>,
    down_weights: Vec<i8>,
    output_weights: Vec<i8>,
) -> Result<MiniTransformerMlpModel, TrainError> {
    if MINI_TRANSFORMER_D_MODEL != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * 4
        || (MINI_TRANSFORMER_HEADS != MINI_TRANSFORMER_LEGACY_V4_HEADS
            && MINI_TRANSFORMER_HEADS != MINI_TRANSFORMER_LEGACY_V4_HEADS * 4)
        || MINI_TRANSFORMER_HIDDEN_DIM != MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * 4
    {
        return Err(TrainError::InvalidModel(
            "unsupported legacy V4 geometry upgrade",
        ));
    }
    let embeddings = widen_legacy_model_rows_i16(&embeddings, BYTE_VOCAB)?;
    let position_embeddings = widen_legacy_model_rows_i16(&position_embeddings, context_seq_len)?;
    let q_weights = widen_legacy_model_matrix(&q_weights, 2)?;
    let k_weights = widen_legacy_model_matrix(&k_weights, 2)?;
    let v_weights = widen_legacy_model_matrix(&v_weights, 4)?;
    let o_weights = widen_legacy_model_matrix(&o_weights, 4)?;
    let up_weights = widen_legacy_up_or_gate_matrix(&up_weights)?;
    let gate_weights = widen_legacy_up_or_gate_matrix(&gate_weights)?;
    let down_weights = widen_legacy_down_matrix(&down_weights)?;
    let output_weights = widen_legacy_output_matrix(&output_weights)?;
    MiniTransformerMlpModel::new(
        context_seq_len,
        embeddings,
        position_embeddings,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        up_weights,
        gate_weights,
        down_weights,
        output_weights,
    )
}

fn legacy_model_dim_index(index: usize, replica: usize) -> Result<usize, TrainError> {
    let old_head_dim = MINI_TRANSFORMER_LEGACY_V4_D_MODEL / MINI_TRANSFORMER_LEGACY_V4_HEADS;
    let new_head_dim = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS;
    if index >= MINI_TRANSFORMER_LEGACY_V4_D_MODEL || replica >= 4 {
        return Err(TrainError::InvalidModel("legacy model index out of range"));
    }
    let head = index / old_head_dim;
    let dim = index % old_head_dim;
    if MINI_TRANSFORMER_HEADS == MINI_TRANSFORMER_LEGACY_V4_HEADS {
        Ok(head * new_head_dim + dim * 4 + replica)
    } else if MINI_TRANSFORMER_HEADS == MINI_TRANSFORMER_LEGACY_V4_HEADS * 4
        && new_head_dim == old_head_dim
    {
        Ok((head * 4 + replica) * new_head_dim + dim)
    } else {
        Err(TrainError::InvalidModel(
            "unsupported legacy model head mapping",
        ))
    }
}

fn widen_legacy_model_rows_i16(values: &[i16], rows: usize) -> Result<Vec<i16>, TrainError> {
    if values.len() != rows * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy row tensor mismatch"));
    }
    let mut out = vec![0_i16; rows * MINI_TRANSFORMER_D_MODEL];
    for row in 0..rows {
        for old_dim in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
            let value = values[row * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_dim];
            for replica in 0..4 {
                out[row * MINI_TRANSFORMER_D_MODEL + legacy_model_dim_index(old_dim, replica)?] =
                    value;
            }
        }
    }
    Ok(out)
}

fn widen_legacy_model_matrix(values: &[i8], output_replicas: usize) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_D_MODEL
        || !(1..=4).contains(&output_replicas)
    {
        return Err(TrainError::InvalidModel("legacy attention tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_D_MODEL];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        for output_replica in 0..output_replicas {
            let new_output = legacy_model_dim_index(old_output, output_replica)?;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
                let new_input = legacy_model_dim_index(old_input, 0)?;
                out[new_output * MINI_TRANSFORMER_D_MODEL + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_up_or_gate_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy MLP tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
        for output_replica in 0..4 {
            let new_output = old_output * 4 + output_replica;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
                let new_input = legacy_model_dim_index(old_input, 0)?;
                out[new_output * MINI_TRANSFORMER_D_MODEL + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_down_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != MINI_TRANSFORMER_LEGACY_V4_D_MODEL * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
        return Err(TrainError::InvalidModel("legacy MLP tensor mismatch"));
    }
    let mut out = vec![0_i8; MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM];
    for old_output in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        for output_replica in 0..4 {
            let new_output = legacy_model_dim_index(old_output, output_replica)?;
            for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM {
                let new_input = old_input * 4;
                out[new_output * MINI_TRANSFORMER_HIDDEN_DIM + new_input] =
                    values[old_output * MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM + old_input];
            }
        }
    }
    Ok(out)
}

fn widen_legacy_output_matrix(values: &[i8]) -> Result<Vec<i8>, TrainError> {
    if values.len() != BYTE_VOCAB * MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
        return Err(TrainError::InvalidModel("legacy output tensor mismatch"));
    }
    let mut out = vec![0_i8; BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL];
    for output in 0..BYTE_VOCAB {
        for old_input in 0..MINI_TRANSFORMER_LEGACY_V4_D_MODEL {
            let new_input = legacy_model_dim_index(old_input, 0)?;
            out[output * MINI_TRANSFORMER_D_MODEL + new_input] =
                values[output * MINI_TRANSFORMER_LEGACY_V4_D_MODEL + old_input];
        }
    }
    Ok(out)
}

impl MiniTransformerBlockLowRankExpert {
    pub fn new_for_model(
        model: &MiniTransformerMlpModel,
        rank: usize,
        projection_seed: u64,
    ) -> Result<Self, TrainError> {
        Self::new_for_model_with_residual_shift(model, rank, projection_seed, 0)
    }

    pub fn new_for_model_with_residual_shift(
        model: &MiniTransformerMlpModel,
        rank: usize,
        projection_seed: u64,
        residual_shift: u8,
    ) -> Result<Self, TrainError> {
        let transformer_layers = model.checked_transformer_layers()?;
        if rank == 0 || rank > MINI_TRANSFORMER_D_MODEL || residual_shift > 15 {
            return Err(TrainError::InvalidConfig);
        }
        let parameter_count = transformer_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .and_then(|value| value.checked_mul(rank))
            .ok_or(TrainError::InvalidConfig)?;
        Ok(Self {
            trunk_model_hash: model.model_hash(),
            transformer_layers,
            rank,
            projection_seed,
            residual_shift,
            expansion_weights_q15: vec![0_i16; parameter_count],
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.expansion_weights_q15.len()
    }

    pub fn validate_for_model(&self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let expected = self
            .transformer_layers
            .checked_mul(MINI_TRANSFORMER_D_MODEL)
            .and_then(|value| value.checked_mul(self.rank))
            .ok_or(TrainError::InvalidConfig)?;
        if self.trunk_model_hash != model.model_hash()
            || self.transformer_layers != model.checked_transformer_layers()?
            || self.rank == 0
            || self.rank > MINI_TRANSFORMER_D_MODEL
            || self.residual_shift > 15
            || self.expansion_weights_q15.len() != expected
        {
            return Err(TrainError::InvalidModel("block expert/model mismatch"));
        }
        Ok(())
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.transformer_layers == 0
            || self.rank == 0
            || self.rank > MINI_TRANSFORMER_D_MODEL
            || self.residual_shift > 15
            || self.expansion_weights_q15.len()
                != self
                    .transformer_layers
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_mul(self.rank))
                    .ok_or(TrainError::InvalidConfig)?
        {
            return Err(TrainError::InvalidModel("invalid block expert"));
        }
        let mut out = Vec::with_capacity(80 + self.expansion_weights_q15.len() * 2);
        out.extend_from_slice(MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_D_MODEL, "d_model exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "heads exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HIDDEN_DIM, "hidden_dim exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(&self.trunk_model_hash.to_le_bytes());
        push_model_usize(&mut out, self.transformer_layers, "layers exceed u64")?;
        push_model_usize(&mut out, self.rank, "rank exceeds u64")?;
        out.extend_from_slice(&self.projection_seed.to_le_bytes());
        out.push(self.residual_shift);
        out.extend_from_slice(&[0_u8; 7]);
        push_model_usize(
            &mut out,
            self.expansion_weights_q15.len(),
            "block expert parameters exceed u64",
        )?;
        for &weight in &self.expansion_weights_q15 {
            out.extend_from_slice(&weight.to_le_bytes());
        }
        let checksum = hash_u8_slice(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("valid block expert should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        const HEADER_WITH_CHECKSUM: usize = 80;
        if bytes.len() < HEADER_WITH_CHECKSUM
            || &bytes[..MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC.len()]
                != MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC
        {
            return Err(TrainError::InvalidModel("bad block expert artifact"));
        }
        let checksum_offset = bytes.len() - 8;
        let mut checksum_cursor = checksum_offset;
        let checksum = read_u64_le(bytes, &mut checksum_cursor)?;
        if checksum_cursor != bytes.len() || hash_u8_slice(&bytes[..checksum_offset]) != checksum {
            return Err(TrainError::InvalidModel("block expert checksum mismatch"));
        }
        let mut offset = MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let trunk_model_hash = read_u64_le(bytes, &mut offset)?;
        let transformer_layers = read_model_usize(bytes, &mut offset)?;
        let rank = read_model_usize(bytes, &mut offset)?;
        let projection_seed = read_u64_le(bytes, &mut offset)?;
        let residual_shift = *bytes.get(offset).ok_or(TrainError::InvalidModel(
            "missing block expert residual shift",
        ))?;
        if bytes
            .get(offset + 1..offset + 8)
            .ok_or(TrainError::InvalidModel(
                "missing block expert reserved bytes",
            ))?
            .iter()
            .any(|&value| value != 0)
        {
            return Err(TrainError::InvalidModel("block expert reserved bytes"));
        }
        offset += 8;
        let parameter_count = read_model_usize(bytes, &mut offset)?;
        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || transformer_layers == 0
            || rank == 0
            || rank > MINI_TRANSFORMER_D_MODEL
            || residual_shift > 15
            || parameter_count
                != transformer_layers
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_mul(rank))
                    .ok_or(TrainError::InvalidModel("block expert size overflow"))?
            || bytes.len()
                != HEADER_WITH_CHECKSUM
                    .checked_add(
                        parameter_count
                            .checked_mul(2)
                            .ok_or(TrainError::InvalidModel("block expert payload overflow"))?,
                    )
                    .ok_or(TrainError::InvalidModel("block expert artifact overflow"))?
        {
            return Err(TrainError::InvalidModel("block expert header mismatch"));
        }
        let mut expansion_weights_q15 = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            let end = offset
                .checked_add(2)
                .ok_or(TrainError::InvalidModel("block expert offset overflow"))?;
            let raw: [u8; 2] = bytes
                .get(offset..end)
                .ok_or(TrainError::InvalidModel("truncated block expert"))?
                .try_into()
                .map_err(|_| TrainError::InvalidModel("truncated block expert"))?;
            expansion_weights_q15.push(i16::from_le_bytes(raw));
            offset = end;
        }
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel("block expert payload mismatch"));
        }
        Ok(Self {
            trunk_model_hash,
            transformer_layers,
            rank,
            projection_seed,
            residual_shift,
            expansion_weights_q15,
        })
    }
}

impl MiniTransformerAdamOptimizerState {
    pub fn new_for_model(
        model: &MiniTransformerMlpModel,
        config: IntegerAdamConfig,
    ) -> Result<Self, TrainError> {
        if !config.is_valid() {
            return Err(TrainError::InvalidConfig);
        }
        let parameter_count = model.optimizer_parameter_count()?;
        Ok(Self {
            context_seq_len: model.context_seq_len,
            step: 0,
            bound_model_hash: model.model_hash(),
            config,
            first_moments: vec![0_i64; parameter_count],
            second_moments: vec![0_u64; parameter_count],
            update_residuals: vec![0_i64; parameter_count],
        })
    }

    pub fn validate_for_model(&self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let parameter_count = model.optimizer_parameter_count()?;
        if !self.config.is_valid()
            || self.context_seq_len != model.context_seq_len
            || self.bound_model_hash != model.model_hash()
            || self.first_moments.len() != parameter_count
            || self.second_moments.len() != parameter_count
            || self.update_residuals.len() != parameter_count
        {
            return Err(TrainError::InvalidModel("optimizer state/model mismatch"));
        }
        Ok(())
    }

    pub fn bind_to_model(&mut self, model: &MiniTransformerMlpModel) -> Result<(), TrainError> {
        let parameter_count = model.optimizer_parameter_count()?;
        if !self.config.is_valid()
            || self.context_seq_len != model.context_seq_len
            || self.first_moments.len() != parameter_count
            || self.second_moments.len() != parameter_count
            || self.update_residuals.len() != parameter_count
        {
            return Err(TrainError::InvalidModel("optimizer state shape mismatch"));
        }
        self.bound_model_hash = model.model_hash();
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.first_moments.len()
    }

    pub fn state_hash(&self) -> Result<u64, TrainError> {
        let bytes = self.try_to_bytes()?;
        let checksum_offset = bytes
            .len()
            .checked_sub(8)
            .ok_or(TrainError::InvalidModel("optimizer checksum offset"))?;
        let mut offset = checksum_offset;
        read_u64_le(&bytes, &mut offset)
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if !self.config.is_valid()
            || self.first_moments.len() != self.second_moments.len()
            || self.first_moments.len() != self.update_residuals.len()
            || self.context_seq_len == 0
        {
            return Err(TrainError::InvalidModel("invalid optimizer state"));
        }
        let payload_bytes = self
            .parameter_count()
            .checked_mul(24)
            .ok_or(TrainError::InvalidModel("optimizer state size overflow"))?;
        let mut out = Vec::with_capacity(
            80_usize
                .checked_add(payload_bytes)
                .ok_or(TrainError::InvalidModel("optimizer artifact size overflow"))?,
        );
        out.extend_from_slice(MINI_TRANSFORMER_ADAM_STATE_MAGIC);
        out.extend_from_slice(&checked_u32(BYTE_VOCAB, "byte vocab exceeds u32")?.to_le_bytes());
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_D_MODEL, "d_model exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HEADS, "heads exceeds u32")?.to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(MINI_TRANSFORMER_HIDDEN_DIM, "hidden_dim exceeds u32")?.to_le_bytes(),
        );
        push_model_usize(&mut out, self.context_seq_len, "context length exceeds u64")?;
        push_model_usize(
            &mut out,
            self.parameter_count(),
            "parameter count exceeds u64",
        )?;
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.bound_model_hash.to_le_bytes());
        out.extend_from_slice(&self.config.learning_rate.to_le_bytes());
        out.push(self.config.step_shift);
        out.push(self.config.beta1_decay_shift);
        out.push(self.config.beta2_decay_shift);
        out.push(0);
        out.extend_from_slice(&self.config.epsilon.to_le_bytes());
        for &value in &self.first_moments {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.second_moments {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for &value in &self.update_residuals {
            out.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = hash_u8_slice(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("valid optimizer state should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        const FIXED_BYTES_WITH_CHECKSUM: usize = 80;
        if bytes.len() < FIXED_BYTES_WITH_CHECKSUM {
            return Err(TrainError::InvalidModel("optimizer artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_ADAM_STATE_MAGIC.len()] != MINI_TRANSFORMER_ADAM_STATE_MAGIC {
            return Err(TrainError::InvalidModel("bad optimizer magic"));
        }
        let checksum_offset = bytes
            .len()
            .checked_sub(8)
            .ok_or(TrainError::InvalidModel("optimizer checksum offset"))?;
        let mut checksum_cursor = checksum_offset;
        let expected_checksum = read_u64_le(bytes, &mut checksum_cursor)?;
        if checksum_cursor != bytes.len()
            || hash_u8_slice(&bytes[..checksum_offset]) != expected_checksum
        {
            return Err(TrainError::InvalidModel("optimizer checksum mismatch"));
        }

        let mut offset = MINI_TRANSFORMER_ADAM_STATE_MAGIC.len();
        let vocab = read_u32_le(bytes, &mut offset)? as usize;
        let d_model = read_u32_le(bytes, &mut offset)? as usize;
        let heads = read_u32_le(bytes, &mut offset)? as usize;
        let hidden_dim = read_u32_le(bytes, &mut offset)? as usize;
        let context_seq_len = read_model_usize(bytes, &mut offset)?;
        let parameter_count = read_model_usize(bytes, &mut offset)?;
        let step = read_u64_le(bytes, &mut offset)?;
        let bound_model_hash = read_u64_le(bytes, &mut offset)?;
        let learning_rate = read_u32_le(bytes, &mut offset)? as i32;
        let step_shift = *bytes
            .get(offset)
            .ok_or(TrainError::InvalidModel("missing optimizer step shift"))?;
        let beta1_decay_shift = *bytes
            .get(offset + 1)
            .ok_or(TrainError::InvalidModel("missing optimizer beta1 shift"))?;
        let beta2_decay_shift = *bytes
            .get(offset + 2)
            .ok_or(TrainError::InvalidModel("missing optimizer beta2 shift"))?;
        let reserved = *bytes
            .get(offset + 3)
            .ok_or(TrainError::InvalidModel("missing optimizer reserved byte"))?;
        offset += 4;
        let epsilon = read_u64_le(bytes, &mut offset)?;
        let config = IntegerAdamConfig {
            learning_rate,
            step_shift,
            beta1_decay_shift,
            beta2_decay_shift,
            epsilon,
        };
        if vocab != BYTE_VOCAB
            || d_model != MINI_TRANSFORMER_D_MODEL
            || heads != MINI_TRANSFORMER_HEADS
            || hidden_dim != MINI_TRANSFORMER_HIDDEN_DIM
            || context_seq_len == 0
            || reserved != 0
            || !config.is_valid()
        {
            return Err(TrainError::InvalidModel("optimizer header mismatch"));
        }
        let expected_len = FIXED_BYTES_WITH_CHECKSUM
            .checked_add(
                parameter_count
                    .checked_mul(24)
                    .ok_or(TrainError::InvalidModel("optimizer payload overflow"))?,
            )
            .ok_or(TrainError::InvalidModel("optimizer artifact overflow"))?;
        if bytes.len() != expected_len {
            return Err(TrainError::InvalidModel(
                "optimizer artifact length mismatch",
            ));
        }
        let first_moments = read_i64_vec(bytes, &mut offset, parameter_count)?;
        let second_moments = read_u64_vec(bytes, &mut offset, parameter_count)?;
        let update_residuals = read_i64_vec(bytes, &mut offset, parameter_count)?;
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel(
                "optimizer payload length mismatch",
            ));
        }
        Ok(Self {
            context_seq_len,
            step,
            bound_model_hash,
            config,
            first_moments,
            second_moments,
            update_residuals,
        })
    }
}

impl MiniTransformerMlpSwarmWorkerArtifact {
    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        if self.worker_count == 0
            || self.worker.worker_index >= self.worker_count
            || self.model.model_hash() != self.worker.final_model_hash
        {
            return Err(TrainError::InvalidModel("bad swarm worker artifact"));
        }
        let model_bytes = self.model.try_to_bytes()?;
        let mut out = Vec::with_capacity(checked_model_capacity(224, &[model_bytes.len()])?);
        out.extend_from_slice(MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC);
        push_model_usize(&mut out, self.worker_count, "worker count exceeds u64")?;
        push_model_usize(&mut out, self.token_count, "token count exceeds u64")?;
        out.extend_from_slice(&self.token_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            self.base_window_offset,
            "base window offset exceeds u64",
        )?;
        push_model_usize(&mut out, self.base_stride, "base stride exceeds u64")?;
        push_model_optional_usize(
            &mut out,
            self.base_max_windows,
            "base max windows exceeds u64",
        )?;
        out.extend_from_slice(&self.base_model_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            self.worker.worker_index,
            "worker index exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.window_offset,
            "worker window offset exceeds u64",
        )?;
        push_model_usize(&mut out, self.worker.stride, "worker stride exceeds u64")?;
        push_model_optional_usize(
            &mut out,
            self.worker.max_windows,
            "worker max windows exceeds u64",
        )?;
        out.extend_from_slice(&self.worker.window_hash.to_le_bytes());
        push_model_usize(&mut out, self.worker.windows, "worker windows exceeds u64")?;
        push_model_usize(
            &mut out,
            self.worker.examined_windows,
            "worker examined windows exceeds u64",
        )?;
        push_model_usize(&mut out, self.worker.updates, "worker updates exceeds u64")?;
        push_model_usize(
            &mut out,
            self.worker.accepted_batch_count,
            "worker accepted batches exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rejected_batch_count,
            "worker rejected batches exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rollback_count,
            "worker rollbacks exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.rejected_window_count,
            "worker rejected windows exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_invalid_forward_count,
            "worker invalid forward count exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.initial_total_error,
            "worker initial total error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_total_error,
            "worker final total error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.initial_probability_error_q15,
            "worker initial probability error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_probability_error_q15,
            "worker final probability error exceeds u64",
        )?;
        push_model_usize(
            &mut out,
            self.worker.final_accuracy_per_mille,
            "worker final accuracy exceeds u64",
        )?;
        out.extend_from_slice(&self.worker.final_model_hash.to_le_bytes());
        out.extend_from_slice(&self.worker.final_logits_hash.to_le_bytes());
        push_model_usize(
            &mut out,
            model_bytes.len(),
            "worker model bytes exceeds u64",
        )?;
        out.extend_from_slice(&model_bytes);
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer swarm worker artifact should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len() {
            return Err(TrainError::InvalidModel("swarm worker artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len()]
            != MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC
        {
            return Err(TrainError::InvalidModel("bad swarm worker magic"));
        }

        let mut offset = MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC.len();
        let worker_count = read_model_usize(bytes, &mut offset)?;
        let token_count = read_model_usize(bytes, &mut offset)?;
        let token_hash = read_u64_le(bytes, &mut offset)?;
        let base_window_offset = read_model_usize(bytes, &mut offset)?;
        let base_stride = read_model_usize(bytes, &mut offset)?;
        let base_max_windows = read_model_optional_usize(bytes, &mut offset)?;
        let base_model_hash = read_u64_le(bytes, &mut offset)?;
        let worker_index = read_model_usize(bytes, &mut offset)?;
        let window_offset = read_model_usize(bytes, &mut offset)?;
        let stride = read_model_usize(bytes, &mut offset)?;
        let max_windows = read_model_optional_usize(bytes, &mut offset)?;
        let window_hash = read_u64_le(bytes, &mut offset)?;
        let windows = read_model_usize(bytes, &mut offset)?;
        let examined_windows = read_model_usize(bytes, &mut offset)?;
        let updates = read_model_usize(bytes, &mut offset)?;
        let accepted_batch_count = read_model_usize(bytes, &mut offset)?;
        let rejected_batch_count = read_model_usize(bytes, &mut offset)?;
        let rollback_count = read_model_usize(bytes, &mut offset)?;
        let rejected_window_count = read_model_usize(bytes, &mut offset)?;
        let final_invalid_forward_count = read_model_usize(bytes, &mut offset)?;
        let initial_total_error = read_model_usize(bytes, &mut offset)?;
        let final_total_error = read_model_usize(bytes, &mut offset)?;
        let initial_probability_error_q15 = read_model_usize(bytes, &mut offset)?;
        let final_probability_error_q15 = read_model_usize(bytes, &mut offset)?;
        let final_accuracy_per_mille = read_model_usize(bytes, &mut offset)?;
        let final_model_hash = read_u64_le(bytes, &mut offset)?;
        let final_logits_hash = read_u64_le(bytes, &mut offset)?;
        let model_len = read_model_usize(bytes, &mut offset)?;
        let model_end = offset
            .checked_add(model_len)
            .ok_or(TrainError::InvalidModel(
                "swarm worker model offset overflow",
            ))?;
        let model_bytes = bytes
            .get(offset..model_end)
            .ok_or(TrainError::InvalidModel("swarm worker model truncated"))?;
        offset = model_end;
        if offset != bytes.len() || worker_count == 0 || worker_index >= worker_count {
            return Err(TrainError::InvalidModel("bad swarm worker header"));
        }

        let model = MiniTransformerMlpModel::from_bytes(model_bytes)?;
        if model.model_hash() != final_model_hash {
            return Err(TrainError::InvalidModel("swarm worker model hash mismatch"));
        }
        Ok(Self {
            worker_count,
            token_count,
            token_hash,
            base_window_offset,
            base_stride,
            base_max_windows,
            base_model_hash,
            worker: MiniTransformerMlpSwarmWorkerTrace {
                worker_index,
                window_offset,
                stride,
                max_windows,
                token_hash,
                window_hash,
                windows,
                examined_windows,
                updates,
                accepted_batch_count,
                rejected_batch_count,
                rollback_count,
                rejected_window_count,
                final_invalid_forward_count,
                initial_total_error,
                final_total_error,
                initial_probability_error_q15,
                final_probability_error_q15,
                final_accuracy_per_mille,
                final_model_hash,
                final_logits_hash,
            },
            model,
        })
    }

    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(&mut out, "schema", MINI_TRANSFORMER_SWARM_WORKER_SCHEMA);
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(
            &mut out,
            "task",
            "wiki_bard_mini_transformer_mlp_swarm_worker",
        );
        comma(&mut out);
        out.push_str("\"data\":{");
        push_usize_field(&mut out, "token_count", self.token_count);
        comma(&mut out);
        push_hash_field(&mut out, "token_hash", self.token_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"swarm\":{");
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "base_window_offset", self.base_window_offset);
        comma(&mut out);
        push_usize_field(&mut out, "base_stride", self.base_stride);
        comma(&mut out);
        push_optional_usize_field(&mut out, "base_max_windows", self.base_max_windows);
        comma(&mut out);
        push_hash_field(&mut out, "base_model_hash", self.base_model_hash);
        out.push('}');
        comma(&mut out);
        push_quoted(&mut out, "worker");
        out.push(':');
        push_mini_transformer_swarm_worker(&mut out, &self.worker);
        comma(&mut out);
        out.push_str("\"artifact\":{");
        push_string_field(&mut out, "format", "nsrlswarm-worker");
        comma(&mut out);
        push_string_field(&mut out, "magic", "NSRLWK1");
        comma(&mut out);
        push_usize_field(
            &mut out,
            "model_bytes",
            self.model
                .try_to_bytes()
                .map(|bytes| bytes.len())
                .unwrap_or(0),
        );
        out.push('}');
        out.push('}');
        out.push('\n');
        out
    }
}

impl MiniTransformerMlpSwarmModel {
    pub fn new(
        best_worker_index: usize,
        workers: Vec<MiniTransformerMlpModel>,
    ) -> Result<Self, TrainError> {
        let first = workers
            .first()
            .ok_or(TrainError::InvalidModel("empty mini transformer swarm"))?;
        if best_worker_index >= workers.len() {
            return Err(TrainError::InvalidModel("swarm best worker out of range"));
        }
        let context_seq_len = first.context_seq_len;
        if context_seq_len == 0
            || workers
                .iter()
                .any(|worker| worker.context_seq_len != context_seq_len)
        {
            return Err(TrainError::InvalidModel(
                "swarm worker context length mismatch",
            ));
        }
        Ok(Self {
            context_seq_len,
            best_worker_index,
            workers,
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn model_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.context_seq_len);
        hasher.update_usize(self.best_worker_index);
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.model_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn embedding_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.embedding_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn attention_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.attention_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn mlp_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.mlp_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn output_head_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.update_usize(self.workers.len());
        for worker in &self.workers {
            hasher.update_bytes(&worker.output_head_hash().to_le_bytes());
        }
        hasher.finish()
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        let worker_blobs = self
            .workers
            .iter()
            .map(MiniTransformerMlpModel::try_to_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        let payload_bytes = worker_blobs.iter().try_fold(0_usize, |total, blob| {
            total
                .checked_add(8)
                .and_then(|value| value.checked_add(blob.len()))
                .ok_or(TrainError::InvalidModel("swarm artifact length overflow"))
        })?;
        let mut out = Vec::with_capacity(checked_model_capacity(32, &[payload_bytes])?);
        out.extend_from_slice(MINI_TRANSFORMER_SWARM_MODEL_MAGIC);
        out.extend_from_slice(
            &checked_u32(
                self.context_seq_len,
                "mini transformer swarm context_seq_len exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.workers.len(),
                "mini transformer swarm worker count exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(
            &checked_u32(
                self.best_worker_index,
                "mini transformer swarm best worker exceeds u32",
            )?
            .to_le_bytes(),
        );
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&self.model_hash().to_le_bytes());
        for blob in worker_blobs {
            out.extend_from_slice(
                &checked_u64(blob.len(), "mini transformer worker blob exceeds u64")?.to_le_bytes(),
            );
            out.extend_from_slice(&blob);
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.try_to_bytes()
            .expect("mini transformer swarm model should fit on-disk format")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        let header_len = MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len() + 4 + 4 + 4 + 4 + 8;
        if bytes.len() < header_len {
            return Err(TrainError::InvalidModel("swarm artifact too short"));
        }
        if &bytes[..MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len()] != MINI_TRANSFORMER_SWARM_MODEL_MAGIC
        {
            return Err(TrainError::InvalidModel("bad swarm magic"));
        }
        let mut offset = MINI_TRANSFORMER_SWARM_MODEL_MAGIC.len();
        let context_seq_len = read_u32_le(bytes, &mut offset)? as usize;
        let worker_count = read_u32_le(bytes, &mut offset)? as usize;
        let best_worker_index = read_u32_le(bytes, &mut offset)? as usize;
        let reserved = read_u32_le(bytes, &mut offset)?;
        let expected_model_hash = read_u64_le(bytes, &mut offset)?;
        if worker_count == 0 || best_worker_index >= worker_count || reserved != 0 {
            return Err(TrainError::InvalidModel("bad swarm header"));
        }

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let blob_len = read_u64_le(bytes, &mut offset)? as usize;
            let blob_end = offset
                .checked_add(blob_len)
                .ok_or(TrainError::InvalidModel("swarm worker offset overflow"))?;
            let blob = bytes
                .get(offset..blob_end)
                .ok_or(TrainError::InvalidModel("swarm worker blob truncated"))?;
            workers.push(MiniTransformerMlpModel::from_bytes(blob)?);
            offset = blob_end;
        }
        if offset != bytes.len() {
            return Err(TrainError::InvalidModel("swarm artifact length mismatch"));
        }

        let model = Self::new(best_worker_index, workers)?;
        if model.context_seq_len != context_seq_len {
            return Err(TrainError::InvalidModel("swarm context hash mismatch"));
        }
        if model.model_hash() != expected_model_hash {
            return Err(TrainError::InvalidModel("swarm model hash mismatch"));
        }
        Ok(model)
    }

    pub fn to_expert_manifest(&self) -> Result<MiniTransformerMlpSwarmExpertManifest, TrainError> {
        Ok(MiniTransformerMlpSwarmExpertManifest {
            artifact_format: "nsrlswarm",
            artifact_magic: "NSRLSW1",
            artifact_byte_count: self.try_to_bytes()?.len(),
            model_id: MINI_TRANSFORMER_SWARM_MODEL_ID,
            tokenizer: BYTE_TOKENIZER_ID,
            context_seq_len: self.context_seq_len,
            worker_count: self.worker_count(),
            best_worker_index: self.best_worker_index,
            parameter_bytes: self.parameter_bytes(),
            model_hash: self.model_hash(),
            embedding_hash: self.embedding_hash(),
            attention_hash: self.attention_hash(),
            mlp_hash: self.mlp_hash(),
            output_head_hash: self.output_head_hash(),
            worker_model_hashes: self
                .workers
                .iter()
                .map(MiniTransformerMlpModel::model_hash)
                .collect(),
            worker_parameter_bytes: self
                .workers
                .iter()
                .map(mini_transformer_mlp_parameter_bytes)
                .collect(),
        })
    }

    pub fn parameter_bytes(&self) -> usize {
        self.workers
            .iter()
            .map(mini_transformer_mlp_parameter_bytes)
            .fold(0_usize, usize::saturating_add)
    }
}

fn mini_transformer_mlp_parameter_bytes(model: &MiniTransformerMlpModel) -> usize {
    model
        .embeddings
        .len()
        .saturating_add(model.position_embeddings.len())
        .saturating_mul(core::mem::size_of::<i16>())
        .saturating_add(model.q_weights.len())
        .saturating_add(model.k_weights.len())
        .saturating_add(model.v_weights.len())
        .saturating_add(model.o_weights.len())
        .saturating_add(model.up_weights.len())
        .saturating_add(model.gate_weights.len())
        .saturating_add(model.down_weights.len())
        .saturating_add(model.output_weights.len())
}

impl MiniTransformerMlpSwarmExpertManifest {
    pub fn capability_tags(&self) -> &'static [&'static str] {
        MINI_TRANSFORMER_SWARM_CAPABILITY_TAGS
    }

    pub fn supports_capability(&self, capability: &str) -> bool {
        self.capability_tags().contains(&capability)
    }

    pub fn to_json_line(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_string_field(
            &mut out,
            "schema",
            MINI_TRANSFORMER_SWARM_EXPERT_MANIFEST_SCHEMA,
        );
        comma(&mut out);
        push_string_field(&mut out, "authority", AUTHORITY);
        comma(&mut out);
        push_string_field(&mut out, "model", self.model_id);
        comma(&mut out);
        out.push_str("\"artifact\":{");
        push_string_field(&mut out, "format", self.artifact_format);
        comma(&mut out);
        push_string_field(&mut out, "magic", self.artifact_magic);
        comma(&mut out);
        push_usize_field(&mut out, "bytes", self.artifact_byte_count);
        comma(&mut out);
        push_hash_field(&mut out, "model_hash", self.model_hash);
        out.push('}');
        comma(&mut out);
        out.push_str("\"tokenizer\":{");
        push_string_field(&mut out, "id", self.tokenizer);
        comma(&mut out);
        push_string_field(&mut out, "contract", "identity_u8_bytes");
        out.push('}');
        comma(&mut out);
        out.push_str("\"interfaces\":{");
        push_string_field(&mut out, "input_schema", "nsrl.byte_prompt.v1");
        comma(&mut out);
        push_string_field(&mut out, "output_schema", "nsrl.byte_generation.v1");
        comma(&mut out);
        push_string_field(
            &mut out,
            "generation_trace_schema",
            MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA,
        );
        out.push('}');
        comma(&mut out);
        out.push_str("\"numeric_contract\":{");
        push_string_field(&mut out, "residual_scale", "q15_i16");
        comma(&mut out);
        push_string_field(&mut out, "weight_dtype", "qint8");
        comma(&mut out);
        push_string_field(&mut out, "activation_dtype", "qint16");
        comma(&mut out);
        push_string_field(&mut out, "accumulator_dtype", "qint64");
        comma(&mut out);
        push_string_field(&mut out, "softmax", "base2_q15");
        out.push('}');
        comma(&mut out);
        out.push_str("\"model_shape\":{");
        push_usize_field(&mut out, "context_seq_len", self.context_seq_len);
        comma(&mut out);
        push_usize_field(&mut out, "worker_count", self.worker_count);
        comma(&mut out);
        push_usize_field(&mut out, "best_worker_index", self.best_worker_index);
        comma(&mut out);
        push_usize_field(&mut out, "vocab", BYTE_VOCAB);
        comma(&mut out);
        push_usize_field(&mut out, "d_model", MINI_TRANSFORMER_D_MODEL);
        comma(&mut out);
        push_usize_field(&mut out, "heads", MINI_TRANSFORMER_HEADS);
        comma(&mut out);
        push_usize_field(&mut out, "hidden_dim", MINI_TRANSFORMER_HIDDEN_DIM);
        out.push('}');
        comma(&mut out);
        out.push_str("\"hashes\":{");
        push_hash_field(&mut out, "model_hash", self.model_hash);
        comma(&mut out);
        push_hash_field(&mut out, "embedding_hash", self.embedding_hash);
        comma(&mut out);
        push_hash_field(&mut out, "attention_hash", self.attention_hash);
        comma(&mut out);
        push_hash_field(&mut out, "mlp_hash", self.mlp_hash);
        comma(&mut out);
        push_hash_field(&mut out, "output_head_hash", self.output_head_hash);
        comma(&mut out);
        push_hash_array_field(&mut out, "worker_model_hashes", &self.worker_model_hashes);
        out.push('}');
        comma(&mut out);
        out.push_str("\"capabilities\":{");
        push_string_array_field(&mut out, "tags", self.capability_tags());
        out.push('}');
        comma(&mut out);
        out.push_str("\"routing_hints\":{");
        push_string_field(&mut out, "router", "deterministic_symbolic");
        comma(&mut out);
        push_string_field(&mut out, "default_composition", "average_logits");
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "supported_compositions",
            &["average_logits", "confidence_weighted", "confidence_router"],
        );
        comma(&mut out);
        push_string_field(&mut out, "confidence_signal", "top_logit_margin_q8");
        comma(&mut out);
        push_string_field(&mut out, "tie_breaker", "lowest_worker_index");
        out.push('}');
        comma(&mut out);
        out.push_str("\"budgets\":{");
        push_usize_field(&mut out, "artifact_bytes", self.artifact_byte_count);
        comma(&mut out);
        push_usize_field(&mut out, "parameter_bytes", self.parameter_bytes);
        comma(&mut out);
        push_usize_array_field(
            &mut out,
            "worker_parameter_bytes",
            &self.worker_parameter_bytes,
        );
        comma(&mut out);
        push_bool_field(&mut out, "wasm_bundle_budget_known", false);
        out.push('}');
        comma(&mut out);
        push_string_array_field(
            &mut out,
            "known_non_claims",
            &[
                "not_a_general_purpose_language_model",
                "byte_level_contract_only",
                "single_block_mini_transformer_workers",
                "router_hints_are_symbolic_not_learned",
                "wasm_bundle_budget_not_measured_yet",
            ],
        );
        out.push('}');
        out.push('\n');
        out
    }
}
