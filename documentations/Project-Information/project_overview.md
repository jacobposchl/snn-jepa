# Aligning Spiking Neural Networks with Biological Latent Manifolds via JEPA

## Research Problem

Neural population codes change dramatically across brain states (active vs. passive, different behavioral contexts), causing decoders trained in one state to fail in another. We don't understand the underlying circuit mechanisms driving these shifts. Existing deep learning models achieve high decoding accuracy but don't reveal interpretable neural dynamics or circuit motifs.

Additionally, traditional approaches face a fundamental **scalability problem**: neural recordings have arbitrary population sizes across sessions, making it impossible to train a single model across the full Allen Institute dataset (153 sessions, 200,000 neurons) using fixed-width binning methods.

## Our Solution

We use **event-based unit tokenization** combined with **LeJEPA** to discover the "ground truth" latent dynamics from real neural recordings across all 153 sessions - what computations the brain is actually performing, while filtering out noise and irrelevant variability. We then test whether biologically-constrained Spiking Neural Networks can implement these same computations. If successful, the learned SNN connectivity patterns should reveal minimal, interpretable circuit mechanisms sufficient for predictive neural coding.

### Key Innovation: Unit Tokenization

Instead of fixed-width binning (e.g., 10ms bins), we treat neural activity as a **sparse stream of events** where each neuron has a learnable identity. This allows us to:
- Train on the full 153-session dataset without neuron correspondence problems
- Handle arbitrary population sizes (50 neurons to 1000+ neurons per session)
- Preserve precise spike timing information
- Scale to 200,000+ unique neurons

### Solution Justifications

- **Scalability**: Unit tokenization solves the arbitrary population size problem, enabling training on the complete Allen Institute dataset
- **Mechanism discovery**: SNNs are interpretable, allowing us to test whether imposed biological constraints are sufficient for the computations JEPA discovers
- **Noise filtering**: JEPA's predictive objective discovers computations rather than reconstructing noisy spikes
- **Causal validity**: Temporal causality is preserved throughout both teacher and student models
- **Connects to neuroscience**: Learned circuits can be compared to known biological motifs

## Research Questions

### Main Question
Can spike-based neural networks, constrained by biological dynamics (temporal integration, spike timing, refractory periods), learn to reproduce the same latent predictive dynamics that LeJEPA models learn from neural recordings across multiple sessions and brain regions?

### Secondary Questions
- What circuit connectivity patterns emerge to support cross-session predictive computation?
- Do these patterns recapitulate known biological motifs (E/I balance, divisive normalization, etc.)?
- Do SNNs trained on one brain state generalize to others, and if not, what circuit mechanisms differ?
- Can a single "universal" latent language capture dynamics across different mice, brain regions, and behavioral states?

## Methodology

Our approach consists of two phases:

### Phase 1: Multi-Session "Teacher" Model

**Goal**: Discover a "universal" latent language that generalizes across different mice, brain regions (V1, thalamus, hippocampus), and behavioral states.

#### A. Event-Based Neural Tokenization

Instead of fixed-width binning, neural activity is treated as a sparse event stream:

- **Unit Embeddings**: Each of the 200,000 neurons is assigned a unique D-dimensional learnable vector, allowing the model to learn the functional "identity" of each neuron regardless of which session it was recorded in.

- **Spike Tokens**: Each spike event (unit_id, timestamp) is represented as a token **z = e_unit + t_pos**, where e_unit is the unit embedding and t_pos encodes the precise timestamp.

- **Unit Delimiters**: To inform the model which neurons are "online" even if silent, `[START]` and `[END]` tokens are added for every unit in the recording window.

#### B. PerceiverIO Backbone (The Standardizer)

To process thousands of spike tokens efficiently and solve the arbitrary population size problem:

- **Cross-Attention Bottleneck**: A fixed set of learned latent tokens (e.g., 512 tokens) acts as queries to "pull" information from the variable-length spike sequence. This standardizes neural data into a fixed "latent language".

- **Causal Transformer Blocks**: Latents are processed through 24 layers of self-attention with **causal masking** so that latents at time t cannot attend to future activity, ensuring learnable dynamics for a causal SNN.

- **Rotary Position Encoding (RoPE)**: Timestamps are injected directly into the attention mechanism to preserve fine temporal structure of the neural code.

#### C. LeJEPA Predictive Objective

To filter out noise and ensure stable representations:

- **Target Encoder**: An Exponential Moving Average (EMA) version of the context encoder provides stable targets for future latent states.

- **Predictor Head**: A lightweight network predicts future latent tokens from current ones.

- **SIGReg (Sketched Isotropic Gaussian Regularization)**: Prevents representational collapse by forcing latents into a symmetrical, high-variance distribution (~N(0,I)), making the space easier for the SNN to map.

**Training Details**:
- Input: Raw spike streams from multiple sessions (no session-specific alignment needed)
- Output: Fixed-dimensional latent representations at each time step
- Loss: LeJEPA loss = (1-λ) * prediction_loss + λ * SIGReg_loss
- Optimization: AdamW with cosine annealing scheduler

### Phase 2: Biologically-Constrained "Student" Model

**Goal**: Test if a Spiking Neural Network (SNN), limited by biological physics, can implement the predictive computations discovered by the Teacher.

#### A. SNN Architecture & Inputs

The SNN uses **Leaky Integrate-and-Fire (LIF)** neurons:

- **Input**: Raw spike trains (same as Teacher)
- **Integration**: LIF neurons integrate information over time using membrane potential to bridge gaps between incoming spikes
- **Dynamics**: Realistic time constants, refractory periods, and synaptic delays

#### B. Distillation via Latent Alignment

Freeze the Teacher and use it as a reference:

- **CCA Loss**: Since the SNN's internal state and Teacher's latent tokens reside in different dimensions, we use **Rotationally-invariant Canonical Correlation Analysis (CCA)** to align them.

- **Surrogate Gradients**: To train the non-differentiable SNN, we use a **fast sigmoid** surrogate function to backpropagate the CCA error into the SNN's synaptic weights.

#### C. Biological Guardrails

To ensure the resulting circuit is interpretable and realistic:

- **Homeostatic Firing Rate Penalty**: Regularization keeps the SNN's mean firing rate within the biological range of **5–20Hz**, preventing unrealistic "bursting" solutions.

- **Dale's Principle** (optional): Enforce separate excitatory and inhibitory neuron populations.

- **Temporal Constraints**: Include realistic refractory periods and specific time constants for synaptic decay.

- **Sparse Connectivity** (optional): Encourage biologically plausible sparse connection patterns.

**Training Details**:
- Frozen Teacher provides target latent representations
- SNN trained to match these representations via CCA loss
- Additional homeostatic regularization for biological plausibility
- Evaluation on both latent matching and downstream decoding tasks

### Future Extensions

- **Phase 3**: Compare neuron models (LIF → Izhikevich → multi-compartment) to test effect of biophysical complexity
- **Phase 4**: Test cross-state generalization (train on active behavior, test on passive viewing)
- **Phase 5**: Identify circuit motif differences across brain regions and behavioral states
- **Phase 6**: Test transfer learning to new sessions/mice with zero-shot or few-shot adaptation

## Resources

### Dataset: Allen Institute Visual Behavior Neuropixels
- [Link to dataset](https://brain-map.org/our-research/circuits-behavior/visual-behavior)
- **200,000 recorded neurons** across 153 experimental sessions
- **153 experimental sessions** with varied population sizes (50-1000+ neurons per session)
- **Visual areas & subcortical structures**: V1, LM, AL, RL, AM, PM, thalamus, hippocampus, midbrain
- **Task-dependent modulation**: Each session includes passive stimulus replay blocks for investigating task-dependent effects on sensory and behavioral coding
- **Rich behavioral data**: Change detection task with licking responses, running speed, pupil diameter

### Computation
- **Google Colab** with GPU/TPU support
- **Efficient data loading**: PyTorch DataLoaders with streaming for large-scale spike data
- **Distributed training**: Multi-GPU support for PerceiverIO training across sessions

## Metrics of Success

We define success through six sequential milestones:

### Phase 0: Data Pipeline & Infrastructure **[COMPLETED]**
- Download and preprocess Allen Institute Visual Behavior Neuropixels dataset
- Implement event-based tokenization with unit embeddings
- Validate data quality: mean firing rates in biological range, stable recordings
- **Success metric**: Clean pipeline producing correctly formatted spike event streams

### Phase 1: LeJEPA Teacher Baseline
- Train PerceiverIO + LeJEPA on multi-session neural data with unit tokenization
- Validate predictive performance on held-out time windows across sessions
- **Success metrics**:
  - Prediction MSE significantly better than chance
  - Latent space shows temporal structure
  - No representational collapse (verified via SIGReg)
  - Model generalizes to held-out sessions

### Phase 2: SNN Proof of Concept ("Student" Model)
- Train basic SNN (LIF neurons, no strict biological constraints yet) to match JEPA latents
- **Success metrics**:
  - CCA similarity > 0.7 between SNN and JEPA latents on held-out data
  - SNN maintains stable firing patterns
  - Prediction performance on downstream decoding tasks (stimulus identity & behavioral choice) within 10% of JEPA

### Phase 3: Biological Constraint Analysis
- Add constraints (homeostatic firing rate penalty, Dale's principle, sparse connectivity)
- Compare constrained vs. unconstrained SNN performance
- **Success metrics**:
  - Constrained SNN achieves >80% of unconstrained performance
  - Firing rates remain in 5-20Hz biological range
  - Learned connectivity shows interpretable structure (E/I balance, sparse patterns)

### Phase 4: Cross-State Generalization
- Train on active behavior blocks, test on passive viewing (& vice versa)
- **Success metrics**:
  - Within-state CCA similarity > 0.7 (replication of Phase 2)
  - Cross-state CCA similarity > 0.5 (demonstrates partial generalization)
  - Identify which circuit parameters differ across states

### Phase 5: Cross-Region & Cross-Session Analysis
- Analyze learned representations across brain regions (V1, thalamus, hippocampus)
- Test zero-shot transfer to new sessions
- **Success metrics**:
  - Identify region-specific vs. universal circuit motifs
  - Quantify transfer learning performance
  - Characterize functional differences in learned embeddings across regions

## Related Work

### Self-Supervised Learning on Neural Data
Recent work has applied self-supervised learning to neural recordings, including BrainBERT for intracranial field potentials and Brain-JEPA for fMRI data. These methods use predictive objectives (masked prediction, future forecasting) to learn representations without task labels. However, these approaches use standard deep learning architectures that lack biological constraints and don't reveal interpretable circuit mechanisms.

### Predictive Models of Neural Dynamics
LFADS pioneered latent dynamics modeling for single-trial spike data, while the Neural Data Transformer (NDT) provided a non-recurrent alternative with faster inference. More recent foundation models achieve strong cross-subject generalization by pretraining on massive datasets (~135,000 neurons). While these models excel at prediction and generalization, their black-box nature limits mechanistic interpretability. **Critically, these models struggle with arbitrary population sizes across sessions.**

### Event-Based Neural Encoding
The **POYO** framework introduced event-based tokenization for neural data, treating each spike as a discrete token with learned unit embeddings. This approach scales to large multi-session datasets by avoiding the fixed input dimensionality constraint of binning methods. Our work extends this by combining event-based tokenization with JEPA's predictive objective and SNN distillation.

### Spiking Neural Networks and Biological Constraints
SNNs trained with biologically plausible learning rules (STDP, E/I balance) have been applied to neural data, with some work showing emergence of biological tuning properties. Knowledge distillation from Transformers to SNNs (LDD, CKDSNN) has succeeded in computer vision tasks. However, this distillation approach has not been applied to learning latent dynamics from real multi-session neural recordings.

### Interpretable Circuit Mechanisms
Recent work on Recurrent Mechanistic Models (RMMs) combines predictive power with mechanistic interpretability by learning conductance-based dynamics from intracellular recordings. Sparse autoencoders have also been applied to extract interpretable features from neural network representations trained on neural data.

## Our Contribution

To our knowledge, no prior work has combined:
1. **LeJEPA-style predictive learning** with SIGReg for stable, collapse-free representations
2. **Biologically-constrained SNN distillation** for mechanism discovery

This intersection represents a novel approach to understanding the computational sufficiency of biological constraints and extracting interpretable circuit motifs from large-scale real neural data.

## Why This Unified Approach is "Reviewer-Solid"

1. **Scalability**: Unit tokenization allows training on the **full 153-session dataset** without the "lack of neuron correspondence" problem that limits traditional models.

2. **Noise-Free Discovery**: Using JEPA rather than standard autoencoders ensures the Teacher discovers **computations** (predictive latents) rather than just reconstructing noisy raw spikes.

3. **Causal Validity**: The combination of causal masking in the Teacher and real-time processing in the SNN ensures "mechanism discovery" is biologically plausible.

4. **Isotropic Stability**: Enforcing Gaussianity in Phase 1 provides the SNN with a stable, well-distributed landscape to learn, reducing the risk of "dead" neurons in the student model.

5. **Interpretability**: Unlike black-box foundation models, our approach yields explicit circuit connectivity patterns that can be analyzed and compared to known biological motifs.

6. **Generalization Testing**: Multi-session training enables rigorous tests of cross-state, cross-region, and cross-subject generalization - addressing key questions about neural code universality.