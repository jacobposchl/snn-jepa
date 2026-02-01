# Aligning Spiking Neural Networks with Biological Latent Manifolds via JEPA

# Research Problem:

Neural population codes change dramatically across brain states (active
vs. passive, different behavioral contexts), causing decoders trained in
one state to fail in another. We don\'t understand the underlying
circuit mechanisms driving these shifts, existing deep learning models
achieve high decoding accuracy but don\'t reveal interpretable neural
dynamics or circuit motifs.

# Our Solution:

We use JEPA to discover the \"ground truth\" latent dynamics from real
neural recordings - what computations the brain is actually performing,
while filtering out noise and irrelevant variability through its
predictive objective. We then test whether biologically-constrained
Spiking Neural Networks can implement these same computations. If
successful, the learned SNN connectivity patterns should reveal minimal,
interpretable circuit mechanisms sufficient for predictive neural
coding.

Solution justifications:

-   **Mechanism discovery**: SNNs are interpretable which allows us to
    > test whether imposed biological constraints are sufficient for the
    > computations JEPA discovers

-   **Connects to neuroscience**: Learned circuits can be compared to
    > known biological motifs

# Research Questions: 

Main question:

-   Can spike-based neural networks, constrained by biological dynamics
    > (temporal integration, spike timing, refractory periods), learn to
    > reproduce the same latent predictive dynamics that LeJEPA models
    > learn from neural recordings?

Secondary questions:

-   What circuit connectivity patterns emerge to support this
    > computation?

-   Do these patterns recapitulate known biological motifs (E/I balance,
    > divisive normalization, etc.)?

-   Do SNNs trained on one brain state generalize to others, and if not,
    > what circuit mechanisms differ?

# Methodology:

Our approach proposes two phases:

Phase 1: *[Teacher model discovers latent dynamics]{.underline}*

Train a LeJEPA model on Allen Institute Neuropixels dataset (Visual
Behavior) of mice visual cortex & subcortical structures while
performing change detection task.

Training:

-   Embed neural population data into latent space

    -   Spike data binned (10ms) OR use similar temporal aggregation
        > method as
        > [[POSSM]{.underline}](https://arxiv.org/abs/2506.05320) paper

    -   Data is region / task specific

-   Predict future latent spaces via JEPA

    -   Use [[LeJEPA]{.underline}](https://arxiv.org/abs/2511.08544) for
        > proved solution to representation collapse

Phase 2: *[Distilling to bio-constrained SNNs]{.underline}*

Train a SNN to produce the same latent representation learned from the
teacher model, while using constraints for biological plausibility.

Training:

-   Use LIF neurons where: input = spike trains (binned same as
    > teacher); output = latent representations

-   Matches teacher latent representations via Rotationally-invariant
    > Canonical Correlation Analysis (CCA)

-   Regularization: Homeostatic firing rate penalty (keep mean firing
    > rate in 5-20Hz biological range to prevent quiescent collapse)

-   Gradients: Surrogate gradient method (e.g., fast sigmoid) to
    > backpropagate through discrete spikes

*[Future Extensions:]{.underline}*

-   Compare neuron models (LIF → Izhikevich) to test effect of
    > biophysical complexity

-   Test cross-state generalization (train on active behavior, test on
    > passive viewing)

-   Test across several sessions & brain regions

# Resources:

-   **DATASET**: Allen Institute Neuropixels Visual Behavior

    -   [[link]{.underline}](https://brain-map.org/our-research/circuits-behavior/visual-behavior)

    -   200,000 recorded neurons

    -   153 experimental sessions

    -   Visual areas & subcortical structures (thalamus, hippocampus,
        > midbrain)

    -   Sessions includes a passive stimulus replay block that allows
        > investigation of task-dependent modulation sensory and
        > behavioral
        > coding![](media/image1.png){width="2.734063867016623in"
        > height="2.0447058180227473in"}

-   **COMPUTATION:** Google Colab

    -   Need to find a way to handle dataset sizes (e.g. PyTorch
        > DataLoaders)

# Metrics of Success:

We define success through six sequential milestones, each with
quantitative criteria:

[*Phase 0: Data Pipeline & Infrastructure*
**[(COMPLETED)]{.mark}**]{.underline}

-   Download and preprocess Allen Institute Visual Behavior Neuropixels
    > dataset

-   Implement spike binning (10ms) with verification of firing rate
    > distributions

-   Validate data quality: mean firing rates in biological range, stable
    > across recording

-   *Success metric:* Clean pipeline producing correctly formatted input
    > tensors

[Phase 1: LeJEPA Baseline]{.underline}

-   Train Transformer-based JEPA to predict future latent states from
    > neural data

-   Validate predictive performance on held-out time windows

-   *Success metrics:*

    -   Prediction MSE significantly better than chance

    -   Latent space shows temporal structure

    -   No representational collapse

[Phase 2: SNN Proof of Concept (\"Student\" Model)]{.underline}

-   Train basic SNN (LIF neurons, no biological constraints yet) to
    > match JEPA latents

-   *Success metrics:*

    -   CCA similarity \> 0.7 between SNN and JEPA latents on held-out
        > data

    -   SNN maintains stable firing

    -   Prediction performance on downstream task (decoding; stimulus
        > identity & behavioral choice) within 10% of JEPA

[Phase 3: Biological Constraint Analysis]{.underline}

-   Add constraints (e.g., homeostatic firing rate penalty, Dale\'s
    > principle)

-   Compare constrained vs. unconstrained SNN performance

-   *Success metrics:*

    -   Constrained SNN achieves \>80% of unconstrained performance

    -   Firing rates remain in biological range

    -   Learned connectivity shows interpretable structure (E/I balance,
        > sparse connectivity)

[Phase 4: Cross-State Generalization (Extension)]{.underline}

-   Train on active behavior blocks, test on passive viewing (& vice
    > versa)

-   *Success metrics:*

    -   Within-state CCA similarity \> 0.7 (replication of Phase 2)

    -   Cross-state CCA similarity \> 0.5 (demonstrates partial
        > generalization)

    -   Identify which circuit parameters differ across states

[Phase 5: Model Complexity Comparison **(If Time Permits)**]{.underline}

-   Compare LIF vs. other neuron models

-   *Success metrics:*

    -   Quantify performance vs. computational cost tradeoff

    -   Identify which dynamics (bursting, adaptation, etc.) improve
        > latent matching

# Related Work

**[Self-Supervised Learning on Neural Data]{.underline}**

Recent work has applied self-supervised learning to neural recordings,
including BrainBERT for intracranial field potentials and Brain-JEPA for
fMRI data. These methods use predictive objectives (masked prediction,
future forecasting) to learn representations without task labels.
However, these approaches use standard deep learning architectures
(Transformers, VAEs) that lack biological constraints and don\'t reveal
interpretable circuit mechanisms.

**[Predictive Models of Neural Dynamics]{.underline}**

LFADS pioneered latent dynamics modeling for single-trial spike data,
while the Neural Data Transformer (NDT) provided a non-recurrent
alternative with faster inference. More recent foundation models achieve
strong cross-subject generalization by pretraining on massive datasets
(\~135,000 neurons). While these models excel at prediction and
generalization, their black-box nature limits mechanistic
interpretability.

**[Spiking Neural Networks and Biological Constraints]{.underline}**

SNNs trained with biologically plausible learning rules (STDP, E/I
balance) have been applied to neural data, with some work showing
emergence of biological tuning properties. Knowledge distillation from
Transformers to SNNs (LDD, CKDSNN) has succeeded in computer vision
tasks. However, this distillation approach has not been applied to
learning latent dynamics from real neural recordings.

**[Interpretable Circuit Mechanisms]{.underline}**

Recent work on Recurrent Mechanistic Models (RMMs) combines predictive
power with mechanistic interpretability by learning conductance-based
dynamics from intracellular recordings. Sparse autoencoders have also
been applied to extract interpretable features from neural network
representations trained on neural data.

**[Our Contribution]{.underline}**

To our knowledge, no prior work has combined JEPA-style predictive
learning on spike recordings with biologically-constrained SNNs for
mechanism discovery. This intersection, using teacher-student
distillation to test whether SNNs can implement the same latent dynamics
that Transformers discover, represents a novel approach to understanding
the computational sufficiency of biological constraints and extracting
interpretable circuit motifs from real neural data.