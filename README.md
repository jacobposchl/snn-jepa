# SNN-JEPA: Spiking Neural Networks - Joint-Embedding Predictive Architecture

SNN-JEPA is a self-supervised learning framework designed to learn compact, informative representations of high-dimensional neural population activity. It combines **Spiking Neural Networks (SNNs)** with the **Joint-Embedding Predictive Architecture (JEPA)** to model the temporal dynamics of brain activity recorded via Neuropixels.

## 🧠 Key Features

- **Spiking Neural Encoders**: Leverages `snntorch` to process raw neural spike trains, maintaining the temporal precision and biological realism of spiking dynamics.
- **Joint-Embedding Predictive Architecture (JEPA)**: Learns by predicting future latent states from past context, avoiding the need for expensive manual labels.
- **Isotropic Gaussian Regularization (SIGReg)**: Employs Sketched Isotropic Gaussian Regularization to prevent latent space collapse and ensure representations are well-distributed.
- **Multi-Modal Encoders**: Supports both SNNs and Transformer-based encoders for neural data.
- **Neuropixels Integration**: Built-in support for Allen Institute Visual Behavior Neuropixels datasets.

## 📁 Project Structure

- `src/models/`: Architecture definitions for `SNNEncoder`, `TransformerEncoder`, and `NeuralPredictor`.
- `src/losses/`: Implementation of `lejepa_loss` and `SIGReg` regularization.
- `src/data/`: Data loading and preprocessing pipelines for Neuropixels data.
- `src/plots/`: Visualization tools for latent space analysis, neural activity, and training diagnostics.
- `proof_of_concept_edit.py`: Main execution script for training and evaluating the model on a single session.

---

## 🚀 Remote GPU Setup via VS Code Tunnel

Running this project often requires significant GPU memory (e.g., NVIDIA A100). The following setup allows you to use Google Colab's powerful hardware while maintaining a professional development environment in VS Code.

### Phase 1: The "Power Switch" (Colab Notebook)

Run these cells in an A100-enabled Colab notebook to prepare the environment and open the tunnel.

1. **Clone the Repo**:
   ```python
   !git clone https://github.com/jacobposchl/snn-jepa
   %cd snn-jepa
   ```

2. **Mount Data**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Unzip your preprocessed data to the VM's local SSD for speed
   !unzip /content/drive/MyDrive/path/to/your/processed.zip -d /content/preprocessed_data
   ```

3. **Launch the Tunnel**:
   ```python
   !curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
   !tar -xf vscode_cli.tar.gz
   !./code tunnel
   ```
   *Follow the link provided in the output to authenticate with GitHub.*

### Phase 2: The "Control Room" (VS Code)

Once the tunnel is active, connect from your local VS Code:

1. **Connect**: 
   - Install the **Remote - Tunnels** extension in VS Code.
   - Click the blue icon (bottom-left) > **Connect to Tunnel**.
2. **Atomic Installation**: 
   Open the VS Code terminal (`Ctrl + ` `) and run the following to ensure the exact legacy versions required for AllenSDK compatibility:
   ```bash
   pip install torch torchvision torchaudio snntorch tqdm lsd timg pynwb SimpleITK allensdk==2.16.2 "numpy<1.24" "pandas<2.0.0"
   ```
3. **Run with Backend Override**:
   Since the tunnel is headless, you must override the Matplotlib backend to avoid errors:
   ```bash
   export MPLBACKEND=Agg
   python3 proof_of_concept_edit.py <SESSION_ID> --dataset-dir "/content/preprocessed_data/processed.pkl"
   ```

## 📊 Visualization

The project generates several diagnostic plots in the `runs/` directory, including:
- **Latent Space Trajectories**: Visualizing how neural states evolve over time.
- **Prediction Accuracy**: Comparing predicted future latents against actual encoded latents.
- **Firing Rate Distributions**: Ensuring the SNN maintains biological firing rates via homeostatic penalties.
