# Scalable Training of Transformer Language Models

This project provides a framework for training and profiling a Transformer-based language model. It describes experiments with training a model on the WikiText-103 dataset, with the goal of establishing a baseline, measuring system performance, and then applying optimizations such as gradient checkpointing and FlashAttention to improve efficiency. This work is part of an effort to learn how to train modern language models in a scalable way.

## Baseline Configuration

For the baseline experiment, a 12-layer Transformer model was used with the WikiText-103 dataset. The model has 12 attention heads and an embedding size of 768.

-   **Hardware**: Training was conducted on the TACC cluster using an NVIDIA A100 GPU (40GB), 24 CPU cores, and 96GB of RAM.
-   **Software**: The implementation relies on Hugging Face libraries including `transformers`, `datasets`, `accelerate`, and `tokenizers`, alongside standard data science tools like `pandas` and `matplotlib`.
-   **Training Parameters**: The training ran for 4000 steps with a batch size of 8, a sequence length of 512, and the AdamW optimizer with a cosine learning rate schedule.

## Setup and Installation

### Prerequisites

-   Python 3.10+
-   `make` build automation tool
-   An NVIDIA GPU with CUDA is highly recommended for performance.

### 1. Create a Virtual Environment

It is strongly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with system-wide packages.

```bash
# Navigate to the project root
cd /path/to/your/project

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configure Cache Paths

The project uses environment variables to control where Hugging Face datasets and models are cached. This prevents filling up your home directory.

```bash
# Copy the example environment file
cp .env.example .env

# Source the file to export the variables into your current shell session
# You will need to do this every time you open a new terminal.
source .env
```

### 3. Install Dependencies

With the virtual environment activated, install the required Python packages using the provided `Makefile` target.

```bash
make install
```

## Usage

The project uses a `Makefile` to simplify common tasks like training, profiling, and plotting.

### Training

Several training configurations are available:

-   **Baseline Training**: Train the standard model as defined in `configs/base.yaml`.
    ```bash
    make train
    ```

-   **Train with Gradient Checkpointing**: Enable gradient checkpointing to save memory at the cost of a small performance hit.
    ```bash
    make train-ckpt
    ```

-   **Train with Flash Attention & AMP**: Train using Flash Attention and Automatic Mixed Precision (AMP) for significant speedups and reduced memory usage.
    ```bash
    make train-flash
    ```

-   **Train a Larger Model**: As an example, train a model with more layers and a larger embedding dimension.
    ```bash
    make train-larger
    ```

### Profiling

The following commands run targeted profiling sweeps to measure performance under different conditions. The results are saved as CSV files in `results/logs/`.

-   **Profile Sequence Length**: Measure throughput and memory as sequence length changes.
    ```bash
    make profile-len
    ```

-   **Profile Batch Size (No AMP)**: Measure performance as batch size changes, without mixed precision.
    ```bash
    make profile-batch-noamp
    ```

-   **Profile Batch Size (With AMP)**: Measure performance as batch size changes, with mixed precision enabled.
    ```bash
    make profile-batch-amp
    ```

-   **Profile AMP On/Off**: A direct comparison of training with and without AMP.
    ```bash
    make profile-amp
    ```

### Generating Plots

After running the training and profiling commands, you can generate a series of plots to visualize the results.

```bash
make plots
```

This command reads the various CSV log files from `results/logs/` and generates the following plots in `results/plots/`:
-   GPU Memory vs. Sequence Length
-   GPU Memory vs. Batch Size (AMP vs. No AMP)
-   Throughput vs. Batch Size (AMP vs. No AMP)
-   Baseline Training Progress (Perplexity & Accuracy vs. Steps)
-   Optimization Comparisons (Baseline vs. Grad Ckpt, Baseline vs. Flash Attn)

### Cleaning Up

To remove all generated logs and plots from the `results/` directory:

```bash
make clean
```
