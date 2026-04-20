# MAML Metagenomics Classification Pipeline

## Overview

A **Model-Agnostic Meta-Learning (MAML)** pipeline for sparse microbiome classification with state-of-the-art architectures and rigorous evaluation methodology. This notebook implements robust few-shot learning techniques to classify microbiome samples with minimal labeled data, addressing real-world challenges in metagenomics research.

## Key Features

###  Core Innovations

- **Novel Model Architectures**
  - **TabularResNet Encoder**: Residual blocks with skip connections optimized for tabular bacterial abundance data
  - **Prototypical Networks**: Metric-learning head with learnable temperature scaling for stable cosine similarity logits
  - **GatedMLP**: Feature gating mechanism with LayerNorm and Dropout to dynamically suppress irrelevant bacteria
  - **StandardMLP & ProtoNetEncoder**: Baseline architectures for comparative analysis

- **Compositional Mixup Augmentation**
  - Intra-class data augmentation generating biologically plausible synthetic samples
  - Solves extreme sparsity in few-shot settings (expands 3-shot support sets to dozens of effective samples)
  - Beta-distributed convex combinations of support samples preserve biological properties

- **Rigorous Evaluation Fixes**
  - **Query Set Isolation**: Evaluation metrics computed only on unseen test queries (not on memorized support sets)
  - **5-Seed Cross-Validation**: Validates architectural robustness across multiple random initializations
  - Eliminates data leakage artifacts from naive evaluation protocols

###  Biological Data Processing

- **Centered Log-Ratio (CLR) Transformation**
  - Pseudo-count addition (1e-5) prevents log(0) numerical issues
  - Standard z-score normalization ensures feature stability
  - Preserves compositional constraints inherent to microbiome sequencing data
  - Proper handling of zero-inflated bacterial abundance distributions

## Architecture

### Model Components

```
Input (N bacterial species)
    ↓
TabularResNetEncoder / GatedMLP / ProtoNetEncoder
    ├─ Input projection with BatchNorm
    ├─ Residual blocks (2x)
    └─ Output embedding dimension=64
    ↓
PrototypicalHead
    ├─ Prototype calculation (mean per class)
    ├─ Cosine similarity with normalization
    └─ Temperature-scaled logits
    ↓
Predictions (Class logits)
```

### Meta-Learning Flow

```
Meta-Training Loop (Multiple Tasks/Episodes)
    ├─ Sample Task: N-way K-shot classification
    │   ├─ Support Set: N classes × K samples
    │   └─ Query Set: N classes × Q samples
    ├─ Inner Loop (Task Adaptation)
    │   ├─ Apply Intra-class Mixup (if training)
    │   ├─ Multiple gradient steps with fast weights
    │   └─ Functional forward pass maintains MAML compatibility
    └─ Outer Loop (Meta-update)
        ├─ Backprop through inner loop (2nd-order MAML)
        └─ Update model parameters
```

## Project Structure

### Cells Overview

1. **Cell 1: Environment Setup and Imports**
   - Package installation (numpy, pandas, torch, scikit-learn, captum, etc.)
   - Global seed configuration for reproducibility
   - GPU/CPU device detection and setup

2. **Cell 2: Model Architectures**
   - `TabularResNetBlock`: Residual block implementation
   - `TabularResNetEncoder`: Lightweight encoder with 2 residual blocks
   - `PrototypicalHead`: Prototype-based classification head
   - `MetagenomicsFewShotModel`: Full integration of encoder + head

3. **Cell 3: Data Loading and Preprocessing**
   - `file_to_dataframe()`: Parse microbiome data files
   - `preprocess_metaml_data()`: CLR transformation + standardization
   - `OptimizedSpeciesDataset`: PyTorch dataset wrapper

4. **Cell 4: Samplers and Dataloaders**
   - `FastNShotTaskSampler`: Efficient N-shot task generation
   - Episode-based sampling (multiple tasks per epoch)
   - Class-balanced sampling strategy

5. **Cell 5: Meta-Learning Core**
   - `meta_gradient_step()`: Main MAML computation loop
   - `apply_intra_class_mixup()`: Synthetic sample generation
   - `train()`: Full training iteration
   - `evaluate()`: Validation/test evaluation
   - Metrics: accuracy, F1-score, ROC-AUC calculation

6. **Cell 6: MAML-Specific Implementations**
   - Functional forward passes for fast weight updates
   - Gradient hook replacement for 1st-order MAML
   - 2nd-order MAML with higher-order derivatives

7. **Cell 7: Main Execution Pipeline**
   - Data loading from Kaggle or local files
   - k-fold cross-validation
   - Model training and evaluation across 5 seeds
   - Baseline comparisons (Random Forest, Transfer Learning)
   - Model checkpointing

8. **Cell 8: Results Aggregation**
   - Cross-seed statistics computation
   - Mean accuracy, F1, ROC-AUC across all seeds
   - Comprehensive metrics reporting

9. **Cell 9: Validation Suite and Visual Diagnostics**
   - Classification reports with precision/recall/F1
   - Confusion matrix heatmaps
   - ROC curves (multi-class compatible)
   - Precision-Recall curves
   - Per-seed validation visualization

## Key Improvements Over Original

| Aspect | Original | Improved |
|--------|----------|----------|
| **Evaluation** | Combined support + query sets (leakage) | Query-set only (strict isolation) |
| **Validation** | Single seed (1729) | 5-seed cross-validation |
| **Data Augmentation** | None (extreme overfitting) | Compositional mixup (synthetic samples) |
| **Architecture** | Basic MLP | GatedMLP + TabularResNet + PrototypicalNet |
| **Normalization** | Pre-learned scaler | Per-episode CLR + Layer normalization |
| **Accuracy** | ~87% (inflated, leakage) | Realistic metrics (proper evaluation) |

## Usage

### Prerequisites

```python
# Core dependencies
numpy, pandas, torch, scikit-learn
matplotlib, seaborn, plotnine
captum (for feature attribution, optional)
```

### Quick Start

1. **Load and execute Cell 1** to set up environment
2. **Execute Cells 2-6** to define models and utilities
3. **Execute Cell 7** for main meta-learning pipeline
   - Automatically handles data loading
   - Trains across all 5 seeds
   - Saves best models per seed
4. **Execute Cell 8** to aggregate cross-seed metrics
5. **Execute Cell 9** for visualization and diagnostics

### Example: Custom Microbiome Data

```python
# Load your data
df = pd.read_csv("your_microbiome_data.csv", index_col=0)
species_cols = df.columns[:-1]  # All but target
target_col = df.columns[-1]     # Target column

# Preprocess
X_processed = preprocess_metaml_data(df[species_cols])

# Create dataset
dataset = OptimizedSpeciesDataset(df, species_cols, target_col)

# Configure task parameters
n_way = 2      # Binary classification
k_shot = 3     # Few shots per class
q_queries = 5  # Query samples per class
```

### Hyperparameters Configuration

```python
# Meta-learning parameters
meta_batch_size = 4        # Number of tasks per meta-batch
inner_lr = 0.01            # Inner loop learning rate
inner_train_steps = 5      # Adaptation steps per task
outer_lr = 0.001           # Outer loop (meta) learning rate
epochs = 100               # Meta-training epochs

# Mixup parameters
num_synthetic_per_class = 2
alpha = 0.4                # Beta distribution parameter

# Model parameters
embed_dim = 64             # Embedding dimension
dropout_p = 0.3            # Dropout probability
temperature = 10.0         # Prototypical head temperature
```

## Performance Metrics

### Evaluation Framework

- **Accuracy**: Query-set only (strict, no memorization bias)
- **F1-Score**: Macro-averaged for class balance consideration
- **ROC-AUC**: Probability calibration and ranking metric
- **Confusion Matrix**: Per-class classifier behavior

### Cross-Seed Robustness

Results are reported as mean ± std across 5 seeds to ensure:
- Architectural stability
- Reproducibility
- Statistical significance
- Generalization capacity

## Results Examples

```
========================================
FINAL AGGREGATED METRICS ACROSS ALL SEEDS
========================================
Accuracy:  [Mean] % ± [Std] %
F1 Score:  [Mean] ± [Std]
ROC AUC:   [Mean] ± [Std]
```

Per-seed validation reports include:
- Classification metrics per class
- Confusion matrices with annotations
- ROC and PR curves with AUC scores
- Visual diagnostics for each seed

## Advanced Features

### 1. Functional Forward Pass
Enables weight updates compatible with automatic differentiation for 2nd-order MAML:

```python
y_pred = model.functional_forward(x, fast_weights)
# fast_weights: OrderedDict of adapted parameters
```

### 2. Gradient Hook Replacement
Implements 1st-order MAML approximation for memory efficiency:

```python
# Aggregates task gradients without 2nd-order computation
sum_task_gradients = average_gradients_across_tasks()
```

### 3. Feature Gating Mechanism
Learns which bacterial features are predictive:

```python
gate = sigmoid(W_gate @ x)
x_gated = x * gate  # Element-wise feature masking
```

## Data Format Requirements

### Input CSV/TSV Format
```
Sample_ID    Bacteria_1    Bacteria_2    ...    Disease_Status
sample_001   1200          450           ...    healthy
sample_002   850           1100          ...    diseased
```

### Expected Preprocessing

- **Rows**: Individual patients/samples
- **Columns**: Bacterial taxon relative abundances or counts
- **Target**: Binary/multi-class disease label
- **Missing Values**: Should be handled before loading

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `meta_batch_size` or `embed_dim` |
| Poor generalization | Increase `inner_train_steps` or `num_synthetic_per_class` |
| NaN losses | Check for log(0) in preprocessing; verify pseudocount |
| Imbalanced accuracy | Adjust `alpha` in mixup or use class weighting |

## References & Methodology

### Key Publications
- MAML (Model-Agnostic Meta-Learning): Finn et al. (2017)
- Prototypical Networks: Snell et al. (2017)
- Compositional microbiome analysis: Aitchison (1982)

### Validation Best Practices
- Query-set isolation prevents leakage
- Cross-seed validation ensures reproducibility
- Compositional transforms respect microbiome properties

## File Outputs

- **Model Checkpoints**: `MetagenomicsFewShotModel_seed_*.pt`
- **Validation Plots**: Generated in notebook cells
- **Metrics Reports**: Printed to notebook output
- **Confusion Matrices**: Stored as numpy arrays

## License & Attribution

This pipeline combines state-of-the-art meta-learning with domain-specific microbiome preprocessing. Designed for biological researchers and ML practitioners working with sparse, high-dimensional compositional data.

---

**Last Updated**: April 2026  
**Validation**: ✅ 5-Seed Cross-Validation Complete
