# LSTM + Decision Tree Integration

This directory contains an integrated machine learning pipeline that combines LSTM time series prediction with Decision Tree classification for hard drive failure prediction analysis.

## Files Overview

- **`smart.py`** - LSTM neural network for time series prediction of SMART attributes
- **`CT.py`** - Decision Tree classifier with support for LSTM prediction analysis
- **`integration_demo.py`** - Interactive demo showing different integration workflows
- **`README_Integration.md`** - This file

## Integration Workflow

The integration allows you to:

1. **Train LSTM** on time series SMART data to predict future values
2. **Generate Features** from LSTM predictions (prediction scores, trends, anomalies)
3. **Classify** using Decision Trees whether drives will fail based on LSTM-derived features

## Quick Start

### 1. Train LSTM Model (smart.py)

First, train the LSTM model on your SMART data:

```bash
python smart.py
```

This will:
- Load SMART data from `../../data`
- Train an LSTM to predict future SMART values
- Save the trained model to `models/LSTM/lstm_model.pth`
- Optionally export predictions for CT analysis

### 2. Run Decision Tree Analysis (CT.py)

Then analyze with Decision Trees in one of three ways:

#### Option A: Standard CT Analysis
```python
# In CT.py main() function, set:
use_lstm_predictions = False
```

#### Option B: CT Analysis on LSTM Predictions  
```python
# In CT.py main() function, set:
use_lstm_predictions = True
lstm_features_path = "ct_analysis/ct_features.csv"
```

#### Option C: Integrated Pipeline
```python
# In CT.py main() function, set:
use_lstm_predictions = True
# Ensure smart_model_path points to your trained LSTM model
```

### 3. Interactive Demo

Run the interactive demo to see all integration options:

```bash
python integration_demo.py
```

## Integration Features

### LSTM Prediction Features Generated

When LSTM predictions are converted for CT analysis, the following features are created:

- **`lstm_avg_prediction`** - Average prediction score across time steps
- **`lstm_max_prediction`** - Maximum prediction score
- **`lstm_min_prediction`** - Minimum prediction score  
- **`lstm_prediction_variance`** - Variance in predictions
- **`lstm_prediction_trend`** - Trend from first to last prediction
- **`lstm_high_risk_days`** - Number of days above risk threshold
- **`lstm_anomaly_score`** - Average standard deviation (anomaly indicator)
- **`smart_X_lstm_avg`** - Average LSTM prediction for each SMART attribute
- **`smart_X_lstm_max`** - Maximum LSTM prediction for each SMART attribute
- **`smart_X_lstm_trend`** - Trend in LSTM predictions for each SMART attribute

### Feature Selection Methods

The CT analysis supports different feature selection approaches:

- **`'all'`** - Use all available features (LSTM + original SMART)
- **`'lstm_only'`** - Use only LSTM-derived features
- **`'smart_only'`** - Use only original SMART attributes (no LSTM features)

## Output Files

### LSTM Outputs (smart.py)
- `models/LSTM/lstm_model.joblib` - Trained LSTM model
- `models/LSTM/lstm_model_metrics.json` - Model training metrics
- `models/LSTM/final_loss.png` - Training loss curve
- `models/LSTM/test_predictions.png` - Prediction visualization

### CT Analysis Outputs
- `models/DT/best_gini_tree.joblib` - Best Gini decision tree
- `models/DT/best_entropy_tree.joblib` - Best entropy decision tree
- `ct_lstm_analysis/ct_lstm_analysis_results.json` - Analysis results
- `ct_lstm_analysis/feature_importance.png` - Feature importance plot

### Integration Outputs
- `ct_analysis/lstm_predictions.csv` - Raw LSTM predictions
- `ct_analysis/ct_features.csv` - CT-compatible features
- `integrated_analysis/comprehensive_analysis_report.json` - Full pipeline results

## Key Functions

### smart.py Functions

- **`generate_lstm_predictions()`** - Generate predictions and save to CSV
- **`create_ct_compatible_features()`** - Convert predictions to CT features
- **`export_for_ct_analysis()`** - Complete export pipeline

### CT.py Functions

- **`load_lstm_predictions()`** - Load LSTM features from CSV
- **`merge_lstm_with_ground_truth()`** - Merge with failure labels
- **`prepare_lstm_features_for_ct()`** - Prepare features for training
- **`analyze_lstm_predictions_with_ct()`** - Complete CT analysis pipeline
- **`run_integrated_lstm_ct_analysis()`** - Full integration pipeline

## Configuration

### smart.py Configuration
```python
# In smart.py main section:
test_existing = False  # Set True to test existing model
days_to_train = 3      # Input sequence length
days_to_predict = 2    # Prediction horizon
path = "../../data/data_Q1_2025"  # Data path
export_for_ct = True   # Export for CT analysis
```

### CT.py Configuration
```python
# In CT.py main() function:
use_lstm_predictions = True  # Use LSTM integration
lstm_features_path = "ct_analysis/ct_features.csv"
smart_model_path = "models/LSTM/lstm_model.pth"
data_path = "../../data/data_Q1_2025"
```

## Example Workflow

```python
# 1. Train LSTM and export predictions
import smart
predictions_df, ct_features_df = smart.export_for_ct_analysis(
    model_path="models/LSTM/lstm_model.pth",
    dataset_path="../../data/data_Q1_2025",
    output_dir="ct_analysis"
)

# 2. Analyze with Decision Trees
import CT
results = CT.analyze_lstm_predictions_with_ct(
    lstm_features_path="ct_analysis/ct_features.csv",
    ground_truth_path="../../data/data_Q1_2025",
    feature_selection_method='all',
    output_dir="ct_lstm_analysis"
)

print(f"Best accuracy: {max(results['gini_accuracy'], results['entropy_accuracy']):.4f}")
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure LSTM model is trained first by running `smart.py`

2. **Path errors**: Check that data paths in both scripts point to the correct directories

3. **Missing dependencies**: Ensure all required packages are installed:
   ```bash
   pip install torch pandas scikit-learn imbalanced-learn tqdm matplotlib joblib
   ```

4. **Memory issues**: Reduce batch size or chunk size if running out of memory

### Error Messages

- **"LSTM features file not found"**: Run `smart.py` with `export_for_ct = True` first
- **"Could not import smart module"**: Ensure both files are in the same directory
- **"No valid sequences found"**: Check that your data has sufficient time series length

## Performance Tips

1. **Feature Selection**: Use `'lstm_only'` for faster training if original SMART features aren't needed

2. **Data Size**: For large datasets, consider sampling or using chunked processing

3. **Model Complexity**: Adjust Decision Tree parameters (`depth`, `leaf`) based on feature count

4. **GPU Usage**: LSTM training automatically uses GPU if available

## Next Steps

After running the integration, you can:

1. **Analyze Feature Importance** - See which LSTM-derived features are most predictive
2. **Tune Hyperparameters** - Optimize both LSTM and Decision Tree parameters  
3. **Compare Approaches** - Evaluate LSTM+CT vs. standalone approaches
4. **Deploy Models** - Use the best performing pipeline for production prediction

## Contact

For questions or issues with the integration, please check the individual script documentation and error messages for specific guidance.
