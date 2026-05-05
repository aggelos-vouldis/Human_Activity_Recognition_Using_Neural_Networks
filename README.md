# Human Activity Recognition - Neural Network Classifier

This project implements a neural network-based classifier for human activity recognition using the HAR-PUC-Rio dataset. The model distinguishes between five different activities: sitting down, standing up, standing, walking, and sitting.

## Dataset:

The model uses the HAR-PUC-Rio dataset which contains motion sensor data from 4 inertial measurement units (IMUs) placed on different body positions. Each sample includes:

- Gender (Woman/Man → 1/2)
- Age, Height, Weight, BMI
- 12 sensor readings (X, Y, Z coordinates from 4 sensors)

## Features:

- Data preprocessing with undersampling to handle class imbalance
- Quantile transformation for feature normalization
- Neural network classifier with:
  - Input layer (17 features)
  - Hidden layer (22 nodes, ReLU activation, L2 regularization)
  - Output layer (5 nodes, softmax activation)
- Nadam optimizer with EMA momentum
- 5-fold cross-validation for robust evaluation
- Early stopping to prevent overfitting

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras
```

## Usage

### Train and Evaluate Model

```bash
python project_1.py
```

Output includes per-fold and final average metrics:

- MSE (Mean Squared Error)
- Accuracy
- CE (Categorical Cross-Entropy)

### Generate Learning Curves

```bash
python Learning_curve_maker.py
```

This plots training vs validation accuracy across different training set sizes.

### Test Transformations

```bash
python transformations_testing.py
```

### Configuration

Key hyper-parameters in project_1.py:

```python
HIDDEN_LAYER_NODES = 22
EPOCHS = 500
BATCH_SIZE = 38
LEARNING_RATE = 0.001
MOMENTUM = 0.2
REGULATION = 0.1      # L2 regularization
```

### Class Mapping

|  Activity   | Label |
| :---------: | :---: |
| sittingdown |   1   |
| standingup  |   2   |
|  standing   |   3   |
|   walking   |   4   |
|   sitting   |   5   |

### Preprocessing Steps

1. Label encoding - Convert gender and activity strings to integers
2. Undersampling - Balance all 5 activity classes to the minority class size
3. Quantile transformation - Map features to a uniform distribution
4. One-hot encoding - Convert class labels to categorical format

## Results

The model outputs:

- Fold-by-fold metrics (MSE, Accuracy, CE)
- Final cross-validated out-of-sample performance\

Example output format:

```
#Fold 1 (MSE): 0.12345 (Accuracy): 0.87654 (CE): 0.45678
Final, out of sample (MSE): 0.12345 (Accuracy): 0.87654 (CE): 0.45678
```

## Notes

- The `user` column is dropped as it doesn't contribute to activity classification
- Early stopping patience is set to 200 epochs
- Model uses Exponential Moving Average for smoother weight updates

## License

This project is for educational purposes. The HAR-PUC-Rio dataset has its own usage terms - please cite appropriately if used in research.
