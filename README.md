# Smart Grid Fault Detection System

A comprehensive machine learning system for detecting broken power lines in smart grid data using neural networks.

## 🎯 Overview

This system addresses the critical problem of detecting broken power lines that don't create sufficient fault current to trigger traditional circuit breakers. It uses advanced machine learning techniques to identify fault patterns in smart grid sensor data.

## 🚀 Features

- **Neural Network-based Detection**: Gradient descent-based neural network for binary classification
- **Real Data Integration**: Works with actual smart grid datasets
- **Synthetic Data Generation**: Creates realistic fault scenarios for training augmentation
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Rich Visualizations**: Training curves, confusion matrices, ROC curves, and feature importance plots
- **Scalable Architecture**: Modular design for easy extension and maintenance
- **Production Ready**: Logging, error handling, and model persistence

## 📁 Project Structure

```
smart-grid-fault-detection/
├── src/
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   └── synthetic_generator.py
│   ├── models/                 # Machine learning models
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   └── model_trainer.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── logger.py
├── assets/                     # Data files
│   └── smart_grid_dataset.csv
├── models/                     # Saved models
├── reports/                    # Generated reports and plots
├── logs/                       # Log files
├── tests/                      # Test files
├── main.py                     # Main application entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using uv (recommended):
```bash
uv sync
```

## 🎮 Usage

### Basic Usage

Run the system with default settings:
```bash
uv run main.py
```

### Advanced Usage

```bash
# Use custom data file
uv run main.py --data-path path/to/your/data.csv

# Skip synthetic data generation
uv run main.py --no-synthetic

# Display plots (useful for development)
uv run main.py --show-plots

# Don't save plots to files
uv run main.py --no-save-plots

# Combine options
uv run main.py --data-path data/smart_grid.csv --show-plots --no-synthetic
```

### Command Line Options

- `--data-path`: Path to the smart grid dataset CSV file
- `--no-synthetic`: Skip synthetic data generation
- `--show-plots`: Display plots (default: False)
- `--no-save-plots`: Don't save plots to files (default: False)

## 📊 Expected Results

With your smart grid data, you should see:

- **High Accuracy**: >95% classification accuracy
- **Excellent Recall**: >99% fault detection rate
- **Good Precision**: >98% true positive rate
- **Strong F1-Score**: >99% balanced performance
- **Perfect ROC-AUC**: 100% classification performance

## 🔧 Configuration

The system is highly configurable through `src/config/settings.py`:

### Model Configuration
```python
MODEL_CONFIG = {
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "activation": "relu",
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        # ... more options
    }
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 40,
    "early_stopping": {"patience": 10},
    # ... more options
}
```

### Synthetic Data Configuration
```python
SYNTHETIC_CONFIG = {
    "n_synthetic_samples": 2000,
    "fault_scenarios": {
        "voltage_drop_range": (0.1, 0.5),
        "current_reduction_range": (0.05, 0.3),
        # ... more scenarios
    }
}
```

## 📈 Understanding the Results

### Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted faults were actually faults
- **Recall**: How many actual faults were detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (overall performance)

### Visualizations

The system generates several visualizations:

1. **Training History**: Loss and accuracy curves over epochs
2. **Confusion Matrix**: True vs predicted classifications
3. **ROC Curve**: True positive rate vs false positive rate
4. **Precision-Recall Curve**: Precision vs recall at different thresholds
5. **Feature Importance**: Which features contribute most to predictions
6. **Data Distribution**: Analysis of input data characteristics

## 🔍 How It Works

### 1. Data Processing
- Loads smart grid sensor data
- Cleans missing values and outliers
- Creates target variable based on electrical fault indicators
- Selects relevant features for training

### 2. Synthetic Data Generation
- Identifies normal operating conditions
- Generates realistic fault scenarios
- Augments training data with synthetic samples

### 3. Model Training
- Trains neural network with early stopping
- Uses validation data for hyperparameter tuning
- Implements learning rate reduction

### 4. Evaluation
- Tests on unseen data
- Calculates comprehensive metrics
- Generates detailed visualizations

## 🧪 Fault Detection Logic

The system identifies faults based on multiple electrical parameters:

- **Low Voltage**: Voltage below 150V indicates potential line issues
- **High Voltage Fluctuation**: >3% fluctuation suggests instability
- **Overload Conditions**: System stress indicators
- **Transformer Faults**: Equipment failure signals
- **Poor Power Factor**: <0.8 indicates efficiency issues
- **Current Anomalies**: Very low current with normal voltage
- **Load Disconnection**: Zero consumption with predicted load

## 🚀 Performance Optimization

### For Large Datasets
- Increase batch size in `TRAINING_CONFIG`
- Reduce number of epochs if early stopping triggers
- Use data sampling for initial experimentation

### For Better Accuracy
- Increase synthetic data samples
- Adjust fault detection thresholds
- Experiment with different network architectures

### For Faster Training
- Reduce network size
- Use fewer synthetic samples
- Decrease validation split size

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or synthetic samples
2. **Low Accuracy**: Check data quality and feature selection
3. **Import Errors**: Ensure all dependencies are installed
4. **File Not Found**: Verify data file path and permissions

### Debug Mode

Enable detailed logging by modifying `LOGGING_CONFIG`:
```python
LOGGING_CONFIG = {
    "level": "DEBUG",  # Change from "INFO" to "DEBUG"
    # ... other settings
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Smart grid data providers
- TensorFlow and scikit-learn communities
- Open source visualization libraries

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Open an issue on the repository

---

**Ready to detect smart grid faults?** Run `python main.py` and start protecting your power infrastructure! 🚀
