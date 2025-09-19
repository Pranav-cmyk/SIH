"""
Smart Grid Fault Detection System - Main Application

This is the main entry point for the Smart Grid Fault Detection system.
It orchestrates the complete pipeline from data loading to model training and evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import DataProcessor, SyntheticDataGenerator
from src.models import ModelTrainer
from src.utils import VisualizationManager, setup_logging
from src.config import (
    DATASET_CONFIG, 
    SYNTHETIC_CONFIG,
    MODELS_DIR,
    REPORTS_DIR
)

logger = setup_logging()


class SmartGridFaultDetectionApp:
    """
    Main application class for Smart Grid Fault Detection.
    
    This class orchestrates the complete pipeline:
    1. Data loading and preprocessing
    2. Synthetic data generation
    3. Model training
    4. Evaluation and visualization
    """
    
    def __init__(self, data_path: Optional[Path] = None, 
                 generate_synthetic: bool = True,
                 save_plots: bool = True,
                 show_plots: bool = False):
        """
        Initialize the application.
        
        Args:
            data_path: Path to the smart grid dataset
            generate_synthetic: Whether to generate synthetic fault data
            save_plots: Whether to save visualization plots
            show_plots: Whether to display plots
        """
        self.data_path = data_path or DATASET_CONFIG["raw_data_file"]
        self.generate_synthetic = generate_synthetic
        self.save_plots = save_plots
        self.show_plots = show_plots
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.synthetic_generator = SyntheticDataGenerator()
        self.model_trainer = ModelTrainer()
        self.visualizer = VisualizationManager(save_plots=save_plots, show_plots=show_plots)
        
        logger.info("Smart Grid Fault Detection Application initialized")
    
    def run_pipeline(self) -> dict:
        """
        Run the complete fault detection pipeline.
        
        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Smart Grid Fault Detection Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            self.data_processor.load_data(self.data_path)
            self.data_processor.clean_data()
            self.data_processor.create_target_variable()
            X, y = self.data_processor.select_features()
            
            # Get data summary
            data_summary = self.data_processor.get_data_summary()
            logger.info(f"Data Summary: {data_summary}")
            
            # Step 2: Generate synthetic data (optional)
            if self.generate_synthetic:
                logger.info("Step 2: Generating synthetic fault data")
                X, y = self.synthetic_generator.generate_synthetic_faults(X, y)
            else:
                logger.info("Step 2: Skipping synthetic data generation")
            
            # Step 3: Train neural network model
            logger.info("Step 3: Training neural network model")
            results = self.model_trainer.train_neural_network(X, y)
            
            # Step 4: Create visualizations
            logger.info("Step 4: Creating visualizations")
            self._create_visualizations(results, X, y)
            
            # Step 5: Save model and generate report
            logger.info("Step 5: Saving model and generating report")
            model_path = self.model_trainer.save_best_model()
            
            # Get final summary
            final_summary = self._generate_final_summary(results, data_summary, model_path)
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _create_visualizations(self, results: dict, X, y) -> None:
        """Create all visualizations."""
        # Data distribution analysis
        data_with_target = X.copy()
        data_with_target['Broken_Line_Fault'] = y
        self.visualizer.plot_data_distribution(data_with_target, 'Broken_Line_Fault')
        
        # Model evaluation visualizations
        if 'test_data' in results and 'predictions' in results:
            X_test, y_test = results['test_data']
            y_pred = results['predictions']
            y_pred_proba = results['probabilities']
            
            # Confusion matrix
            self.visualizer.plot_confusion_matrix(y_test, y_pred, "Neural Network")
            
            # ROC curve
            self.visualizer.plot_roc_curve(y_test, y_pred_proba, "Neural Network")
            
            # Precision-Recall curve
            self.visualizer.plot_precision_recall_curve(y_test, y_pred_proba, "Neural Network")
        
        # Feature importance
        try:
            feature_names = X.columns.tolist()
            importance_scores = self.model_trainer.get_feature_importance(
                "neural_network", feature_names
            )
            self.visualizer.plot_feature_importance(importance_scores, "Neural Network")
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
        
        # Model comparison (if multiple models)
        comparison_df = self.model_trainer.get_model_comparison()
        if not comparison_df.empty:
            self.visualizer.plot_model_comparison(comparison_df)
    
    def _generate_final_summary(self, results: dict, data_summary: dict, model_path: str) -> dict:
        """Generate final summary of the pipeline."""
        test_metrics = results.get('test_metrics', {})
        
        summary = {
            "data_summary": data_summary,
            "model_performance": test_metrics,
            "model_path": model_path,
            "reports_directory": str(REPORTS_DIR),
            "classification_report": self.model_trainer.get_classification_report(),
            "training_summary": self.model_trainer.get_training_summary()
        }
        
        # Log final results
        logger.info("\n" + "=" * 40)
        logger.info("FINAL RESULTS")
        logger.info("=" * 40)
        logger.info(f"Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {test_metrics.get('precision', 0):.4f}")
        logger.info(f"Recall:    {test_metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score:  {test_metrics.get('f1_score', 0):.4f}")
        logger.info(f"ROC-AUC:   {test_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Reports saved to: {REPORTS_DIR}")
        
        return summary


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(
        description="Smart Grid Fault Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --data-path data/smart_grid.csv   # Use custom data file
  python main.py --no-synthetic                    # Skip synthetic data generation
  python main.py --show-plots                      # Display plots
  python main.py --no-save-plots                   # Don't save plots
        """
    )
    
    parser.add_argument(
        "--data-path", 
        type=Path,
        help="Path to the smart grid dataset CSV file"
    )
    
    parser.add_argument(
        "--no-synthetic", 
        action="store_true",
        help="Skip synthetic data generation"
    )
    
    parser.add_argument(
        "--show-plots", 
        action="store_true",
        help="Display plots (default: False)"
    )
    
    parser.add_argument(
        "--no-save-plots", 
        action="store_true",
        help="Don't save plots to files (default: False)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize and run application
        app = SmartGridFaultDetectionApp(
            data_path=args.data_path,
            generate_synthetic=not args.no_synthetic,
            save_plots=not args.no_save_plots,
            show_plots=args.show_plots
        )
        
        # Run the pipeline
        results = app.run_pipeline()
        
        # Print classification report
        print("\nClassification Report:")
        print(results["classification_report"])
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
