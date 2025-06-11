#!/usr/bin/env python3
"""
Integration Demo: LSTM + Decision Tree Pipeline

This script demonstrates how to use the integrated LSTM and Decision Tree analysis.
It shows three different workflows:

1. Standard Decision Tree analysis on raw data
2. Decision Tree analysis on LSTM-generated features  
3. Complete integrated pipeline (LSTM prediction → CT analysis)

Usage:
    python integration_demo.py
"""

import os
import sys

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_standard_ct_analysis():
    """Demo 1: Standard CT analysis on raw data"""
    print("🌳 Demo 1: Standard Decision Tree Analysis")
    print("="*50)
    
    import CT
    
    # This will run the standard CT analysis
    # You can modify CT.py main() function to set use_lstm_predictions = False
    print("Running standard CT analysis...")
    print("To run this demo, set use_lstm_predictions = False in CT.py main() function")
    print()

def demo_lstm_prediction_generation():
    """Demo 2: Generate LSTM predictions for CT analysis"""
    print("🧠 Demo 2: Generate LSTM Predictions")
    print("="*40)
    
    try:
        import smart
        
        # Parameters
        model_path = "models/LSTM/lstm_model.pth"
        dataset_path = "../../data/data_Q1_2025"
        output_dir = "demo_ct_analysis"
        
        # Check if LSTM model exists
        joblib_path = model_path.replace('.pth', '.joblib')
        if not (os.path.exists(model_path) or os.path.exists(joblib_path)):
            print(f"❌ LSTM model not found at {model_path}")
            print("   Please train the LSTM model first by running smart.py")
            return False
        
        # Generate LSTM predictions for CT analysis
        print("Generating LSTM predictions for CT analysis...")
        predictions_df, ct_features_df = smart.export_for_ct_analysis(
            model_path=model_path,
            dataset_path=dataset_path,
            output_dir=output_dir
        )
        
        if predictions_df is not None and ct_features_df is not None:
            print(f"✅ Successfully generated {len(ct_features_df)} drive-level features")
            print(f"📁 Files saved to: {output_dir}/")
            return True
        else:
            print("❌ Failed to generate LSTM predictions")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import smart module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error generating LSTM predictions: {e}")
        return False

def demo_ct_analysis_on_lstm():
    """Demo 3: CT analysis on LSTM predictions"""
    print("🔬 Demo 3: Decision Tree Analysis on LSTM Predictions")
    print("="*55)
    
    try:
        import CT
        
        # Parameters
        lstm_features_path = "demo_ct_analysis/ct_features.csv"
        dataset_path = "../../data/data_Q1_2025"
        
        # Check if LSTM features exist
        if not os.path.exists(lstm_features_path):
            print(f"❌ LSTM features not found at {lstm_features_path}")
            print("   Please run demo_lstm_prediction_generation() first")
            return False
        
        # Run CT analysis on LSTM predictions
        print("Analyzing LSTM predictions with Decision Trees...")
        results = CT.analyze_lstm_predictions_with_ct(
            lstm_features_path=lstm_features_path,
            ground_truth_path=dataset_path,
            feature_selection_method='all',
            output_dir="demo_ct_lstm_analysis"
        )
        
        if results:
            print(f"✅ CT analysis completed!")
            print(f"🎯 Gini accuracy: {results['gini_accuracy']:.4f}")
            print(f"🎯 Entropy accuracy: {results['entropy_accuracy']:.4f}")
            print(f"📊 Analyzed {results['test_samples']} test samples")
            return True
        else:
            print("❌ CT analysis failed")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import CT module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in CT analysis: {e}")
        return False

def demo_integrated_pipeline():
    """Demo 4: Complete integrated pipeline"""
    print("🚀 Demo 4: Complete Integrated LSTM + CT Pipeline")
    print("="*52)
    
    try:
        import CT
        
        # Parameters
        smart_model_path = "models/LSTM/lstm_model.pth"
        dataset_path = "../../data/data_Q1_2025"
        
        # Check if LSTM model exists
        joblib_path = smart_model_path.replace('.pth', '.joblib')
        if not (os.path.exists(smart_model_path) or os.path.exists(joblib_path)):
            print(f"❌ LSTM model not found at {smart_model_path}")
            print("   Please train the LSTM model first by running smart.py")
            return False
        
        # Run integrated pipeline
        print("Running complete integrated analysis pipeline...")
        results = CT.run_integrated_lstm_ct_analysis(
            smart_model_path=smart_model_path,
            dataset_path=dataset_path,
            output_dir="demo_integrated_analysis"
        )
        
        if results:
            print(f"✅ Integrated pipeline completed!")
            print(f"📊 Processed {results['lstm_predictions_count']} LSTM predictions")
            print(f"🎯 Best CT accuracy: {max(results['ct_analysis_results']['gini_accuracy'], results['ct_analysis_results']['entropy_accuracy']):.4f}")
            return True
        else:
            print("❌ Integrated pipeline failed")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in integrated pipeline: {e}")
        return False

def main():
    """Main demo function"""
    print("🎯 LSTM + Decision Tree Integration Demo")
    print("="*60)
    print()
    
    print("This demo shows how to integrate LSTM predictions with Decision Tree analysis.")
    print("Choose a demo to run:")
    print()
    print("1. Generate LSTM predictions for CT analysis")
    print("2. Run CT analysis on existing LSTM predictions")
    print("3. Run complete integrated pipeline")
    print("4. Run all demos in sequence")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                print()
                demo_lstm_prediction_generation()
                print()
            elif choice == '2':
                print()
                demo_ct_analysis_on_lstm()
                print()
            elif choice == '3':
                print()
                demo_integrated_pipeline()
                print()
            elif choice == '4':
                print()
                print("🔄 Running all demos in sequence...")
                print()
                
                # Run demos in logical order
                success1 = demo_lstm_prediction_generation()
                if success1:
                    success2 = demo_ct_analysis_on_lstm()
                    if success2:
                        demo_integrated_pipeline()
                print()
            else:
                print("❌ Invalid choice. Please enter 0-4.")
                continue
                
            print("\n" + "-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()
