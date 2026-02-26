#!/usr/bin/env python3
"""
Model Training Script
Trains the skincare recommendation model and saves it to disk
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.recommendation import model
from config.settings import active_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main training function"""
    try:
        print("\n" + "="*70)
        print("🎯 SKINCARE RECOMMENDATION MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load dataset
        print("\n📂 Step 0/5: Loading dataset...")
        dataset_path = active_config.DATASET_PATH
        print(f"   Dataset path: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"❌ ERROR: Dataset not found at {dataset_path}")
            return 1
        
        model.load_dataset(dataset_path)
        print(f"✅ Dataset loaded: {len(model.data)} products")
        print(f"   Columns: {list(model.data.columns)}")
        print(f"   Missing values handled: YES")
        
        # Step 2: Train model
        print("\n🚀 Starting training process...")
        metrics = model.train()
        
        # Step 3: Save model
        print("\n💾 Saving model...")
        model_path = active_config.MODEL_PATH
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        
        # Summary
        print("\n" + "="*70)
        print("🎉 SUCCESS - MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"\n📊 Final Statistics:")
        print(f"   • Model Location: {model_path}")
        print(f"   • Model File Size: {os.path.getsize(model_path) / 1024:.2f} KB")
        print(f"   • Label Encoders: {len(model.label_encoders)}")
        print(f"   • Features: {model.feature_columns}")
        print(f"\n🎯 Performance Metrics:")
        print(f"   • Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   • Precision: {metrics['precision']:.4f}")
        print(f"   • Recall:    {metrics['recall']:.4f}")
        print(f"   • F1-Score:  {metrics['f1']:.4f}")
        print(f"\n✅ Model ready for API use!")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED")
        print(f"Error: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
