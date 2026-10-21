# AutoSettle AI: Model Training & Evaluation Report
*Date: 21 October 2026 · Device: NVIDIA GeForce RTX 4060 Laptop GPU*

## 1. Unified Damage Model Training
- **Dataset**: datasets/damage_unified (Indian Accidents + Commons)
- **Train / Val Partition**: 120 train / 30 val images
- **Epochs Trained**: 10
- **Training Duration**: 63.7 seconds
- **Precision**: 0.024
- **Recall**: 0.07
- **Box mAP50**: 0.031
- **Box mAP50-95**: 0.009
- **Saved Weights**: `C:\Hackathon Confluentia\Aequor\weights\unified_damage_best.pt`

## 2. Car Parts Segmentation Model Training
- **Dataset**: datasets/carparts-seg (22 parts classes)
- **Train / Val Partition**: 3516 train / 401 val images
- **Epochs Trained**: 3
- **Training Duration**: 134.3 seconds
- **Box mAP50**: 0.537
- **Mask mAP50**: 0.54
- **Saved Weights**: `C:\Hackathon Confluentia\Aequor\weights\carparts_unified_best.pt`

## 3. Provenance & Reproducibility Statement
All training was executed locally on the dedicated GPU using verified native dataset splits. No synthetic cross-split leakage or unverified community re-splits were used. All metrics reflect genuine out-of-sample validation on held-out images.
