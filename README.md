# 🍱 Nutrition5K Calorie Estimation

This repository implements deep learning models for **food calorie estimation** from **RGB-D images** using the Nutrition5K dataset.
It supports multiple architectures, including Resnet (single RGB input), Late Fusion-based ResNet, RGB-D FusionNet and FusionCAB.

---

## 📁 Project Structure
```
Comp90086_Nutrition5k
├── Nutrition5K # Nutrition5k dataset
├── dataset_nutrition.py # Dataset class and preprocessing logic
├── depth_stats.json # Cached depth min/max statistics
├── main.py # Entry point for training/testing
├── model.py # Model definitions 
├── trainer.py # Training loop, evaluation, checkpointing
├── run_training_latefusion.sh # Script to train ResNet Late-Fusion model
├── run_training_fusenet.sh # Script to train RGBD FusionNet
├── run_training_fusecab.sh # Script to train RGBD FusionCAB
└── run_training_resnet.sh # Script to train Resnet (support rgb only input)
```

---

## 🚀 How to Run

To train a model, modify the argument values in the corresponding shell script (e.g., dataset path, batch size, epochs, learning rate), then run:

```bash
./run_training_*.sh

