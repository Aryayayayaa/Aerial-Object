# Project Aegis: Aerial Target Recognition (ATR)
UAV vs. Bird Classification using Transfer Learning. Compares MobileNetV2 (best performance &amp; smallest size for edge deployment), EfficientNetB0, and ResNet50 models. Achieved high Recall (approx 98%) for critical drone detection. Features Fine-Tuning, Keras Callbacks, and detailed metrics.

---

# 🦅 UAV vs. Bird Classification for Aerial Surveillance

This project implements a high-performance image classification system using Transfer Learning to distinguish between **Unmanned Aerial Vehicles (UAVs/Drones)** and **Birds** in aerial imagery. This distinction is critical for airspace security, surveillance, and minimizing false alarms in drone detection systems.

The core of the project involves comparing and optimizing three state-of-the-art Convolutional Neural Networks (CNNs) for a balanced performance across high-stakes security metrics (Recall) and computational efficiency (Model Size).

## 🚀 Key Models and Results

Three pre-trained models were evaluated using fine-tuning optimization strategies.

| Model | Optimization Strategy | Model Size (MB) | Best Test Recall | Best Test F1-Score | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | Fine-Tuning (Top 40 Layers) | **20 MB** | **~0.985** | ~0.970 | **FINAL CHOICE:** Best Recall and smallest size for edge/mobile deployment. |
| **EfficientNetB0**| Fine-Tuning (Top 60 Layers) | 29 MB | ~0.975 | **~0.980** | Highest overall accuracy/F1, but larger size. |
| **ResNet50** | Fine-Tuning (Top 50 Layers) | 98 MB | ~0.660 | ~0.740 | Poor Recall performance after optimization; not suitable for security. |

*Note: The final deployed model is **Optimized MobileNetV2** due to its critical high Recall score (low False Negatives) and exceptional efficiency.*

## 💡 Technical Implementation

### 1. Architecture
We utilized **Transfer Learning** by loading the ImageNet-pre-trained weights of the base model and attaching a custom classification head. The core steps involve:
1.  **Phase 1 (Feature Extractor):** Train only the new top layers.
2.  **Phase 2 (Fine-Tuning):** Unfreeze the top layers of the backbone (e.g., top 40 layers for MobileNetV2) and retrain with a very low learning rate ($\mathbf{1e-4}$) to adapt the features to the new domain.



[Image of Transfer Learning Architecture Diagram]


### 2. Hyperparameter Optimization
Instead of brute-force search, a targeted approach using Keras Callbacks was implemented:
* **Early Stopping:** Monitors `val_loss` and stops training when improvement plateaus, preventing overfitting.
* **Model Checkpoint:** Saves the model weights only when the `val_loss` reaches a new minimum, ensuring the final model retains the best generalization weights.

### 3. Evaluation Metrics (Business Focus)
* **Recall (Security Critical):** Prioritized as it measures the ability to detect *all* actual Drones (minimizing False Negatives/Missed Threats).
* **F1-Score:** Used to balance the high Recall with sufficient Precision (minimizing False Positives/False Alarms).

## 💻 Setup and Usage

### Prerequisites

* Python 3.8+
* TensorFlow / Keras
* Numpy, Pandas, Matplotlib, Seaborn

### Installation

```bash
git clone [https://github.com/YourUsername/uav-bird-classification.git](https://github.com/YourUsername/uav-bird-classification.git)
cd uav-bird-classification
pip install -r requirements.txt 
# Ensure your dataset is correctly organized into 'train', 'validation', and 'test' folders

---
