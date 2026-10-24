# MDC-ConvNeXt: Soybean Leaf Disease Recognition System
Official PyTorch Implementation | Paper in Submission Stage

Title: 《MDC-ConvNeXt: A Lightweight Deep Learning Framework for Soybean Leaf Disease Recognition》

Propose an improved ConvNeXt V2 model (MDC-ConvNeXt) that integrates Mixed Local Channel Attention (MLCA) and Multi-Scale Deep Convolution (MDConv) to achieve high-precision classification of multiple types of soybean leaf diseases (Grey spot, Healthy, Macular, Mosaic). It effectively suppresses complex background noise and multi-scale lesion conflicts, providing an efficient, deployable solution for precise identification and early warning of soybean foliar diseases, and supporting precision agriculture.

## 1. Research background and model positioning
Soybean is among the most traded agricultural commodities worldwide, yet foliar diseases limit productivity. Traditional visual diagnosis is inefficient and subjective. To address this, this project constructs a lightweight deep learning model utilizing an optimized ConvNeXt V2 architecture, combined with the MLCA (Mixed Local Channel Attention) module and the MDConv (Multi-Scale Deep Convolution) module. It achieves automatic multi-class recognition of common soybean leaf diseases, balancing lightweight design, real-time performance, and high representational capacity for edge deployment.

## 2. Core innovation points

**1. Dual-module collaborative attention and multi-scale feature fusion:**
*   **MLCA (Mixed Local Channel Attention):** Embedded into each block, utilizing local and global adaptive pooling and 1D convolution to dynamically capture multi-scale symptom modeling, enhancing key lesion features and suppressing background interference.
*   **MDConv (Multi-Scale Deep Convolution):** Replaces standard depthwise convolution with parallel multi-scale grouped convolutions (kernel sizes 3, 5, 7, 9) to extract lesion features under different receptive fields, improving adaptability to multi-scale diseases.

**2. Optimized lightweight backbone extraction architecture:**
Based on the improved ConvNeXt V2, the depthwise convolution kernel is expanded from 7×7 to 9×9, GRN layers are removed to reduce edge inference latency, and stage depths are redistributed to [3, 3, 27, 2] to strengthen the middle stage and reduce redundant high-level stages.

**3. DPFM (Dual-Branch Parallel Fusion Module) & Efficiency:**
Designed an asymmetric parallel branch structure with learnable fusion weights to achieve intra-layer fusion of attention and multi-scale features. The model has 15.27M parameters and 2.54G FLOPs. While achieving 99.34% overall recognition accuracy, the inference time is only 35.90ms per image (batch size 16), demonstrating strong potential for edge deployment.

## 3. Experimental dataset

**3.1 Dataset description**
The self-built soybean dataset was collected from Henan Province, China, comprising 7,516 images covering 4 categories: Grey spot, Healthy, Macular (Brown spot), and Mosaic. The dataset is divided into training, validation, and testing sets at a 3:1:1 ratio. Additionally, the public PlantVillage wheat dataset is used to validate cross-crop generalization.

**3.2 Dataset structure**
The dataset folder should be organized as follows. (You can download the dataset from [Insert your Baidu Netdisk / Google Drive link here]):
```text
dataset/
├── train/
│   ├── Grey_spot/
│   ├── Healthy/
│   ├── Macular/
│   └── Mosaic/
├── val/
│   ├── Grey_spot/
│   ├── Healthy/
│   ├── Macular/
│   └── Mosaic/
└── test/
    ├── Grey_spot/
    ├── Healthy/
    ├── Macular/
    └── Mosaic/
```
4. Experimental environment configuration
4.1 Dependency Installation
Recommend using Anaconda to create virtual environments and ensure that the dependent versions match.

Create and activate a virtual environment:

bash
conda create -n mdc-convnext python=3.11
conda activate mdc-convnext
Install PyTorch and TorchVision (compatible with CUDA 12.8, CPU users can replace it with CPU version):

bash
pip install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/cu128
Install other dependency libraries:

bash
pip install numpy matplotlib opencv-python pandas pillow tqdm timm scikit-learn seaborn
4.2 Hardware Requirements

OS: Windows 11 64-bit / Linux

GPU: Recommended NVIDIA GPU (e.g., RTX 5060 Laptop GPU or higher).

Python: 3.11.15

PyTorch: >= 2.0.0

5. Experimental results
5.1 Comparison of Core Indicators
The performance comparison between MDC-ConvNeXt and other mainstream deep learning models on the soybean leaf disease dataset is as follows:

Model	Params (M)	FLOPs (G)	Inference Time (ms)	Accuracy (%)
Classical ConvNeXt V2	14.99	2.44	29.00	98.34
Improved ConvNeXt V2	15.09	2.49	25.00	98.40
Improved ConvNeXt V2 + MLCA	15.09	2.49	28.00	99.00
Improved ConvNeXt V2 + MDConv	15.27	2.54	33.00	98.54
MDC-ConvNeXt (Ours)	15.27	2.54	35.90	99.34
Note: The model achieves 99.34% accuracy with a lightweight design, balancing accuracy and efficiency for real-time edge screening.

6. Code usage instructions
6.1 Model Training
Run the train.py script to start training, supporting configuration adjustment through parameters:

bash
python train.py --data_dir ./dataset --epochs 100 --batch_size 16 --lr 0.001
Main configuration items (can be manually modified in the script):

data_dir: Path to your dataset.

img_size: 224

batch_size: 16 (Default, adjust according to VRAM)

epochs: 100

lr: 0.001

Automatically complete after running:

Load training/validation/test sets.

Train the model and output loss, accuracy for each round.

Automatically save the best model (based on validation set accuracy) to ./weights/best_model.pth.

Output the final test set report.

6.2 Testing

bash
python test.py --weights ./weights/best_model.pth --data_dir ./dataset/test
7. Project file structure
text
MDC-ConvNeXt/
├── dataset/                  # Put your dataset here
├── weights/                  # Saved model weights (.pth)
├── ConvNext_MLCA_MDConv.py   # Core model architecture (MLCA, MDConv, DPFM)
├── train.py                  # Training script
├── test.py                   # Evaluation script
├── requirements.txt          # Dependencies
└── README.md
8. Known issues and precautions
Pre-trained Weights: Due to GitHub's file size restrictions, the pre-trained weight file (convnextv2_nano.pt or .pth) is not included directly in this repository. You can download it from the Releases section of this repository.

VRAM usage: During training, if batch_size=16 leads to OOM (Out of Memory), reduce it to 8 or 4, and appropriately lower the learning rate.

Data Augmentation: The model uses weak data augmentation and low regularization strategies to preserve critical lesion features. Modifying the augmentation strategy may impact final accuracy.

Environment mismatch: If you encounter CUDA incompatibility when installing PyTorch, you can adjust the PyTorch version according to your local CUDA version or switch to CPU.

9. References and contact information
9.1 Reference Method
The paper is currently in the submission stage and will be updated to BiBTeX format after its official publication. Currently available for temporary reference:

bibtex
@article{wang2026mdc,
  title={MDC-ConvNeXt: A Lightweight Deep Learning Framework for Soybean Leaf Disease Recognition},
  author={Wang, Jiayi and Zhao, Hao and Xu, Laixiang and ...},
  journal={[Journal Name]},
  year={2026},
  note={Manuscript submitted for publication}
}
9.2 Contact Information
If you encounter code running issues or academic exchange needs, please contact:

Email: wangjiayi@huuc.edu.cn

GitHub Issue: Submit an issue directly to this repository and we will respond within 1-3 business days.
