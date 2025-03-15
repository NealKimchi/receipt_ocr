# Receipt Text Detection

A deep learning solution for detecting text regions in receipts and invoices using an attention-enhanced U-Net architecture. This project focuses on the bounding box detection component of a receipt OCR pipeline, with plans for expansion into full text recognition and extraction.

## Project Overview

This project implements a text detection model for receipts and invoices, capable of identifying and localizing text regions in images. The model uses a U-Net architecture enhanced with attention mechanisms to improve detection accuracy, particularly for challenging receipt layouts.

The current implementation concentrates on accurate text region detection and bounding box prediction, which serves as the foundation for a complete OCR system. The model also provides confidence estimation for detected regions, allowing for quality filtering in downstream applications. Future development will extend the system to include text recognition within the detected regions and structured data extraction for specific receipt fields like items, prices, and dates.

## Notebooks

In the notebooks subdirectory there are two notebooks, data_parser and evaluation_results. If you run both you will see the results, data_parser showing the dataloading and the preprocessing of the images. In evaluation_results you will 
see a sample I was able to download from Talapas of a few of the validation images with the bounding boxes drawn on them from my latest model. While the precision is 0.6983 and the f1 score is 0.4863 for the box metrics I do think it is significantly
better than that. Looking at the images, the text map is dead on, the only difference is it doesn't seem to separate lines, I believe this can be solved with some hyperparameter tuning like max box height or less aggressive merging.

## Data Source

The model is trained using the Invoices and Receipts OCR Dataset from Hugging Face (mychen76/invoices-and-receipts_ocr_v2). This dataset provides a rich collection of receipt and invoice scans accompanied by comprehensive annotations.

Each sample in the dataset contains an image of a receipt or invoice, a unique identifier, and both parsed and raw data. The raw data field includes a JSON structure with three key components: recognized text (ocr_words), bounding box coordinates for each text element (ocr_boxes), and field classifications (ocr_labels). This rich annotation allows the model to learn both the visual appearance and spatial layout of text in receipts.

## Project Structure

The project is organized into several Python modules, each handling specific aspects of the text detection pipeline:

```
├── configs/                
    ├── detection_config.yaml
├── models/
    ├── eval_detection.py       # Script to evaluate model
    ├── loss.py                 # loss functions 
    ├── metrics.py              # metric equations
    ├── model.py                # model architecture
├── notebooks/
    ├── data_parser.ipynb       # testing data loader
├── output/
    ├── data_parser.ipynb       # testing data loader
├── train/
    ├── eval_20250314_173949    # Contains evaluation of latest model
    ├── run_20250314_123521     # Contains latest model
├── utils/
    ├── ocr_utils.py            # OCR-specific utilities
    ├── text_utils.py           # Text processing utilities
    ├── data_loading.py         # DataLoader class
├── requirements.txt            # requirements
├── .gitignore                  # .gitignore
```

The modular design allows for easy extension and modification of individual components as the project evolves toward a complete OCR solution.

## Model Architecture

The text detection model builds upon a U-Net architecture, incorporating several enhancements specifically designed for the challenges of receipt text detection. The architecture enables capturing multi-scale features for robust detection of text at various sizes and orientations, with attention mechanisms that improve feature selection at skip connections to focus on relevant text regions.

The model produces multiple outputs through specialized branches: a pixel-level text segmentation map, a confidence estimation map for reliability assessment, and precise bounding box coordinates for text localization.

### Building Blocks

The network is constructed from several fundamental building blocks. The ConvBlock serves as the basic unit, consisting of a 2D convolution with a 3×3 kernel (stride 1, padding 1), followed by batch normalization for training stability and a ReLU activation function. This basic block appears throughout the network to process and refine features.

For downsampling in the encoder path, the DownBlock module combines MaxPooling (2×2) for spatial dimension reduction with two consecutive ConvBlocks that double the channel depth. This progressive reduction in spatial dimensions with increasing feature depth allows the network to capture increasingly abstract features.

The upsampling process in the decoder path is handled by the UpBlock module, which uses transposed convolution (2×2, stride 2) to expand spatial dimensions. A key aspect here is the concatenation of features from the corresponding encoder level via skip connections, followed by two ConvBlocks for feature refinement. These skip connections are critical for preserving fine spatial details that might otherwise be lost during downsampling.

### Attention Mechanism

A crucial enhancement to the standard U-Net architecture is the incorporation of attention gates. The AttentionBlock implements a soft attention mechanism that allows the model to focus on relevant features in the skip connections. 

The attention mechanism works by processing both a gating signal from deeper layers and feature maps from the encoder. Each undergoes a separate convolutional projection followed by batch normalization. These projections are then added together, passed through a ReLU activation, and further processed by a convolution and sigmoid activation to produce attention coefficients. These coefficients effectively weight the feature maps, highlighting important regions while suppressing irrelevant ones before they're passed to the decoder.

This attention mechanism significantly improves the model's ability to focus on text regions and ignore background elements, which is particularly important for receipts with complex layouts and varying background patterns.

### Network Architecture

The encoder path begins with initial dual ConvBlocks that process the input image (3 channels) into 64-channel feature maps. It then proceeds through four DownBlocks that progressively increase feature channels (64→128→256→512→1024) while halving spatial dimensions at each step. This creates a hierarchical representation of features at multiple scales.

The attention gates are positioned at each skip connection, with four modules carefully tuned to the channel depths at each level. The gating signals come from the decoder path, while the feature maps come from the corresponding encoder level. This allows higher-level semantic information to guide the selection of relevant lower-level features.

The decoder path consists of four UpBlocks that progressively decrease feature channels (1024→512→256→128→64) while doubling spatial dimensions. Each upsampling step incorporates attention-enhanced features from the encoder through skip connections, allowing the network to recover spatial details while maintaining focus on relevant text regions.

### Output Branches

The model produces three complementary outputs through specialized branches:

The text map head consists of a single convolutional layer (64→1) with sigmoid activation, producing a pixel-wise probability map of text presence. This map effectively segments the image into text and non-text regions.

The confidence head applies a reduced channel convolution followed by sigmoid activation (ConvBlock 64→32, then Conv 32→1), estimating the reliability of each predicted text region. This confidence score is valuable for downstream filtering and quality assessment.

The box regression head is a multi-layer branch for precise coordinate prediction, consisting of two ConvBlocks (64→64→32) followed by a final convolution (32→4) with sigmoid activation. This branch outputs normalized [x1, y1, x2, y2] coordinates for each detected text region.

This multi-task learning approach enables the model to simultaneously segment text regions, estimate confidence, and predict bounding boxes, leveraging shared feature representations across these related tasks.

## Data Preprocessing and Augmentation

Receipt images present unique challenges for computer vision models due to varying formats, lighting conditions, print quality, and physical handling (folding, crumpling, etc.). To address these challenges, the project implements a sophisticated augmentation pipeline through the ReceiptAugmentation class in data_loading.py.

The augmentation strategy includes carefully selected geometric transformations that preserve text readability while simulating realistic document variations. Safe rotations are limited to ±15 degrees to avoid excessive text distortion. Optical and grid distortions simulate the warping often seen in scanned or photographed receipts, while elastic transformations replicate the effect of paper deformations.

Appearance variations are also crucial for model robustness. The pipeline includes brightness and contrast adjustments to handle different lighting conditions, gamma corrections to account for scanner variations, and CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance low-contrast regions that are common in thermal-printed receipts.

To simulate realistic capture artifacts, the augmentation includes several noise types: Gaussian noise for general sensor noise, multiplicative noise for illumination variations, and ISO noise to replicate camera sensor effects. Motion and Gaussian blur are added to simulate out-of-focus scenarios that frequently occur when receipts are captured with mobile devices.

All images undergo standard ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] to facilitate transfer learning and stable training. The augmentation pipeline is applied with a probability of 0.7 during training, balancing between augmented and original samples for optimal learning.

## Training Process

The training pipeline incorporates several advanced techniques to ensure robust model performance. The loss function combines three components, each addressing different aspects of the text detection task: Dice loss for the text segmentation map, Focal loss for confidence estimation, and Generalized IoU (GIoU) loss for bounding box regression.

Dice loss is particularly effective for handling the class imbalance inherent in text detection, where text regions typically occupy a small portion of the receipt. Focal loss further addresses this imbalance by focusing on hard examples during confidence prediction. GIoU loss provides better gradients for bounding box regression compared to traditional L1/L2 losses, especially for non-overlapping boxes.

The training process employs learning rate scheduling with cosine annealing and warmup to facilitate convergence. The warmup phase gradually increases the learning rate during initial epochs, allowing the model to establish stable gradient directions before accelerating learning. The subsequent cosine annealing smoothly reduces the learning rate to fine-tune the model parameters.

Hard example mining is implemented to focus computational resources on difficult samples. This technique selectively backpropagates gradients from samples with higher losses, effectively increasing the influence of challenging examples during training.

The pipeline includes periodic validation to monitor model performance on unseen data, preventing overfitting. Visualization of predictions at regular intervals provides qualitative insights into the model's learning progress. Comprehensive metric tracking and model checkpointing ensure that the best models are saved based on both validation loss and F1 score.

## Evaluation Metrics

The evaluation of the text detection model employs multiple complementary metrics to assess different aspects of performance. At the segmentation level, the model is evaluated using precision, recall, F1 score, and Intersection over Union (IoU).

Precision measures the proportion of correctly identified text pixels among all pixels predicted as text, providing insight into the model's false positive rate. Recall quantifies the proportion of actual text pixels that were correctly identified, indicating the model's false negative rate. The F1 score provides a balanced measure combining precision and recall, while IoU measures the overlap between predicted and ground truth text regions.

For bounding box evaluation, the metrics include box-level precision, recall, and F1 score. These metrics consider a predicted box correct if its IoU with a ground truth box exceeds a threshold (typically 0.5). This provides a more practical assessment of the model's ability to localize text regions.

Additionally, GIoU is used to evaluate box regression quality, offering a more nuanced measure that accounts for the relative positions and sizes of boxes, even when they don't overlap. This comprehensive evaluation approach ensures that all aspects of the text detection task are properly assessed.

## Usage

### Training

To train the model with default parameters:

```bash
python train_detection.py 
```

The training script will create a timestamped directory within the specified output directory to store model checkpoints, training curves, and visualization samples.

### Evaluation

To evaluate a trained model:

```bash
python eval_detection.py --model_path outputs/detection/run_20250314_123521/best_model.pth
```

The evaluation script will generate comprehensive metrics and save example visualizations to assess model performance.

## Configuration

The model behavior can be customized through the detection_config.yaml file, which contains parameters for the model architecture, data processing, loss function weighting, and training strategy. Key parameters include:

```yaml
model:
  in_channels: 3 
  out_channels: 1

data:
  dataset_name: "mychen76/invoices-and-receipts_ocr_v2"  
  image_size: [512, 512]

loss:
  text_map_weight: 1.0 
  box_weight: 0.1  
  confidence_weight: 0.5 

training:
  batch_size: 8  
  epochs: 5
  learning_rate: 0.0001
```

These parameters can be adjusted to adapt the model to different requirements, balancing between detection accuracy, inference speed, and resource utilization.

## Development Notes

This project was initially conceived as a full receipt OCR tool but currently focuses on the text detection component. This strategic decision allows for thorough optimization of the detection stage before tackling the more complex recognition tasks. The current architecture provides strong foundations for localization, which is a critical first step in the OCR pipeline.

The separation of detection and recognition modules follows established OCR approaches, allowing each component to be specialized for its specific task. Future development will introduce a text recognition module that processes the detected regions to extract the actual text content, followed by structured information extraction to categorize text into meaningful receipt fields like products, prices, and totals.

## Requirements

The project relies on several key Python libraries:

- PyTorch for deep learning model implementation
- albumentations for image augmentation
- Hugging Face datasets for data loading
- OpenCV for image processing
- NumPy for numerical operations
- matplotlib for visualization
- tqdm for progress tracking

## Appendix

[Results]output/run_20250314_123521/train_vis_epoch2_batch249.png

## Acknowledgements

The project acknowledges the valuable contribution of the Invoices and Receipts OCR Dataset provided by mychen76/invoices-and-receipts_ocr_v2 on Hugging Face, which has been instrumental in training and evaluating the text detection model.