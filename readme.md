# Mitigating_Unlearnable_Datasets_CIFAR10

## Introduction
In this project, we aim to train a robust deep learning model on an "unlearnable" (poisoned) dataset. Unlearnable datasets contain specialized, imperceptible noise designed to deceive neural networks and prevent them from generalizing.  
In this challenging computer vision task, we built a robust training pipeline and experimented with integrating pre-trained models as backbones to bypass this protection and extract clean features.  
Furthermore, we implemented specialized pipelines for handling poisoned image tensors, designed robust data sanitization and augmentation mechanisms (such as Mixup) to eliminate the malicious noise, and transformed raw inputs into model-ready formats.  

## Dataset Description
The project utilizes a modified **CIFAR-10** dataset.  
The data is provided as NumPy arrays (`.npy` files) containing the pixel values and corresponding labels.  
The dataset encompasses 10 common categories (such as airplane, automobile, bird, cat, etc.). Crucially, the training set (`x_train_cifar10_unlearn.npy`) has been injected with unlearnable noise, while the validation and test sets remain clean.  

## Feature Engineering & Preprocessing
We design a data processing and augmentation pipeline to enhance model generalization and sanitize the poisoned data.

- Image Sanitization:  
    Using the `tf.data` API, we implemented multiple mapping functions to destroy the malicious noise while preserving semantic structure. The best-performing method (`gray_comp1`) applies color thresholding (truncating pixel values below 79 for training data) followed by heavy JPEG compression (quality=20) and decompression to filter out high-frequency poison signals.
- Online Augmentation (Mixup):  
    Utilizing `tensorflow_probability`, we built a data loader that dynamically blends pairs of training images and their one-hot labels using a Beta distribution. This technique acts as a strong regularizer and further dilutes the effect of the unlearnable noise during training.
- Input Formatting & Augmentation:  
    Input images are dynamically augmented with random horizontal flips and rotations. We then apply `Rescaling` to convert pixel values from `[0, 1]` to `[-1, 1]`, which is the format required by the ResNet50V2 backbone.

## Model Architectures
To overcome the unlearnable noise and evaluate the impact of robust feature extractors, the project implements the following architecture:

- ResNet50V2 Backbone:   
    Integrates a pre-trained ResNet50V2 backbone (trained on ImageNet) to replace standard convolutional stages, leveraging transfer learning for robust feature extraction and faster convergence.
- Classification Head:   
    Utilizes a `GlobalAveragePooling2D` layer followed by a standard `Dense` layer outputting a 10-dimensional tensor (corresponding to the 10 CIFAR-10 classes).
- Optimization Strategy:
    Employs the Adam optimizer with a `CosineDecay` learning rate schedule and `EarlyStopping` to prevent overfitting.

## Coding and Required Environment
- Environment  
    The project uses Docker (`docker-compose.yml` & `Dockerfile`) to establish an isolated development environment containing TensorFlow 2.18.0 and a Jupyter Lab Server.  
    Hardware requires a CUDA-enabled GPU (e.g., NVIDIA GeForce RTX 4070) to accelerate deep learning computations and model training.

- Dependencies
  - Deep Learning: `tensorflow`, `tf-keras`, `tensorflow-probability`
  - Data Processing: `pandas`, `numpy`, `scikit-learn`
  - Visualization & Environment: `matplotlib`, `jupyterlab`, `ipywidgets`, `tqdm`

- Execution
  - Environment Setup:  
  Use `docker-compose up -d --build` to build and launch the container. Access the Jupyter environment via port 8888.
  - Preprocessing:  
  Loads the `.npy` datasets and sets up the `tf.data.Dataset` pipelines, mapping the JPEG compression sanitization functions (`gray_comp1_train` and `gray_comp1_val`) to the data.
  - Training:  
  Compiles the ResNet50V2 architecture and initiates the training process. The model applies online Mixup and stops training automatically when validation accuracy plateaus using the `EarlyStopping` callback.
  - Prediction:  
  Loads the test dataset (`x_test_cifar10.npy`), applies a specialized testing sanitization function (`gray_comp_test` optimized with `@tf.function`), and evaluates the final accuracy of the model using `scikit-learn`.

# References
- [Unlearnable Examples: Making Personal Data Unexploitable](https://arxiv.org/abs/2101.04898)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Data API (tf.data) Documentation](https://www.tensorflow.org/guide/data)