# Model Card for MNIST

<!-- Provide a quick summary of what the model is/does. -->

This is a Convolutional Neural Network (CNN) model to classify grayscale images from the MNIST dataset.


## Table of Contents

- [Model Details](#model-details)
  - [Model Description](#model-description)
  - [Model Sources](#model-sources)
- [Uses](#uses)
  - [Primary Intended Use](#primary-intended-use)
  - [Primary Intended Users](#primary-intended-users)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Bias](#bias)
  - [Risks](#risks)
  - [Recommendations](#recommendations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Training Hyperparameters](#training-hyperparameters)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
    - [Summary](#summary)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Adrià Aumatell, Pol Arevalo, Ignasi Cervero, Zhengyong Ji, Rubén Villanueva
- **Model date** 18-09-2024
- **Model type:** Machine Learning Type, Deep Learning
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Model Sources 

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/mlops-2425q1-mds-upc/MLOps-mdsist

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Primary Intended Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The primary intended use of this model is to classify images of handwritten digits from the MNIST dataset into one of ten categories (0-9). It was specifically designed for image classification tasks without requiring additional fine-tuning or integration into larger applications. This model is ideal for educational, research, and benchmarking purposes within the field of machine learning, particularly in the area of digit recognition.


### Primary Intended Users 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model was developed for academic purposes, aimed at researchers, educators, and students. It is commonly used as a benchmark model in machine learning studies to explore classification techniques and compare algorithms. Its intended users are those in the educational and research communities, particularly for demonstrating and learning about image classification.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is designed specifically for classifying black-and-white images of handwritten digits from the MNIST dataset. It may not perform well on tasks outside of this narrow focus, such as classifying colored images, complex visual objects, or non-digit symbols. For applications requiring the recognition of letters, symbols, or colored images, consider using a model specifically designed for those contexts. Additionally, the model is not intended for fine-grained image classification tasks involving high-resolution or real-world photographs.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Bias

This model was trained only on the MNIST dataset, which includes grayscale images of handwritten digits. As a result, it may struggle with diverse handwriting styles, cultural variations, and digits written in formats not represented in the dataset. Its performance could vary across different demographic groups due to the lack of global handwriting diversity in the training data.

### Risks

This model is limited to recognizing handwritten digits from the MNIST dataset and may not generalize well to tasks involving letters, symbols, or noisy, colored images. Using it outside its intended scope, such as for real-world document analysis, can lead to inaccurate results. Over-reliance on this model in critical applications may result in misclassification with serious consequences.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

It is recommended to use this model only for tasks involving handwritten digit classification in black-and-white images. For tasks requiring recognition of more complex visual data or diverse writing styles, consider using a more advanced or specialized model.

## How to Get Started with the Model

Use the code below to get started with the model.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). For more details about the dataset, please refer to the [MNIST Dataset Card](link-to-dataset-card).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The preprocessing involves creating train, validation, and test splits to ensure proper model evaluation and performance assessment. Data is divided into these sets to train the model, validate it during training, and test it afterward on unseen data.


#### Training Hyperparameters

The key hyperparameters used during training include:

- **Optimizer**: Adam optimizer was used to train the model.
- **Weight decay**: Set to 0, meaning no regularization was applied to the weights.
- **Learning rate**: A learning rate of 1e-3 was used for model updates.
- **Seed**: A seed of 42 was set to ensure reproducibility.
- **Number of layers**: The model consists of 2 layers.
- **Activation function**: ReLU was used as the activation function at each layer.
- **Pooling**: Max pooling was applied during training.
- **Epochs**: The model was trained for 5 epochs.
- **Batch size**: A batch size of 64 was used.
- **Loss function**: Cross Entropy Loss was employed for model optimization.


#### Speeds, Sizes, Times

- **Training time**: 204 seconds
- **Batch size**: 64
- **Parameters**: 206922
- **Parameter size**: 0.79 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model evaluation was conducted using the MNIST dataset, which includes a separate test set of 10,000 grayscale images of handwritten digits. For more information about the testing data and its characteristics, please refer to the [MNIST Dataset Card](https://github.com/mlops-2425q1-mds-upc/MLOps-mdsist/blob/main/mnist_datacard.md).

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation of the MNIST image classification model can be disaggregated by the following factors:

1. **Digit Classes:** The evaluation can be broken down by each digit (0-9) to assess how well the model performs on each specific class. This will help identify any digits that the model struggles with compared to others.

2. **Handwriting Styles:** Although the MNIST dataset is uniform, potential variations in handwriting style can be assessed by simulating different writing styles or analyzing the model’s performance on subsets that reflect diverse handwriting characteristics.

3. **Noise Levels:** Testing can be conducted on images with varying levels of noise (e.g., low, medium, and high) to understand how noise impacts the model’s performance. This will help determine the model’s robustness under less-than-ideal conditions.

4. **Image Quality:** Evaluate the model on images that differ in quality, such as blurred or low-resolution images, to examine how these factors affect classification accuracy.

5. **Test Set Distribution:** Although the MNIST dataset is well-balanced, any deviations from the standard distribution in the test set can be analyzed to see if certain distributions affect performance.

By disaggregating the evaluation across these factors, we can gain deeper insights into the model's performance and identify areas for improvement, ensuring it meets various application needs effectively.

#### Metrics

The performance of the CNN model was evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified digits out of the total digits. It is the primary metric used to evaluate the overall performance of the model on the MNIST dataset. Given that MNIST is a balanced dataset with 10 classes (digits 0-9), accuracy provides a clear and reliable measure of model effectiveness.
  
- **Precision**: The ratio of true positive classifications (correctly predicted digits) to the sum of true positives and false positives. It indicates the model’s ability to avoid false positives.
  
- **Recall**: The ratio of true positive classifications to the sum of true positives and false negatives. It reflects the model's ability to find all the positive instances (digits).
  
- **F1 Score**: The harmonic mean of precision and recall. It balances the two metrics, providing a single measure of performance, particularly useful when precision and recall are uneven.
  
- **Confusion Matrix**: Provides detailed insight into the model’s performance by showing the number of true positive, true negative, false positive, and false negative predictions for each class (digits 0-9). Particularly useful to recognize weak aspects of the model, such as confusing the digits 3 and 8.

### Results

---
Model-index:
  - Name: CNN
    
    - Metadata:
      - FLOPS: 1.29 MMac
      - Parameters: 206922
      - Training data: MNIST
      - Training resources: 1 x NVIDIA GeForce GTX 1650 with Max-Q Design
    
    - Results:
      - Task:
        - Type: Image classification
      - Dataset:
        - Name: MNIST
        - Size: 52.3 MB
      - Metrics:
          - Name: Accuracy
          -  Value: 98.18
    
    - Source:
      - Name: CNN MNIST model
      - url: https://dagshub.com/Zhengyong8119/MLOps-mdsist.mlflow/#/experiments/2/runs/4578160aaefd45d7af36adcc65a1019f/artifacts 
---

#### Summary

Our CNN model, designed for image classification on the MNIST dataset, achieved an impressive accuracy of **98.18%**. The model was trained efficiently, requiring only **1.29 MMac FLOPS** and using **206,922 parameters**. Training was completed on a single **NVIDIA GeForce GTX 1650 with Max-Q Design**, demonstrating that even with modest computational resources, high accuracy can be achieved on standard tasks like digit classification.

For further details and to access the trained model, visit [CNN MNIST Model](https://dagshub.com/Zhengyong8119/MLOps-mdsist.mlflow/#/experiments/2/runs/4578160aaefd45d7af36adcc65a1019f/artifacts).

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

In our project, we utilized CodeCarbon to measure the environmental impact of our model in terms of CO₂ emissions. It's important to note that the environmental impact calculation is based solely on the final model training, rather than including all the training iterations and experiments conducted throughout the entire project development cycle.

- **Hardware Type:** GPU instance (1 x NVIDIA GeForce GTX 1650 with Max-Q Design)
- **Hours used:** 0.0567 hours
- **Country:** Spain
- **Region:** Catalonia
- **Carbon Emitted:** 0.3182 grams of CO2e
- **Power consumption** 0.0018 kWh

## Technical Specifications

### Model Architecture and Objective

The model is built using a Convolutional Neural Network (CNN) architecture specifically designed for classifying images within the MNIST dataset. CNNs are effective for image classification tasks due to their ability to automatically detect and learn spatial hierarchies of features through convolutional layers. This architecture enables the model to accurately recognize and categorize handwritten digits from the dataset, achieving high classification performance. The precision of float32 was used for the model's computations.

#### Hardware

- **GPUs**: 1 x NVIDIA GeForce GTX 1650 GPU
- **Memory**: 1 GB RAM
- **Storage**: 10 GB SSD for datasets and model storage
- **Training time**: 204 seconds

#### Software

- **Operating System**: Ubuntu 22.04 
- **Programming Languages**: Python 3.11
- **Frameworks/Libraries**: PyTorch and Torchvision
- **Version Control**: Git and DVC
- **Other Tools**: Poetry

## Citation

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

If you use this project in your research or applications, please consider citing it as follows:

```bibtex
@misc{mlops2024,
  title = {MLOps: CNN for Classifying MNIST Dataset},
  author = {Aumatell, Adrià and Arevalo, Pol and Cervero, Ignasi and Ji, Zhengyong and Villanueva, Rubén},
  year = {2024},
  howpublished = {GitHub repository},
  note = {Available at: https://github.com/mlops-2425q1-mds-upc/MLOps-mdsist}, 
}
```

## Model Card Authors

Adrià Aumatell, Pol Arevalo, Ignasi Cervero, Zhengyong Ji, Rubén Villanueva

## Model Card Contact

ignasi.cervero@estudiantat.upc.edu
