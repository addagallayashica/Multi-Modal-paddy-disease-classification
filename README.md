# Paddy Disease Classification Using Deep Learning

## About the Project

Our project focuses on classifying paddy diseases using deep learning. We use paddy leaf images to classify different diseases and healthy leaves into 13 categories.

We are implementing and comparing four models: Custom CNN, MobileNetV2, EfficientNet-B0, and Swin Transformer-Tiny. We are also exploring whether adding rice variety and plant age information along with images can improve classification performance.

## Objectives

- Classify paddy diseases using deep learning.
- Implement and compare four different deep learning models.
- Use rice variety and plant age as additional inputs.
- Compare image-only classification with image and metadata-based classification.
- Evaluate the models using different classification metrics.

## Models Used

### 1. Custom CNN

We use a CNN architecture to extract features from paddy leaf images and classify them into different disease categories. This serves as our baseline model.

### 2. MobileNetV2

MobileNetV2 is a lightweight deep learning model. We use it to evaluate paddy disease classification with lower computational requirements.

### 3. EfficientNet-B0

EfficientNet-B0 balances model complexity and computational efficiency. We use it to compare its classification performance with the other models.

### 4. Swin Transformer-Tiny

Swin Transformer-Tiny uses a transformer-based architecture to extract visual features from paddy images. We also extend it to incorporate rice variety and plant age information.

## Input Configurations

We are working with two configurations.

### Configuration A: Image Only

The model takes a paddy leaf image as input and predicts one of the 13 classes.

### Configuration B: Image + Crop Metadata

Along with the paddy image, we use rice variety and plant age as additional inputs. The model combines image features with the metadata before making a prediction.

## Dataset

We use the Paddy Doctor dataset, which contains paddy images and associated metadata.

The metadata includes:
- Image ID
- Disease label
- Rice variety
- Plant age

The classification task includes 13 classes, including healthy paddy plants.

## Evaluation Metrics

We evaluate and compare model performance using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion Matrix

## Current Results

We have evaluated Swin Transformer-Tiny using both configurations.

| Metric | Config A: Image Only | Config B: Image + Metadata |
|---|---:|---:|
| Accuracy | 98.36% | 97.38% |
| Balanced Accuracy | 98.54% | 97.37% |
| Macro F1-score | 98.40% | 97.38% |

In our current experiments, the image-only configuration performed better than the configuration that includes crop metadata. We will compare these results with the other models after their evaluations are completed.

## Technologies Used

- Python
- PyTorch
- Google Colab
- Custom CNN
- MobileNetV2
- EfficientNet-B0
- Swin Transformer-Tiny
- Pandas
- Scikit-learn

## Project Structure

```text
paddy-disease-classification/
├── custom_cnn/
├── mobilenetv2/
├── efficientnet_b0/
├── swin_transformer/
├── results/
├── README.md
└── requirements.txt
```

## Future Work

- Complete the experiments for all four models.
- Compare the performance of both input configurations.
- Analyze classification errors and per-class performance.
- Explore ways to improve classification accuracy.

## Conclusion

This project compares four deep learning models for paddy disease classification using image-only and image-plus-metadata inputs. The aim is to identify how different model architectures and the inclusion of crop metadata affect classification performance.
