# Parkinson-s-Disease-Detection---Deep-learning

## Dataset
The dataset used for this project includes spiral and wave drawings. Due to its large size, the dataset is hosted on Google Drive.

[Download the Dataset from Google Drive](https://drive.google.com/drive/folders/1h3Th-fPh7p7G901MFGrCYuZ9PAp0vCGW?usp=drive_link)

# What is Parkinson's disease?
- Parkinson's disease is a progressive neurological disorder that affects movement.
- It primarily causes tremors, stiffness, and slowness of movement.
- The disease occurs due to the loss of dopamine-producing neurons in the brain.
- Symptoms often start gradually and worsen over time, affecting daily activities.
- Parkinson's can also lead to non-motor symptoms like cognitive impairment and mood changes.
- There is no cure, but treatments are available to manage symptoms and improve quality of life.

# About Project
In this project, we developed a convolutional neural network (CNN) model to detect Parkinson's disease using image data. Here's what our proposed system involves:

Data Collection: We utilized a dataset containing spiral and wave drawings made by individuals, with half of the images from Parkinson's patients (showing tremors) and the other half from healthy individuals.

Data Preprocessing: The images were preprocessed, including resizing, normalization, and augmentation, to improve the model's ability to learn from the data.

Model Development: A CNN was designed and trained to classify the images as either indicative of Parkinson's disease or normal, based on the patterns and irregularities in the drawings.

Training and Validation: The model was trained on 80% of the dataset and validated on the remaining 20%, achieving a validation accuracy of 76%.

Prediction Capability: The final model can take new images as input and predict whether the person has Parkinson's disease based on the drawing.

Application Development: We plan to integrate the trained model into an application that will allow users, particularly in hospitals, to upload drawing images and receive instant diagnostic predictions.







