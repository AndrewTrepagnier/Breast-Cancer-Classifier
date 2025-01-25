# Breast-Cancer-Classifier
A model that classifies breast cancer datasets as either Malignant or Benign with 90%+ accuracy

Most experienced clinicians and doctors can diagnose cancerous masses using fine needle aspirate (FNA) breast biopsy reports with high accuracy. Published statistical analyses of physician diagnoses have demonstrated benchmarks within 75–95% accuracy. My goal with this project was to train and validate a machine learning model to accurately predict whether a detected mass in a real-world breast biopsy report is malignant or benign. Furthermore, the machine learning model should employ probabilistic reasoning to explain why it believes the mass is malignant or benign.

The results are output as a text file. The following report were the actual result of the sklearn breast cancer dataset in a Random Forest Classifier:
=================================================================================================================

BREAST CANCER DATASET - SIMPLE REPORT
Generated on: 2025-01-25 11:28:03
--------------------------------------------------

PATIENT COUNTS:

Total Patients: 569

Malignant (Cancer): 212

Benign (Not Cancer): 357

KEY MEASUREMENTS:

mean radius: 14.13

mean texture: 19.29

mean perimeter: 91.97

mean area: 654.89

mean smoothness: 0.10

MODEL PERFORMANCE:
--------------------

Random Forest:
Accuracy: 97.08%
Error Rate: 2.92%

SVM:
Accuracy: 93.57%
Error Rate: 6.43%

Neural Network:
Accuracy: 95.91%
Error Rate: 4.09%

KNN:
Accuracy: 95.91%
Error Rate: 4.09%

BEST CLASSIFIER:
Random Forest (Accuracy: 97.08%)







==================================================================================


### My To Do List:

Rewrite the script to use only the Random Forest Classifier, as this has proven to be the most successful classifier.

Clean the dataset for improved performance and accuracy.

Implement SHAP for game-theoretic analysis of the model.

Incorporate a larger, more robust dataset. This will require data cleaning scripts to make it more applicable to real clinical settings.

Create a front-end interface for visualizing the model's accuracies and results.
