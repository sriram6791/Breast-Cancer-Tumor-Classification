# Breast Cancer Prediction 🦸‍♀️👩‍🔬

Welcome to the **Breast Cancer Prediction** project! 🎉 This project is designed to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** using machine learning techniques. We'll be working with the **Breast Cancer Wisconsin (Diagnostic)** dataset, sourced from Kaggle. Our goal is to classify breast cancer tumors based on various features and build classification models that help us make accurate predictions.

## 🧠 About the Dataset

Breast cancer is the most common cancer among women worldwide, accounting for **25% of all cancer cases**. In 2015 alone, over **2.1 million individuals** were diagnosed with the disease. The cancer typically begins when cells in the breast grow uncontrollably, forming tumors that can be detected either through **X-rays** or by feeling lumps.

The main challenge lies in classifying these tumors as either **malignant (cancerous)** or **benign (non-cancerous)**. In this project, we leverage **machine learning** to help automate this classification process. We use a set of features such as radius, texture, smoothness, and symmetry of the tumor to make the prediction.

### 📝 Dataset Information

- **Source**: Kaggle
- **Data Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Task**: Classification (Malignant or Benign)
- **Total Instances**: 569
- **Features**: 30 numerical features including mean, radius, texture, area, smoothness, etc.

---

## 🏗️ Project Objectives

The goal of this project is simple: **Predict whether a breast tumor is malignant or benign**. Here's how we approached it:

1. **Understanding and cleaning the dataset**: We started by exploring the dataset to ensure that it was clean and ready for modeling.
2. **Feature Engineering**: We carefully selected the relevant features for our models.
3. **Model Building**: We implemented and evaluated several classification models, including:
   - **Support Vector Machines (SVM)** 🧑‍💻
   - **Logistic Regression** 📈
   - **Gradient Boosting** 🚀
   - **Deep Learning** 🤖
4. **Hyperparameter Tuning**: We fine-tuned the models to improve their performance.
5. **Model Comparison**: We compared the results from the different models to determine which one provides the best performance.

---

## 🚀 Models Used

Here are the four classification models we used:

1. **Support Vector Machine (SVM)**: SVM is a popular choice for classification tasks and is especially effective in high-dimensional spaces. It works by finding the best hyperplane that separates the classes.

2. **Logistic Regression**: A simpler model, but a very effective one for binary classification tasks. It predicts the probability of an instance belonging to a particular class using a logistic function.

3. **Gradient Boosting**: A powerful ensemble model that builds multiple weak learners and combines them to make a stronger, more accurate model. It is robust and performs well on a variety of datasets.

4. **Deep Learning**: Neural networks are a bit overkill for this task, but they can perform quite well on classification problems by learning intricate patterns from the data.

---

## 🔬 Results

### 1. Validation Accuracy

| Model                | Validation Accuracy |
|----------------------|---------------------|
| **SVM**              | 0.9649              |
| **Logistic Regression** | 0.9649            |
| **Gradient Boosting**  | 0.9561             |
| **Deep Learning**     | 0.9649              |

### Validation Classification Report

| Model              | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) | Accuracy |
|--------------------|---------------|------------|--------------|---------------|------------|--------------|----------|
| **SVM**            | 0.97          | 0.97       | 0.97         | 0.95          | 0.95       | 0.95         | 0.96     |
| **Logistic Regression** | 0.97       | 0.97       | 0.97         | 0.95          | 0.95       | 0.95         | 0.96     |
| **Gradient Boosting**  | 0.97       | 0.96       | 0.96         | 0.93          | 0.95       | 0.94         | 0.96     |
| **Deep Learning**    | 0.97       | 0.97       | 0.97         | 0.95          | 0.95       | 0.95         | 0.96     |

### 2. Test Accuracy

| Model                | Test Accuracy |
|----------------------|---------------|
| **SVM**              | 0.9737        |
| **Logistic Regression** | 0.9737      |
| **Gradient Boosting**  | 0.9474       |
| **Deep Learning**     | 0.9649        |

### Test Classification Report

| Model              | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) | Accuracy |
|--------------------|---------------|------------|--------------|---------------|------------|--------------|----------|
| **SVM**            | 0.97          | 0.99       | 0.98         | 0.98          | 0.95       | 0.96         | 0.97     |
| **Logistic Regression** | 0.97       | 0.99       | 0.98         | 0.98          | 0.95       | 0.96         | 0.97     |
| **Gradient Boosting**  | 0.96       | 0.96       | 0.96         | 0.93          | 0.93       | 0.93         | 0.95     |
| **Deep Learning**    | 0.99       | 0.96       | 0.97         | 0.93          | 0.98       | 0.95         | 0.96     |

---

## 📊 Summary and Comparison

### **SVM vs Logistic Regression**
Both **SVM** and **Logistic Regression** performed **similarly** on both validation and test sets. In fact, they both achieved the same validation accuracy of **0.9649** and test accuracy of **0.9737**. However, **SVM** slightly outperformed **Logistic Regression** on the test set, particularly in the recall for class 0 (malignant tumors).

### **Gradient Boosting**
Although **Gradient Boosting** had a slightly lower performance compared to **SVM** and **Logistic Regression**, it still produced **solid results** with a test accuracy of **0.9474**. Precision and recall values for class 1 were slightly lower (0.93 for both).

### **Deep Learning**
The **Deep Learning** model performed quite well, but it did not **outperform** **SVM** or **Logistic Regression** in terms of test accuracy. The **recall** for class 1 was very high (0.98), but the **precision** for class 0 was a bit lower compared to other models (0.99 vs. 0.97).

---

## 🎯 Conclusion

- **Best Overall Performance**: Both **SVM** and **Logistic Regression** performed almost identically, with **SVM** having a slight edge in terms of **test accuracy** and **recall** for class 0.
- **Deep Learning** showed competitive results, particularly in **recall** for class 1, but it slightly lagged behind in **test accuracy**.
- **Gradient Boosting** was the **least performant** among the models, but still managed a reasonable test accuracy of **0.95**.

In conclusion, **SVM** and **Logistic Regression** are the top contenders for this task, with **SVM** taking the slight edge. 🎯

---

## 🚀 How to Run This Project

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   cd breast-cancer-prediction
