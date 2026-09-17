# 🦠 COVID-19 Positivity Prediction Using Naive Bayes

## 📌 Project Overview

This project demonstrates a **Machine Learning classification workflow using the Gaussian Naive Bayes algorithm** to predict whether a person is classified as COVID-19 positive or negative based on a set of provided features.

The main purpose of this project is to practice:

* Data cleaning
* Handling missing values
* Categorical data encoding
* Train-test splitting
* Naive Bayes classification
* Model evaluation
* Confusion matrix visualization

> **Important:** This is an **educational project using a synthetic dataset**. It is not a medically validated COVID-19 diagnostic or susceptibility prediction system and should not be used for medical decisions.

---

## 🎯 Objective

The objective is to classify each record into one of two classes:

```text
0 → Negative
1 → Positive
```

The model uses information such as:

* Age
* Sex
* Fever
* Cough
* Fatigue
* Sore throat
* Breathing difficulty
* Body temperature
* Oxygen level
* Diabetes
* Hypertension
* Exposure history
* Vaccination status

---

## 🧠 Machine Learning Algorithm

### Gaussian Naive Bayes

Naive Bayes is a supervised classification algorithm based on **Bayes' theorem**.

The algorithm calculates the probability of each class based on the available features and selects the class with the higher probability.

For this project, **GaussianNB** is used because the dataset contains continuous numerical variables such as:

* Age
* Body temperature
* Oxygen level

The categorical variables are converted into numerical form before training.

---

## 📊 Dataset

The dataset used in this project is **synthetically generated for educational purposes** and contains intentional missing values so that data preprocessing can be demonstrated.

### Main columns

| Column                 | Description              |
| ---------------------- | ------------------------ |
| `Patient_ID`           | Unique record identifier |
| `Age`                  | Age of the person        |
| `Sex`                  | Sex                      |
| `Fever`                | Presence of fever        |
| `Cough`                | Presence of cough        |
| `Fatigue`              | Presence of fatigue      |
| `Sore_Throat`          | Presence of sore throat  |
| `Breathing_Difficulty` | Breathing difficulty     |
| `Body_Temperature_F`   | Body temperature         |
| `Oxygen_Level`         | Oxygen level             |
| `Diabetes`             | Diabetes status          |
| `Hypertension`         | Hypertension status      |
| `Exposure_History`     | Exposure history         |
| `Vaccinated`           | Vaccination status       |
| `Corona_Positive`      | Target variable          |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Load Data
   ↓
Check Missing Values
   ↓
Remove Patient ID
   ↓
Handle Missing Values
   ↓
Encode Categorical Variables
   ↓
Separate Features and Target
   ↓
Train-Test Split
   ↓
Gaussian Naive Bayes
   ↓
Make Predictions
   ↓
Evaluate Model
   ↓
Confusion Matrix
```

---

## 🧹 Data Preprocessing

### Handling Missing Values

Numerical missing values are handled using statistical values such as the median or mean.

Example:

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Body_Temperature_F"] = df["Body_Temperature_F"].fillna(
    df["Body_Temperature_F"].mean()
)

df["Oxygen_Level"] = df["Oxygen_Level"].fillna(
    df["Oxygen_Level"].mean()
)
```

Categorical missing values are replaced using the most frequent category:

```python
df["Cough"] = df["Cough"].fillna(
    df["Cough"].mode()[0]
)
```

The same approach is applied to the other categorical columns.

### Categorical Encoding

Categorical features are converted into numerical values using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=[
        "Sex",
        "Fever",
        "Cough",
        "Fatigue",
        "Sore_Throat",
        "Diabetes",
        "Hypertension",
        "Breathing_Difficulty",
        "Exposure_History",
        "Vaccinated"
    ],
    drop_first=True
)
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The split is:

* **80% Training**
* **20% Testing**

The model is trained only on the training data and evaluated on the unseen test data.

---

## 🤖 Model Training

The model is created using:

```python
model = GaussianNB()
```

It is then trained using:

```python
model.fit(
    X_train,
    y_train
)
```

Predictions are generated using:

```python
y_pred = model.predict(
    X_test
)
```

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Accuracy measures the proportion of total predictions that were correct.

### Precision

Precision measures how many records predicted as positive were actually positive.

### Recall

Recall measures how many of the actual positive records were correctly identified.

### F1 Score

F1 Score provides a balance between precision and recall.

### Confusion Matrix

The confusion matrix shows:

* True Negatives
* False Positives
* False Negatives
* True Positives

---

## ⚠️ Accuracy and Model Performance

The model does **not necessarily achieve very high accuracy**, and this is expected for this particular project.

The dataset is:

* Small
* Synthetic
* Artificially generated for learning
* Not medically validated
* Potentially imbalanced between positive and negative classes

Therefore, the resulting accuracy should **not be interpreted as the real-world performance of a COVID-19 prediction system**.

The project focuses on learning the machine-learning workflow rather than achieving clinically meaningful prediction performance.

For example, the actual output from the program may look like:

```text
Accuracy : <value from your run>
Precision: <value from your run>
Recall   : <value from your run>
F1 Score : <value from your run>
```

The values should be reported exactly as produced by the model rather than modified to make the project appear more accurate.

---

## 🔲 Confusion Matrix

A confusion matrix is generated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

and visualized using Seaborn.

This provides a better understanding of the types of errors made by the model rather than relying only on accuracy.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

---

## 📁 Project Structure

```text
coronavirus-naive-bayes/
│
├── coronavirus_susceptibility.csv
├── naive_bayes.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd coronavirus-naive-bayes
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python program

```bash
python naive_bayes.py
```

---

## 📚 Key Learning Outcomes

Through this project, I learned how to:

* Load a CSV dataset using Pandas
* Identify missing values
* Handle numerical and categorical null values
* Encode categorical features
* Separate features and target
* Split data into training and testing sets
* Implement Gaussian Naive Bayes
* Generate predictions
* Evaluate a classification model
* Interpret accuracy, precision, recall, and F1-score
* Use a confusion matrix to analyze predictions

---

## 🚧 Limitations

This project has several limitations:

1. The dataset is synthetic rather than a validated clinical dataset.
2. The dataset is relatively small.
3. The class distribution may be imbalanced.
4. The model has not been clinically validated.
5. The results should not be used for diagnosis or medical decision-making.

---

## 🚀 Future Improvements

Possible future improvements include:

* Using a larger, validated public dataset
* Comparing Naive Bayes with Logistic Regression, Decision Tree, Random Forest, KNN, and SVM
* Using cross-validation
* Handling class imbalance
* Hyperparameter tuning
* Deploying the educational model using Streamlit

---

## 👨‍💻 Project Purpose

This project was created as part of my **Machine Learning learning journey** to gain hands-on experience with data preprocessing, classification algorithms, and model evaluation.

#MachineLearning #Python #NaiveBayes #ScikitLearn #DataScience #MachineLearningProject #ArtificialIntelligence
