import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Import Dataset
data = pd.read_csv(r"C:\Users\nag15\.spyder-py3\Spyder Projects\Artificial Intelligence\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
nltk.download('stopwords')
corpus = []
for review in data['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(review))

# Creating the Bag of Words Model and TF-IDF Model
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus).toarray()

y = data.iloc[:, 1].values

# Splitting the dataset into training and test sets
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=0)

# Initialize models with valid keys
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'GaussianNB': GaussianNB()
}

# Initialize an empty list to store results of each model
all_results = []

# Function to evaluate a model and return its metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    bias = model.score(X_train, y_train)
    var = model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
    
    # Create a dictionary for model evaluation metrics
    result = {
        'Model': model.__class__.__name__,
        'Accuracy': acc,
        'Bias (Training Score)': bias,
        'Variance (Test Score)': var,
        'AUC': auc_score
    }
    
    return result

# Evaluate all models and store results in the all_results list
for model_name, model_instance in models.items():
    result = evaluate_model(model_instance, X_train_tfidf, X_test_tfidf, y_train, y_test)
    all_results.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(all_results)

# Streamlit App Title
st.title("Interactive Model Selection")

# Display the preview of all algorithms' results
st.subheader("Preview of All Models' Results")
st.dataframe(results_df)

# Model selection dropdown in Streamlit
model_choice = st.selectbox('Select a model for detailed evaluation', list(models.keys()))

# Select the model based on user's choice
selected_model = models[model_choice]

# Evaluate the selected model (re-evaluate if needed for demonstration purposes)
results_df_selected = evaluate_model(selected_model, X_train_tfidf, X_test_tfidf, y_train, y_test)

# Display individual model evaluation results
st.subheader(f"Evaluation for {model_choice}")
st.write(f"Accuracy: {results_df_selected['Accuracy']:.2f}")
st.write(f"Bias (Training Score): {results_df_selected['Bias (Training Score)']:.2f}")
st.write(f"Variance (Test Score): {results_df_selected['Variance (Test Score)']:.2f}")
st.write(f"AUC: {results_df_selected['AUC']:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
y_pred = selected_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
st.pyplot(fig)

# ROC Curve
if hasattr(selected_model, 'predict_proba'):
    y_pred_proba = selected_model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_choice}')
    st.pyplot(fig)
