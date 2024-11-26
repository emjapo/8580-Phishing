# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file with the data
#file_path = "C:\\Users\\emjap\\Documents\\Clemson\\fall24\\CPSC8580\\8580-Phishing\\Results - Yahoo.csv" # Emily file path
file_path = "C:\\Users\\CEJac\\OneDrive\\Desktop\\SecureRepo\\8580-Phishing\\Results.csv" # Chris file path
yahoo_data = pd.read_csv(file_path)

# Drop the column that has "this one has code" in it
yahoo_data_cleaned = yahoo_data.drop(columns=["Unnamed: 4"])

# Remove the enron ones
yahoo_data_cleaned = yahoo_data_cleaned[yahoo_data_cleaned["LLM"] != "Enron"]

# Insert only the specified LLM
yahoo_data_cleaned = yahoo_data_cleaned[yahoo_data_cleaned["LLM"] == "Pi"]

# Encode "Detected?" column (categorical to numeric: "no" -> 0, "yes" -> 1)
yahoo_data_cleaned["Detected?"] = yahoo_data_cleaned["Detected?"].map(
    {"no": 0, "yes": 1}
)

# Drop rows with missing values from the entire DataFrame
yahoo_data_cleaned = yahoo_data_cleaned.dropna(subset=["Subject", "Email", "Detected?"])
yahoo_data = yahoo_data_cleaned

# Sample data (replace with your own data)
texts = yahoo_data["Subject"] + " " + yahoo_data["Email"]
labels = yahoo_data["Detected?"].astype(int)


# Convert text data into feature vectors
# Changed the vectorization into TF-IDF to give weight to each feature word
vectorizer = TfidfVectorizer(max_features=750, stop_words="english")
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# Prints the feature names with the most significant weight
print(feature_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Use a parameter grid to train specified criteria
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": [3, 8, 15],
    "min_samples_leaf": [1, 2, 5],
    "class_weight": [None, "balanced"],
}

# Create the DecisionTreeClassifier
# dt_classifier = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV using a K-Folds validation technique
cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    verbose=1,
    n_jobs=-1,
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validated Accuracy: {best_score * 100:.2f}%")

# Test on the held-out test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

# Visualize the decision tree using matplotlib
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=feature_names,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
)
plt.show()

# Print the decision tree rules for reference
print("Decision Tree Rules:")
print(export_text(best_model, feature_names=feature_names))

# Test the classifier's performance
accuracy = best_model.score(X_test, y_test)
print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")

feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)

print(importance_df.head(10))
