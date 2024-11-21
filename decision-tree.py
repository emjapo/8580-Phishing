# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data (replace with your own data)
texts = [
    "The project was completed on time and met all expectations",
    "There were delays and unmet deadlines",
    "The work was exceptional and highly appreciated",
    "Several critical issues occurred during development",
    "Outstanding performance and delivery",
    "Repeated failures and lack of support",
    "Excellent team coordination and results",
    "Customer complaints about errors and performance",
]
labels = ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]  # Target labels (yes/no)

# Convert text data into feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()
print(feature_names)
# Convert labels to binary format
y = [1 if label == "yes" else 0 for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the decision tree classifier
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
classifier.fit(X_train, y_train)

# Visualize the decision tree using matplotlib
plt.figure(figsize=(20, 10))
plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
)
plt.show()

# Print the decision tree rules for reference
print("Decision Tree Rules:")
print(export_text(classifier, feature_names=feature_names))

# Test the classifier's performance
accuracy = classifier.score(X_test, y_test)
print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")
