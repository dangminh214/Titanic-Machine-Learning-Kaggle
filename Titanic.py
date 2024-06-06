# linear algebra
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read data
train_data = pd.read_csv("kaggle/input/titanic/train.csv")
train_data.head()
print("train data: \n", train_data)

test_data = pd.read_csv("kaggle/input/titanic/test.csv")
test_data.head()
print("test data: \n", test_data)

# % women survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of survived women:", rate_women)

# % men survived
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of survived men:", rate_men)

# Use Random Forrest Classification to classify
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

# Visualize the survival rates based on gender
survival_rates = [rate_women, rate_men]
labels = ['Women', 'Men']

# Create the bar chart
fig, ax = plt.subplots()
ax.bar(labels, survival_rates, color=['blue', 'green'])

# Add labels and title
ax.set_ylabel('Survival Rate')
ax.set_title('Survival Rate by Gender')
ax.set_ylim(0, 1)

# Display the plot
plt.show()
