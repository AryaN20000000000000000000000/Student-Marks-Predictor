# Student-Marks-Predictor
# Student Marks Predictor - Simple AI Project
# Author: Aryan Borude

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create a simple dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Marks_Scored': [35, 40, 45, 50, 55, 60, 65, 75, 85]
}
df = pd.DataFrame(data)

# Step 2: Separate features (X) and target (y)
X = df[['Hours_Studied']]
y = df['Marks_Scored']

# Step 3: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Display results
print("Predicted Marks:", predictions)
print("Actual Marks:", list(y_test))

# Step 7: Visualize the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Hours Studied vs Marks Scored')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.show()

# Step 8: Try your own input
hours = float(input("Enter study hours to predict marks: "))
predicted = model.predict([[hours]])
print(f"Predicted Marks for {hours} hours of study: {predicted[0]:.2f}")
