from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Actual values
y_actual = [1,1,1,1,1,0,0,0,0,0]

# Model predictions
y_predicted = [1,0,1,0,0,0,1,0,0,0]

# Create confusion matrix
cm = confusion_matrix(y_actual, y_predicted)

print("Confusion Matrix:")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Buy", "Buy"]
)

display.plot()
plt.show()