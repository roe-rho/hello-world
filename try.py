import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int8)

# Use a smaller subset of data
X, _, y, _ = train_test_split(X, y, train_size=5000, stratify=y, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
log_reg.fit(X_train, y_train)

# Function to preprocess and predict a custom hand-drawn digit
def predict_custom_image(image_path, model):
    # Load and preprocess the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))                 # Resize to 28x28 pixels
    img_array = np.array(img)
    img_array = MinMaxScaler().fit_transform(img_array)  # Normalize pixel values

    # Flatten the image to match the model's input shape
    img_flat = img_array.flatten().reshape(1, -1)

    # Predict the digit
    prediction = model.predict(img_flat)[0]
    
    # Display the image and prediction
    plt.imshow(img_array, cmap="gray")
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis("off")
    plt.show()
    
    return prediction

# Test the function with your custom image path
# Replace 'path/to/your/hand-drawn-image.png' with the actual path to your image
custom_image_path = r'C:\Users\HP Envy X360\Desktop\eight.jpg'
predict_custom_image(custom_image_path, log_reg)
