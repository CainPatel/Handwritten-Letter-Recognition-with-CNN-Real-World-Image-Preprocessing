# streamlight script for the lowercase letter predicter

# Upload a photo of a single handwritten letter (A–Z) and the model will classify it.

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np

# 1. Model Setup
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*7*7,64)
        self.fc2 = nn.Linear(64,26)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("emnist_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
letters = "abcdefghijklmnopqrstuvwxyz"

# 2. UI Header
st.title("Handwritten Letter Classifier")
st.write("Upload a picture of a single lowercase handwritten letter (a-z).") 
st.info("For best results, the character should be written on plain white paper," \
        "and the picture should fill the frame while having minimal background objects or lines")

# 3. Uploader
uploaded = st.file_uploader("Choose image", type=["png", "jpg", "jpeg"])

# 4. Processing
if uploaded is not None:
    # converts image to CV format
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    #displays the uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption = "Your Uploaded Image", width = 250)

    # converts color to RGB to grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 

    # compares intensity to neighbors to distinguish shadows from features
    thresh = cv2.adaptiveThreshold( 
        img_gray, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=201, 
        C=15 
    )

    # finds all contours and keeps the one with the largest area. Halts script if no contour is detected.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if len(contours) == 0:
        st.error("No letter detected, try a cleaner image")
        st.stop()
    largest = max(contours, key=cv2.contourArea) 

    #crops image to the largest contour
    x, y, w, h = cv2.boundingRect(largest) 
    cropped = thresh[y:y+h, x:x+w] 

    # makes the image a square
    h, w = cropped.shape
    size = max(h,w) 

    # centers letter inside of the black square
    square = np.zeros((size,size), dtype=np.uint8)
    y_off = (size-h)//2
    x_off = (size-w)//2
    square[y_off:y_off + h, x_off: x_off+w] = cropped

    # adds margin 
    margin = int(size*0.2)
    square = cv2.copyMakeBorder(square, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)

    # resize down to 28 x 28
    resized = cv2.resize(square, (28,28), interpolation=cv2.INTER_AREA)

    # Transposes image
    oriented = resized.T 

    # Normalizes using EMNIST mean and stdev, then Tensorizes image with compatible shape 
    img_norm = oriented.astype("float32")/255.0
    img_norm = (img_norm - 0.1307) / 0.3081 
    input_tensor = torch.tensor(img_norm, dtype=torch.float32).reshape(1,1,28,28)

    # 5. Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Stores top 3 letters with their confidences
    top3 = torch.topk(probs, 3)
    top_letter = letters[top3.indices[0]]
    top_conf = top3.values[0].item()

    if top_conf < 0.3:
        st.warning(f"Not confident this is a clear letter. Best Guess: '{top_letter}', ({top_conf:.0%} Confidence). Try a cleaner image.")
    else:
        st.success(f"Predicted Letter: '{top_letter}' (confidence: {top_conf:.0%})")

    st.write("Top 3 Guesses:")
    for i, p in zip(top3.indices, top3.values):
        st.write(f" {letters[i]}, {p.item():.1%}")





