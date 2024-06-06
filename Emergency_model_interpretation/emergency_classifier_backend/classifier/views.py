from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

def predict_emergency(request):
    if request.method == 'POST' and request.FILES['image']:
        # Load the EmergencyNet model from the classifier directory
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_emergencyNet.h5')
        model = load_model(model_path)

        # Process the uploaded image and make predictions using the EmergencyNet model
        image = request.FILES['image']
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (240, 240))
        img = img / 255.0  # Normalize

        # Reshape the image to match model input shape (batch_size, height, width, channels)
        img = np.expand_dims(img, axis=0)

        # Make predictions
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        classes = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
        result = {'class': classes[class_index], 'confidence': float(prediction[0][class_index])}

        return JsonResponse(result)
    else:
        return render(request, 'upload_image.html')
