

---

# Crop Care: Disease Classification and Treatment Suggestion System

This project provides a scalable solution for classifying crop diseases using a TensorFlow model, served via TensorFlow Serving. A FastAPI-based web service allows users to upload images of crops (starting with potato leaves) and receive disease predictions, along with treatment or care suggestions generated using Google Generative AI.

## Project Overview

### Key Features:
- **Disease Classification**: Identifies crop diseases from images (starting with potatoes) and provides future extensibility to include other crops.
- **Treatment Suggestions**: Offers care recommendations using Google Generative AI based on the diagnosed disease.
- **Dockerized Deployment**: Both the TensorFlow model and the FastAPI web service are containerized with Docker for easy deployment.



## Components

### 1. Model Training
The model was trained on the *PlantVillage* dataset and saved for deployment via TensorFlow Serving. The training process is documented in `training.ipynb`. Currently, the model focuses on potato leaf diseases but is designed to be extendable to other crops in the future.

### 2. TensorFlow Serving API

A Dockerized TensorFlow Serving setup hosts the trained model, allowing it to handle inference requests. The relevant Dockerfile is:

```Dockerfile
FROM tensorflow/serving:latest
COPY plant /models/plant
ENV MODEL_NAME=plant
```

This configures TensorFlow Serving to expose an API for the crop disease classification model.

### 3. Web Service

A FastAPI application serves as the interface for users to upload images, receive disease predictions, and obtain care suggestions. The web service is integrated with Google Generative AI for generating treatment advice based on the model's predictions.

Key functionalities:
- **POST `/predict`**: Upload a crop image to receive a disease classification and care recommendation.
- **GET `/ping`**: Health check for the web service.

#### Web Service Dockerfile

```Dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

This Dockerfile sets up a Python 3.9 environment and runs the FastAPI service.

### 4. Google Generative AI Integration

Once the disease is classified, the model sends a prompt to Google Generative AI's **Gemini 1.5** model, which generates care suggestions based on the diagnosed condition.

## Requirements

- **Docker**: To run the TensorFlow Serving and FastAPI containers.
- **Google Generative AI API key**: Required for generating care suggestions.
- **Python 3.9**: If running the FastAPI service locally.

## Getting Started

### 1. Running TensorFlow Serving

To serve the model via Docker:

```bash
docker build -t crop_model -f Dockerfile .
docker run -p 8501:8501 crop_model
```

This will expose the model API at `http://localhost:8501/v1/models/plant:predict`.

### 2. Running the Web Service

To start the FastAPI web service:

```bash
# Build the web service Docker image
docker build -t crop_web_service ./web_service

# Run the service on port 8001
docker run -p 8001:8001 crop_web_service
```

This starts the FastAPI app on `http://localhost:8001`.

### 3. Using the API

- **Health Check**: 
    ```bash
    curl http://localhost:8001/ping
    ```
- **Upload Image for Prediction**:
    ```bash
    curl -X POST "http://localhost:8001/predict" -F "file=@path_to_your_image.jpg"
    ```

The response includes:
- **Class**: The predicted disease (e.g., Early Blight, Late Blight, Healthy for potatoes).
- **Confidence**: Confidence score for the prediction.
- **Suggestion**: Care or treatment recommendation provided by Google Generative AI.





