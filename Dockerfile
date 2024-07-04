FROM tensorflow/serving:latest
COPY plant /models/plant
ENV MODEL_NAME=plant
