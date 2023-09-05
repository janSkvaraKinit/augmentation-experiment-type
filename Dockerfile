FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY augmentation_script.py ./augmentation_script.py
CMD ["python", "augmentation_script.py"]