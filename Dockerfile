FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the actual API code
COPY . /code

# Hugging Face Spaces exposes port 7860 by default
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
