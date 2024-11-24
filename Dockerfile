# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the app code into the container
COPY ./app /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8000 for Flask
EXPOSE 8000

# Start the Flask app
CMD ["python", "main.py"]
