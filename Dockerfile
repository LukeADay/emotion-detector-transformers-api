# Use the AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Set environment variables for Lambda compatibility
ENV LAMBDA_TASK_ROOT=/var/task

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy the Lambda handler code and model directory if needed
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
