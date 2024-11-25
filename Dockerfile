# Use the AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy the Lambda handler code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
