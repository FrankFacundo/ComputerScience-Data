# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container at /app
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the pip cache, making the image smaller.
# --compile: Compiles Python source files to .pyc files. (Optional, can speed up startup slightly)
RUN pip install --no-cache-dir --compile -r requirements.txt

# 5. Copy the rest of the application code into the container at /app
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run your app using Uvicorn
#    main:app -- tells Uvicorn to look for an object named `app` in a file named `main.py`
#    --host 0.0.0.0 -- makes the server accessible from outside the container
#    --port 8000 -- the port Uvicorn will listen on
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
