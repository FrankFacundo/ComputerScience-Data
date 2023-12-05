# Use Miniconda as the base image
FROM continuumio/miniconda3:4.9.2

# Create a Conda environment for PySpark
RUN conda create -n pyspark-env python=3.9

# Activate the Conda environment
SHELL ["conda", "run", "-n", "pyspark-env", "/bin/bash", "-c"]

# Install Java and PySpark using Conda
RUN conda install -c conda-forge openjdk pyspark

# Copy the PySpark script into the container
COPY simple_script.py /simple_script.py

# Set the entry point to run the PySpark script within the Conda environment
ENTRYPOINT ["conda", "run", "-n", "pyspark-env", "python", "/simple_script.py"]
