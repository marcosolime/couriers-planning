FROM minizinc/minizinc:latest

# Update the package list
RUN apt-get update

# Install Python
RUN apt-get install -y python3 python3-pip

# Copy Minizinc model, Python script and data folder
WORKDIR /app

# Install libraries
COPY requirements.txt ./
RUN pip3 install -r ./requirements.txt

# Creating inst, res and src folders
RUN mkdir -p ./src
RUN mkdir -p ./inst
RUN mkdir -p ./res

# Adding files
COPY src/* ./src
COPY inst/* ./inst
COPY res/* ./res

ENTRYPOINT ["python3", "./src/main.py"]

