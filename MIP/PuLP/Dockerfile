FROM pulp/pulp:latest

# Copy Python script and data folder
WORKDIR /app

# Install libraries
COPY requirements.txt ./
RUN pip3 install -r ./requirements.txt

# Creating inst, res and src folders
RUN mkdir -p ./inst
RUN mkdir -p ./res
RUN mkdir -p ./src

# Adding files
COPY src/* ./src
COPY inst/* ./inst
COPY res/* ./res

ENTRYPOINT ["python3", "./src/main.py"]