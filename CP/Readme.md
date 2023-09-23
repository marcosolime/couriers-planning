# Instructions

## How to lunch the program

You can follow these steps:

1. Download Docker Desktop on your pc
2. Open Docker Desktop (the engine must be running)
3. Open your terminal and navigate into the current directory
4. Build the Docker image with: `docker build -t <image-name> .`
5. Launch the image with: `docker run <image-name> <solver-name> <instance-name>`

Example 1: `docker run mini-img gecode inst01.dat`
Example 2: `docker run mini-img chuffed inst01.dat`

## How to check the results

In the /res folder you can find the solutions in the json format. If you want to manually replicate the solution, you need to:

1. Run the Docker image
2. Navigate in the File section in the Docker container
3. Open the /res folder and fetch the solution

