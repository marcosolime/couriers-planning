# Vehicle Routing Problem

## Problem Description
We are given a fleet of couriers and a set of items that must be distributed to a set of clients. All the couriers start at the same location (depot), whereas the customers are located in different positions on a 2D flat region. Each courier comes with a fixed capacity, meaning that we can stuff items till a given threshold; each item has its relative weight.
The goal of the problem is to plan a route for each courier, reaching all the customers while not exceeding the courier capacities. To be fair, we want to minimize the longest distance travelled by any courier. This way, we expect to have a balanced division among the drivers in term of traveled distance.

## Implementation
To deal with this hard problem (actually, NP-Hard!), we propose three different approches exploiting different modeling strategies:

- CP (Contraint Programming)
- SMT (Satisfiability Modulo Theory)
- MIP (Mixed Integer Linear Programming)

For each variant we implemented the lightweigth and enhanced version (with symmetry breaking constraints). You can find the code implementation, data files and results in the correponding directories. All the folders share the same structure.

- [CP/SMT/MIP]
	* [inst]
	* [res]
	* [src]
	* Dockerfile
	* requirements.txt

'inst' folder contains the problem instances (.dat); 'res' folder is a directory for local solutions; 'src' folder contains the code.

To avoid any incompatibilies or library depencencies, we "containerized" the applications in a Docker environment. After running the docker image, you can find the results in json format in the 'res' folder inside Docker Desktop.

## CP - Constraint Programming
We modeled our problem using MiniZinc, and solved it with Gecode and Chuffed. Additionally, we exploited several search strategies to increase convergence speed.

### How to run the program
1. Build the image: `docker build -t <image-name> ./CP`
2. Run the image: `docker run -v .\res\CP:/app/res cp <image-name> <gecode/chuffed> <instance-name> <sym_on/sym_off>`

You can find the new dumped solutions on the res folder on your local volume '.\res\CP'. Be aware that new solutions will be added to the existing files.

Examples:
- `docker run -v .\res\CP:/app/res cp gecode inst01.dat sym_on`
- `docker run -v .\res\CP:/app/res cp chuffed inst05.dat sym_off`

Note:
Search annotations are only compatible with Gecode solver. To use Chuffed you simply comment out the search annotations and run `solve minimize obj;`.

## SMT - Satisfiability Modulo Theory
We created the model using Z3Py Python library, and solved it with Z3.

### How to run the program
1. Build the image: `docker build -t <image-name> ./SMT`
2. Run the image: `docker run -v .\res\SMT:/app/res <image-name> <instance-name> <sym_on/sym_off>`

Examples:
- `docker run -v .\res\SMT:/app/res smt inst01.dat sym_on`
- `docker run -v .\res\SMT:/app/res smt inst05.dat sym_off`

## MIP - Mixed Integer Linear Programming
We explored two alternatives:
1. Gurobi API + Gurobi solver (with an Academic Licence)
2. PuLP + Cbc solver (no license needed)

### How to run the program
1. Build the image: `docker build -t <image-name> ./MIP/{Gurobi, PuLP}`
2. Run the image: `docker run -v .\res\MIP:/app/res <image-name> <instance-name> <sym_on/sym_off>`

Example:
- `docker run -v .\res\MIP:/app/res pulp inst01.dat sym_on`
- `docker run -v .\res\MIP:/app/res guru inst05.dat sym_off`

Note: To run the Gurobi subpackage you need to provide your personal Gurobi license (we used an academic license). You can see how to obtain a license on the official Gurobi website. Once you have the license, under GurobiAPI folder create a folder named lic and copy your license there. Finally, build and run the docker image.

## RouteTracer
This is a python script that let you visualize the results with graphs and plots.
To test the program, navigate under 'RouteTracer' folder and run `python src/main.py inst01.dat cp/1.json` (example instance).
More info in the instruction file in 'RouteTracer' folder.
