# Vehicle Routing Problem

## Problem Description
We are given a fleet of couriers and a set of items that must be distributed to a set of clients. All the couriers start at the same location (depot), whereas the customers are located in different positions on a 2D flat region. Each courier comes with a fixed capacity, meaning that we can stuff items till a given threshold; each item has its relative weight.
The goal of the problem is to plan a route for each courier, reaching all the customers while not exceeding the courier capacities. To be fair, we want to minimize the longest distance travelled by any courier. This way, we expect to have a balanced division among the drivers in term of traveled distance.

## Implementation
To deal with this hard problem (actually, NP-Hard!), we propose three different approches exploiting different modeling strategies:

- CP (Contraint Programming)
- SMT (Satisfiability Modulo Theory)
- MIP (Mixed Integer Linear Programming)

You can find the code implementation, data files and results in the correponding directories. All the folders share the same structure.

- [CP/SMT/MIP]
	* [inst]
	* [res]
	* [src]
	* Dockerfile
	* requirements.txt

In the inst folder there are the problem instances; in res folder there are the results in json format; in src folder you can find the model implementations. 

To avoid any incompatibilies or library depencencies, we "containerized" the applications in a Docker environment. After running the docker image, you can find the results in json format in the 'res' folder inside Docker Desktop.

## CP - Constraint Programming
We modeled our problem using MiniZinc, and solved it with Gecode and Chuffed. Additionally, we exploited several search strategies to increase convergence speed.

### How to run the program
1. Build the image: `docker build -t <image-name> ./CP`
2. Run the image: `docker run <image-name> <solver-name> <instance-name>`

Eg: `docker run mini-img gecode inst01.dat`

Eg: `docker run mini-img chuffed inst01.dat`

## SMT - Satifiability Modulo Theory
We created the model using Z3Py Python library, and solved it with Z3.

### How to run the program
1. Build the image: `docker build -t <image-name> ./SMT`
2. Run the image: `docker run <image-name> <instance-name>`

Eg: `docker run z3-img inst01.dat`

Eg: `docker run z3-img inst01.dat`

## MIP - Mixed Integer Linear Programming
We explored two alternatives:
1. Gurobi API + Gurobi solver (with an Academic Licence)
2. PuLP + Cbc solver (no license needed)

### How to run the program
1. Build the image: `docker build -t <image-name> ./MIP/{Gurobi, PuLP}`
2. Run the image: `docker run <image-name> <instance-name>`

Eg: `docker run gurobi-img inst01.dat`

Eg: `docker run pulp-img inst01.dat`

Note: To run the Gurobi subpackage you need to provide your personal Gurobi license (we used an academic license). You can see how to obtain a license on the official Gurobi website. Once you have the license, under GurobiAPI folder create a folder named lic and copy your license there. Finally, build and run the docker image.

## TODOs

- CP
	1. Generate json for instances 0-21 with no simmetry constraints using Chuffed
	2. Generate json for instances 0-21 with no simmetry constraints using Gecode

- MIP
	* Gurobi API
		1. Implement constraint 9 (couriers with the same capacity do different routes)
		2. Generate json for instances 0-21 with simmetry constraints
		3. Generate json for instances 0-21 without simmetry constraints

	* PuLP
		1. Implement constraint 9 (couriers with the same capacity do different routes)
		2. Generate json for instances 0-21 with simmetry constraints
		3. Generate json for instances 0-21 without simmetry constraints

- SMT
	1. ~~Write the model in Z3Py~~
	2. Write symmetries
	3. Generate json for instances 0-21 with simmetry constraints
	4. Generate json for instances 0-21 without simmetry constraints

- Other stuff
	* Write the project repot (Overleaf)
	* Ideas on Warm Start
	* Checking/Improving objective bounds

