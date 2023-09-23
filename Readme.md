# Vehicle Routing Problem

## Problem Description
We are given a fleet of couriers and a set of items that must be distributed to the relative clients. All the couriers start at the same location (depot), whereas the customers are located in different positions on a 2D flat region. Each courier comes with a fixed capacity, meaning that we can stuff items till a given threshold; each item has its relative weight.
The goal of the problem is to plan a route for each courier, reaching all the customers while not exceeding the courier capacities. To be fair, we want to minimize the longest distance travelled by any courier. This way, we expect to have a balanced division among the drivers in term of traveled distance.

## Implementation
To deal with this hard problem (actually, NP-Hard!), we propose three different approches exploiting different modeling strategies:

- CP (Contraint Programming)
- SMT (Satisfiability Modulo Theory)
- MIP (Mixed Integer Linear Programming)

You can find the code implementation, data files and results in the correponding directories. All the folders share the same structure.

- [CP/SMT/MIP]
	* [inst]					% instances (.dat)
	* [res]						% results (.json)
	* [src]						% code (.py, .mzn)
	* Dockerfile
	* Readme.md
	* requirements.txt

To avoid any incompatibilies or library depencencies, we "containerized" the applications in a Docker environment. To build and run such containers, you can find the instructions in the specific subfolders.


