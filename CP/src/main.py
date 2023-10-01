'''
MCVRP (Multiple Capacity Vechicle Routing Problem)
'''

# Libraries
from minizinc import Instance, Model, Solver
import numpy as np
import datetime
import time
import sys
import json

# Print the solution
def has_solution(result, elapsed_time):
    if result.status.has_solution():
        print(f'Solution found in {round(elapsed_time)} seconds')
        return True
    else:
        print(f"No solution found before the timeout ({round(elapsed_time)}).")
        return False

def print_solution(result, m, n, elapsed_time, str_solver, str_data):
    # We get the optimal solution (last in the list)
    best_sol = result.solution[-1]
    tour = best_sol.next
    courier = best_sol.courier
    traveled = best_sol.traveled
    int_load = best_sol.int_load
    
    is_optimal = 'true'
    if elapsed_time > 300:
        is_optimal = 'false'
    
    obj = best_sol.objective
    sol = []

    # For each courier, we collect the traversed nodes
    for i in range(m):
        tmp_nodes = []
        next_node = tour[n+i]
        # Idle courier: no clients visited
        if next_node > n+m:
            sol.append([])
            continue
        tmp_nodes.append(next_node)
        while True:
            next_node = tour[next_node-1]
            if next_node > n:
                break
            tmp_nodes.append(next_node)
        sol.append(tmp_nodes)
    
    # We pack the stats in a dictionary...
    to_json = {}
    to_json[str_solver] = {}
    to_json[str_solver]['time'] = round(elapsed_time)
    to_json[str_solver]['optimal'] = is_optimal
    to_json[str_solver]['obj'] = obj
    to_json[str_solver]['sol'] = sol

    # ...and dump to json file
    str_data = str_data.split('.')[0] + '.json'

    with open('./res/' + str_data, 'w') as json_file:
        json.dump(to_json, json_file, indent=4)
    print(f'Solution saved in ./res/{str_data}')
    print(f'Tour: {tour}')
    print(f'Courier: {courier}')
    print(f'Traveled: {traveled}')
    print(f'Int. load: {int_load}')

# Main
# Usage: python3 main.py <gecode/chuffed> <instXY.dat>
def main(argv):
    if len(argv) < 3:
        print('Error: Insufficient n. of parameters provided')
        print(f'Expected: 2, Provided: {len(argv)-1}')
        print('Usage: docker run --rm <container-name> <solver> <data.dat>')
        sys.exit(1)
    
    # Gather parameters
    str_solver = str(argv[1])           # solver name
    str_data = str(argv[2])             # data file (.dat) 

    # Create MiniZinc model
    solver = Solver.lookup(str_solver)
    model = Model('./src/model.mzn')
    instance = Instance(solver, model)

    # Read file
    lines = []
    with open('./inst/'+str_data) as data_file:
        for line in data_file:
            lines.append(line)
    
    # Read data
    m = int(lines[0].rstrip('\n'))      # n. couriers
    n = int(lines[1].rstrip('\n'))      # n. items
    l = list(map(int, lines[2].rstrip('\n').split()))   # courier size
    s = list(map(int, lines[3].rstrip('\n').split()))   # item size
    d = np.zeros((n+1,n+1))   # distance matrix
    for i in range(n+1):
        d_row = list(map(int, lines[i+4].rstrip('\n').split()))
        for j in range(n+1):
            d[i,j] = d_row[j]
    d = d.astype(int)

    # Reduction of number of couriers
    # [TODO]

    # Feed params to the model
    instance['m'] = m
    instance['n'] = n
    instance['l'] = l
    instance['s'] = s
    instance['d'] = d

    # Solve and keep track of time elapsed
    start_time = time.time()
    result = instance.solve(timeout=datetime.timedelta(minutes=5), intermediate_solutions=True)
    elapsed_time = time.time() - start_time

    # Get solution
    # If a solution exists but we have exceeded the max. time (ie >5min)
    # we set optimal = False and show the model
    has_sol = has_solution(result, elapsed_time)
    if has_sol:
        print_solution(result, m, n, elapsed_time, str_solver, str_data)


if __name__ == '__main__':
    main(sys.argv)