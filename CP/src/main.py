'''
MCVRP (Multiple Capacity Vechicle Routing Problem)
'''

# Libraries
from minizinc import Instance, Model, Solver
import numpy as np
import numpy.ma as ma
import datetime
import time
import sys
import json

# Print the solution
def has_solution(result, elapsed_time):
    print(f'Time elapsed: {round(elapsed_time)}')
    
    if result.status.has_solution():
        if elapsed_time >= 299:
            print('Time exceeded, but we have at least one feasible solution.')
        else:
            print('Optimal solution found before timeout.')
        return True
    else:
        print(f"No solution found before the timeout.")
        return False

def print_solution(result, m, n, elapsed_time, str_solver, str_data, lowBound, upBound):
    # We get the optimal solution (last in the list)
    best_sol = result.solution[-1]
    tour = best_sol.next
    courier = best_sol.courier
    traveled = best_sol.traveled
    int_load = best_sol.int_load
    
    is_optimal = 'true'
    if elapsed_time >= 299:
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
    print(f'Upper bound: {upBound}, Lower bound: {lowBound}')
    print(to_json)

def read_data(lines):
    m = int(lines[0].rstrip('\n'))      # n. couriers
    n = int(lines[1].rstrip('\n'))      # n. items
    v = n + 1                           # n. verteces
    l = list(map(int, lines[2].rstrip('\n').split()))   # courier size
    s = list(map(int, lines[3].rstrip('\n').split()))   # item size
    d = np.zeros((n+1,n+1))   # distance matrix
    for i in range(n+1):
        d_row = list(map(int, lines[i+4].rstrip('\n').split()))
        for j in range(n+1):
            d[i,j] = d_row[j]
    d = d.astype(int)
    return m, n, v, l, s, d

def low_up_bound(d, n, v):
    # Lower bound
    distances = []
    for i in range(n):
        distances.append(d[-1, i] + d[i, -1])
    lowBound = min(distances)

    # Upper bound with greedy approach (Nearest Neighbor)
    upBound = 0
    min_index = n
    mask = [False for _ in range(v)]
    mask[min_index] = True
    next_nodes = ma.array(d[min_index,:], mask=mask) # At the beginning, everything but the depot is available
    
    while True:
        upBound += next_nodes.min()         # Neigbhour not masked node with minimum distance
        min_index = next_nodes.argmin()

        mask[min_index] = True              # The node becomes masked (visited)
        next_nodes = ma.array(d[min_index,:], mask=mask)

        if np.all(mask):                            # If all the nodes are visited
            upBound += d[min_index, n]              # we go back to depot and stop
            break
    
    return lowBound, upBound

# Main
# Usage: python3 main.py <gecode/chuffed> <instXY.dat>
def main(argv):
    if len(argv) < 3:
        print('Error: Insufficient n. of parameters provided')
        print(f'Expected: 2, provided: {len(argv)-1}')
        print('Usage: docker run --rm <image-name> <gecode/chuffed> <data.dat>')
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
    m, n, v, l, s, d = read_data(lines)

    # Preprocessing
    lowBound, upBound = low_up_bound(d, n, v)

    # Debug
    # print(lowBound, upBound)

    # Feed params to the model
    instance['m'] = m
    instance['n'] = n
    instance['l'] = l
    instance['s'] = s
    instance['d'] = d
    instance['lowBound'] = lowBound
    instance['upBound'] = upBound

    # Solve and keep track of time elapsed
    start_time = time.time()
    result = instance.solve(timeout=datetime.timedelta(minutes=5), intermediate_solutions=True)
    elapsed_time = time.time() - start_time

    # Get solution
    # If a solution exists but we have exceeded the max. time (ie >5min)
    # we set optimal = False and show the model
    has_sol = has_solution(result, elapsed_time)
    if has_sol:
        print_solution(result, m, n, elapsed_time, str_solver, str_data, lowBound, upBound)


if __name__ == '__main__':
    main(sys.argv)