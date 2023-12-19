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
import os

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

def dump_to_json(str_data: str,
                 str_entry: str,
                 elapsed_time: float, 
                 is_optimal: bool,
                 obj: int,
                 sol: list):
    
    # Collect info for the new entry 
    new_entry = {}
    new_entry[str_entry] = {}
    new_entry[str_entry]['time'] = round(elapsed_time)
    new_entry[str_entry]['optimal'] = is_optimal
    new_entry[str_entry]['obj'] = obj
    new_entry[str_entry]['sol'] = sol

    # Create file_name, eg. 4.json, 13.json ...
    # str_data: inst01.dat
    file_name = ""
    path_file = ""
    if str_data[4] == '0':
        file_name = str_data[5] + '.json'
    else:
        file_name = str_data[4:6] + '.json'
    path_file = './res/' + file_name

    # Check if JSON file exists
    if os.path.exists(path_file):

        # Load existing JSON file
        with open(path_file, 'r') as file:
            data = json.load(file)

        # Append the new entry
        data.update(new_entry)

        # Write the updated data back to JSON file
        with open(path_file, 'w') as file:
            json.dump(data, file, indent=2)

        print('Existing JSON file found. New entry appended.')

    else:
        # Create a new JSON file and add the entry
        with open(path_file, 'w') as file:
            json.dump(new_entry, file, indent=2)
        
        print("New JSON file created with the entry.")

    # debug info
    print(f'Solution saved in {path_file}')
    print(new_entry)

def print_solution(result, m, n, elapsed_time, str_entry, str_data, lowBound, upBound, verbose=True):
    # We get the optimal solution (last in the list)
    best_sol = result.solution[-1]
    tour = best_sol.next
    courier = best_sol.courier
    traveled = best_sol.traveled
    int_load = best_sol.int_load
    
    is_optimal = True
    if elapsed_time >= 299:
        is_optimal = False
    
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
    
    dump_to_json(str_data, str_entry, elapsed_time, is_optimal, obj, sol)

    if verbose:
        print(f'Tour: {tour}')
        print(f'Courier: {courier}')
        print(f'Traveled: {traveled}')
        print(f'Int. load: {int_load}')
        print(f'Upper bound: {upBound}, Lower bound: {lowBound}')

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
    if len(argv) < 4:
        print('Error: Insufficient n. of parameters provided')
        print(f'Expected: 3, provided: {len(argv)-1}')
        print('Usage: docker run --rm <image-name> <gecode/chuffed> <data.dat> <sym_on/sym_off>')
        sys.exit(1)
    
    # Gather parameters
    str_solver = str(argv[1])           # solver name
    str_data = str(argv[2])             # data file (.dat)
    
    # Symmetry activation on-off
    str_entry = None                    # entry name, eg. chuffed_sb
    str_path = None                     # mzn path file
    if str(argv[3]) == 'sym_on':
        str_entry = str_solver + '_sb'
        str_path = './src/model_sym.mzn'
    elif str(argv[3]) == 'sym_off':
        str_entry = str_solver
        str_path = './src/model_no_sym.mzn'
    else:
        print('Error: last parameter should be either sym_on or sym_off.')
        sys.exit(1)

    # Create MiniZinc model
    solver = Solver.lookup(str_solver)
    model = Model(str_path)
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
    has_sol = has_solution(result, elapsed_time)
    
    if has_sol:
        print_solution(result, m, n, elapsed_time, str_entry, str_data, lowBound, upBound)
    else:
        dump_to_json(str_data, str_entry, 300.0, False, 0, []) # Empty solution

if __name__ == '__main__':
    main(sys.argv)