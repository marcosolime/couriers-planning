'''
MCVRP in PuLP
'''
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, LpStatusInfeasible
from pulp import PULP_CBC_CMD
import sys
import time
import json
import os
import numpy as np
import numpy.ma as ma
import signal

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

def get_is_bigger_matrix(m, l):
    '''
    === Arguments ===

    m:          number of couriers
    l:          list of courier capacities

    === Returns ===

    Numpy boolean array of shape (m,m) where item in [i,j]
    is True if courier i has more capacity than courier j; 
    False otherwise.
    '''
    is_bigger_mat = np.zeros((m,m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if l[i] > l[j]:
                is_bigger_mat[i,j] = True
    return is_bigger_mat

def get_is_equal_matrix(m, l):
    '''
    === Arguments ===

    m:          number of couriers
    l:          list of courier capacities

    === Returns ===

    Numpy boolean array of shape (m,m) where item in [i,j]
    is True if courier i has equal capacity than courier j; 
    False otherwise.
    '''
    is_equal_mat = np.zeros((m,m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if l[i] == l[j]:
                is_equal_mat[i,j] = True
    return is_equal_mat

def dump_to_json(str_entry: str,
                 str_data: str,
                 elapsed_time: float,
                 is_optimal: bool,
                 obj: int,
                 sol: list):
    
    # str_data: inst01.dat

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

    print(new_entry)
    print('Solutions dumped to json in res folder')

def main(argv):
    TIMELIMIT = 300     # We let the solver run for this amount of seconds
    TIMEDELTA = 2       # Time window to catch sub-optimal solutions

    # Check arguments
    if len(argv) < 3:
        print('Insufficient arguments')
        print('Usage: docker run <image> <instXY.dat> <sym_on/sym_off')
        sys.exit(1)
    
    # Read file
    str_data = argv[1]

    # On-off symmetries
    str_entry = None        # 'cbc', 'cbc_sb'
    sym_on = None           # True: symmetries on, False: symmetries off

    # On-off symmetries
    if argv[2] == 'sym_on':
        str_entry = 'cbc_sb'
        sym_on = True
    elif argv[2] == 'sym_off':
        str_entry = 'cbc'
        sym_on = False
    else:
        print('Error: last parameter should be either sym_on or sym_off')
        sys.exit(1)

    lines = []
    with open('./inst/'+str_data) as data_file:
        for line in data_file:
            lines.append(line)

    # Read data
    m, n, v, l, s, d = read_data(lines)
    
    # Preprocessing
    lowBound, upBound = low_up_bound(d, n, v)
    is_bigger_mat = get_is_bigger_matrix(m, l)
    is_equal_mat = get_is_equal_matrix(m, l)

    # The model 
    model = LpProblem("Couriers", sense=LpMinimize)

    # Variables
    x = [[[LpVariable(f"x_{i}_{j}_{k}", cat="Binary") for k in range(m)] for j in range(v)] for i in range(v)]
    u = [LpVariable(f"u_{i}", lowBound=1, upBound=v, cat="Integer") for i in range(v)]
    dist = [LpVariable(f"dist_{k}", lowBound=lowBound, upBound=upBound, cat="Integer") for k in range(m)]
    tot_load = [LpVariable(f"tot_load_{k}", lowBound=min(s), upBound=max(l)) for k in range(m)]
    z = LpVariable("z", lowBound=lowBound, upBound=upBound, cat="Integer")

    # (Objective)
    model.setObjective(z)

    # (Solver)
    solver = PULP_CBC_CMD(timeLimit=TIMELIMIT)

    # Constraints

    # (1) Avoid looping around the same node
    for i in range(v):
        for k in range(m):
            model += x[i][i][k] == 0

    # (2) Each node is visited once
    for j in range(n):
        model += lpSum([x[i][j][k] for k in range(m) for i in range(v)]) == 1
    '''
    # Redundant constraint
    for i in range(n):
        model += lpSum([x[i][j][k] for k in range(m) for j in range(v)]) == 1
    '''
        
    # (3) Each vehicle depart and arrives at the depot once
    for k in range(m):
        model += lpSum([x[n][j][k] for j in range(n)]) == 1   # leave once
        model += lpSum([x[i][n][k] for i in range(n)]) == 1   # arrive once

    # (4) Load constraint
    for k in range(m):
        model += tot_load[k] == lpSum([x[i][j][k]*s[j] for i in range(v) for j in range(n)])
        model += tot_load[k] <= l[k]

    # (5) N. arcs in is equal to N. arcs out
    for k in range(m):
        for j in range(v):
            model += lpSum([x[i][j][k] for i in range(v)]) == lpSum([x[j][i][k] for i in range(v)])

    # (6) Subtour elimination with Big-M trick
    M = 1e5
    model += u[n] == 1
    for k in range(m):
        for j in range(n):
            for i in range(n):
                model += u[i] + 1 <= u[j] + M*(1-x[i][j][k])
                model += u[i] + 1 >= u[j] - M*(1-x[i][j][k])

    '''
    # This is the alternative non linear constraint
    model += u[n] == 1
    for i in range(v):
        for j in range(n):
            for k in range(m):
                model += x[i][j][k] * u[j] >= x[i][j][k] * (u[i]+1)
    '''
    
    # (7) Keep track of the traveled distances
    for k in range(m):
        model += lpSum([x[i][j][k]*d[i,j] for i in range(v) for j in range(v)]) == dist[k]
        model += z >= dist[k]
    
    if sym_on:

        # (Sym break 1)
        # (Couriers with more capacity deliver more weight than smaller couriers)
        for k1 in range(m):
            for k2 in range(m):
                if k1!=k2 and is_bigger_mat[k1,k2]:
                    model += tot_load[k1] >= tot_load[k2]

        # (Sym break 2)
        # (Courieres with same capacity do different paths)
        for k1 in range(m):
            for k2 in range(m):
                if k1>k2 and is_equal_mat[k1,k2]:
                    for i in range(v):
                        for j in range(v):
                            model += x[i][j][k1] + x[i][j][k2] <= 1

    # (Solving)
    start_time = time.time()
    model.solve(solver)
    elapsed_time = time.time() - start_time

    # (Printing the status)
    is_optimal = True
    if elapsed_time >= TIMELIMIT-TIMEDELTA:
        is_optimal = False

    print('Done.')
    if model.status != LpStatusInfeasible:
        print(f'Elapsed time: {round(elapsed_time)} second(s)')
        if is_optimal:
            print(f"Optimal solution found")
        else:
            print(f"Feasible solution found, but may be sub-optimal")
    else:
        print("Problem is either infeasible, unbounded or undefined")
        dump_to_json(str_entry, str_data, elapsed_time, False, 0, []) # Empty solution
        sys.exit(1)

    # (Gathering the routes)
    sol = []
    for k in range(m):
        # Arcs in the form [(start, end), (start,end), ...]
        active_arcs = [(i+1,j+1) for i in range(v) for j in range(v) if x[i][j][k].varValue == 1]
        tmp_nodes = []
        # Find first node
        for arc in active_arcs:
            if arc[0] == n+1:
                next_node = arc[1]
                tmp_nodes.append(next_node)
                break
        depot_reached = False
        while not depot_reached:
            for arc in active_arcs:
                if arc[0] == next_node:
                    next_node = arc[1]
                    # End of lap
                    if next_node == n+1:
                        depot_reached = True
                        break
                    tmp_nodes.append(next_node)
        sol.append(tmp_nodes)

    # (Dumping to json)
    dump_to_json(str_entry, str_data, elapsed_time, is_optimal, int(z.varValue), sol)

    # (Debugging)
    print(f'N. of couriers = {m}')
    for k in range(m):
        print(f'courier {k} with total load {tot_load[k].varValue}, available {l[k]}, load occupancy {(tot_load[k].varValue/l[k])*100:.2f}%')


# Timeout utilities
# Get out of infinite loop if no solutions are found within
# time limit
class TimeoutError(Exception):
    pass

def handle_timeout(signum, frame):
    raise TimeoutError('Timeout reached!')

if __name__ == '__main__':
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.alarm(310)

    try:
        main(sys.argv)
    
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print('KeyboardInterrupt: Exiting.')
        sys.exit(0)

    except TimeoutError as e:
        print(f'Error: {e}')

        # Dump to json 
        str_dummy = None
        if sys.argv[2] == 'sym_on':
            str_dummy = 'cbc_sb'
        else:
            str_dummy = 'cbc'
        dump_to_json(str_dummy, sys.argv[1], 300.0, False, 0, [])

    finally:
        # Disable alarm before exiting
        signal.alarm(0)