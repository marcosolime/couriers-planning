import sys
import numpy as np
import numpy.ma as ma
from z3 import *
import json
import time
import signal

def read_data(lines):
    m = int(lines[0].rstrip('\n'))                      # n. couriers
    n = int(lines[1].rstrip('\n'))                      # n. items
    v = n + 1                                           # n. verteces
    l = list(map(int, lines[2].rstrip('\n').split()))   # courier size
    s = list(map(int, lines[3].rstrip('\n').split()))   # item size
    d = np.zeros((v,v))                             # distance matrix
    for i in range(v):
        d_row = list(map(int, lines[i+4].rstrip('\n').split()))
        for j in range(v):
            d[i,j] = d_row[j]
    d = d.astype(int)
    return m, n, v, l, s, d

def on_model(model, params):
    x, obj, m, v, start_time, str_data, str_entry = params
    elapsed_time = time.time() - start_time
    sol = get_solution(m, v, model, x)
    dump_to_json(str_entry, str_data, round(elapsed_time), False, model[obj], sol, True)

def exactly_k(vars, k):
  '''
  === Arguments ===

  vars:     list of Bool variables
  k:        integer constant >= 0

  === Returns ===

  Z3 and-expression where we have exactly k variables set to True in list vars
  '''
  return And(AtMost(*vars, k), AtLeast(*vars, k))

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

def get_solution(m, v, model, x):
    sol = []
    for k in range(m):
        # Arcs in the form [(start, end), (start,end), ...]
        active_arcs = [(i+1,j+1) for i in range(v) for j in range(v) if model[x[i][j][k]] == True]
        tmp_nodes = []
        # Find first node
        for arc in active_arcs:
            if arc[0] == v:
                next_node = arc[1]
                tmp_nodes.append(next_node)
                break
        depot_reached = False
        while not depot_reached:
            for arc in active_arcs:
                if arc[0] == next_node:
                    next_node = arc[1]
                    # End of lap
                    if next_node == v:
                        depot_reached = True
                        break
                    tmp_nodes.append(next_node)
        sol.append(tmp_nodes)
    return sol

def dump_to_json(str_entry: str, 
                 str_data: str,
                 elapsed_time: float,
                 is_optimal: bool,
                 obj, 
                 sol: list, 
                 is_intermediate: bool):
    '''
    === Arguments ===

    str_entry:         'z3' or 'z3_sb'
    str_data:           'inst01.dat'
    elapsed_time:       time interval to solve the instance in seconds
    is_optimal:         True is optimal, False if unknown
    obj:                best found objective value
    sol:                list of nodes visited by each couriers
    is_intermediate:    True if it is a solution found during search, False if final solution
    
    === Description ===

    Writes the solution under res folder in json format
    '''
    new_entry = {}
    new_entry[str_entry] = {}
    new_entry[str_entry]['time'] = round(elapsed_time)
    new_entry[str_entry]['optimal'] = is_optimal
    new_entry[str_entry]['obj'] = int(str(obj))
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
        
    if is_intermediate:
        print('Intermediate solution dumped to json in res folder')
    else:
        print('Final solution dumped to json in res folder')
    print(new_entry)
    print()
    
def print_courier_load(m, model, tot_load, l):
    print('- courier load')
    for i in range(m):
        print(f'\t- courier {i}: {model[tot_load[i]]}/{l[i]}')

def print_courier_dist(m, model, dist, lowBound, upBound):
    print(f'- covered distance (bounds: {lowBound}-{upBound})')
    for i in range(m):
        print(f'\t- courier {i}: {model[dist[i]]}')

def main(argv):
    TIMEOUT = 300

    # Argument check
    if len(argv) < 3:
        print('Insufficient arguments')
        print('Usage: docker run <image> <instXY.dat> <sym_on/sym_off>')
        sys.exit(1)
    
    # Read file
    str_data = argv[1]
    str_entry = None        # 'z3', 'z3_sb'
    sym_on = None           # True: symmetries on, False: symmetries off

    # On-off symmetries
    if argv[2] == 'sym_on':
        str_entry = 'z3_sb'
        sym_on = True
    elif argv[2] == 'sym_off':
        str_entry = 'z3'
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

    # Debug
    # print(lowBound, upBound)

    # Variables
    x = [[[Bool(f'x_{i}_{j}_{k}') for k in range(m)] for j in range(v)] for i in range(v)]
    u = [Int(f'u_{i}') for i in range(v)]
    dist = [Int(f'dist_{i}') for i in range(m)]
    tot_load = [Int(f'tot_load_{i}') for i in range(m)]
    obj = Int(f'obj')

    # Solver
    solver = Optimize()
    params = (x, obj, m, v, time.time(), str_data, str_entry)
    solver.set_on_model(lambda model: on_model(model, params))
    solver.set('maxsat_engine', 'core_maxsat')
    solver.set('timeout', TIMEOUT*1000)

    # Constraints

    # Setting objective bounds
    for k in range(m):
        solver.add(dist[k] <= upBound)
        solver.add(dist[k] >= lowBound)

    # (1) Avoid looping aroung the same node
    for i in range(v):
        solver.add(And([Not(x[i][i][k]) for k in range(m)]))

    # (2) Each node is visited once
    for j in range(n):
        solver.add(exactly_k([x[i][j][k] for k in range(m) for i in range(v)], 1))
        solver.add(exactly_k([x[j][i][k] for k in range(m) for i in range(v)], 1))

    # (3) Each vehicle depart and arrives at the depot once
    for k in range(m):
        solver.add(exactly_k([x[n][j][k] for j in range(n)], 1))
        solver.add(exactly_k([x[j][n][k] for j in range(n)], 1))

    # (4) We stay under the max capacity constraint
    for k in range(m):
        solver.add(tot_load[k] == Sum([If(x[i][j][k], int(s[j]), 0) for i in range(v) for j in range(n)]))
        solver.add(tot_load[k] <= l[k])

    # (5) N. arcs in is equal to N. arcs out
    for k in range(m):
        for j in range(v):
            in_arcs = Sum([If(x[i][j][k], 1, 0) for i in range(v)])
            out_arcs = Sum([If(x[j][i][k], 1, 0) for i in range(v)])
            solver.add(in_arcs == out_arcs)

    # (6) Subtour elimination
    M = 1e5
    solver.add(u[n] == 1)
    for k in range(m):
        for j in range(n):
            for i in range(n):
                solver.add(u[i] + 1 <= u[j] + M*(1 - If(x[i][j][k], 1, 0)))
                solver.add(u[i] + 1 >= u[j] - M*(1 - If(x[i][j][k], 1, 0)))

    # (7) Keep track of the traveled distances
    for k in range(m):
        solver.add(dist[k] == Sum([If(x[i][j][k], int(d[i,j]), 0) for i in range(v) for j in range(v)]))
        solver.add(obj >= dist[k])

    # (Sym break 1)
    # (Couriers with more capacity deliver more weight than smaller couriers)
    if sym_on:

        for k1 in range(m):
            for k2 in range(m):
                if k1!=k2 and is_bigger_mat[k1,k2]:
                    solver.add(tot_load[k1] >= tot_load[k2])

        # (Sym break 2)
        # (Couriers with same capacity do different routes)
        for k1 in range(m):
            for k2 in range(m):
                if k1>k2 and is_equal_mat[k1,k2]:
                    for i in range(v):
                        for j in range(v):
                            solver.add(Not(And(x[i][j][k1], x[i][j][k2])))


    # (OBJ) Minimize the longest distance
    solver.minimize(obj)

    # (Solving)
    is_optimal = False
    status = solver.check()
    elapsed_time = solver.statistics().time
    
    # Statistics
    if elapsed_time >= TIMEOUT:
        print('\nDONE. Solver TERMINATED by timeout')
    else:
        print('\nDONE. Solver found solutions within time limit')

    print('\n*** Statistics ***')
    if status == sat:
        print('Status: SAT', end=' ')

        if elapsed_time <= TIMEOUT:
            is_optimal = True
            print('(optimality proved)')
        else:
            print('(optimality unknown)')

        model = solver.model()

        # Print courier total load
        print_courier_load(m, model, tot_load, l)
        
        # Print courier total distance
        print_courier_dist(m, model, dist, lowBound, upBound)

        # Gathering the routes
        sol = get_solution(m, v, model, x)

        # Dumping to json
        dump_to_json(str_entry, str_data, elapsed_time, is_optimal, model[obj], sol, False)

    elif status == unknown:
        print('Status: UNKNOWN')
        print('Check res folder if solver found any solution')

    else:
        print('Status: UNSAT')


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
        dump_to_json(str_entry, sys.argv[1], 300.0, False, 0, [], False)
    
    finally:
        # Disable alarm before exiting
        signal.alarm(0)
    


