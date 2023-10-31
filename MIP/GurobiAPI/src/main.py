'''
MCVRP using Gurobi

Language: Gurobi Python API
Solver: Gurobi
'''
import sys
import numpy as np
import numpy.ma as ma
import time
import json
import gurobipy as gp

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

def get_model(A, N, V, K, C, W, D, n, v, lb, ub, max_capacity, light_item, is_bigger_mat, is_equal_mat):
    '''
    A:              set of indexes corresponding to the 3D boolean matrix
    N:              set of the items
    V:              set of the items plus depot
    K:              set of the couriers
    C:              dictionary of capacities
    W:              dictionary of weights
    D:              dictionary of distances
    n:              number of items
    v:              number of items + depot
    lb:             lower bound on traveled route
    ub:             upper bound on traveled route
    max_capacity:   biggest courier
    light_item:     lightest item
    is_bigger_mat:  2d boolean matrix of shape (m,m). matrix[i,j] = True if load[i] > load[j]
    is_equal_mat:   2d boolean matrix of shape (m,m). matrix[i,j] = True if load[i] == load[j] 
    '''
    model = gp.Model('Couriers')

    # Variables
    x = model.addVars(A, vtype=gp.GRB.BINARY, name='x')                                 # routes for each courier
    u = model.addVars(V, vtype=gp.GRB.INTEGER, lb=1, ub=v, name ='u')                   # auxiliary vector for subtour elimination
    dist = model.addVars(K, vtype=gp.GRB.INTEGER, lb=lb, ub=ub, name='dist')            # distance travelled by each courier
    tot_load = model.addVars(K, vtype=gp.GRB.INTEGER, lb=light_item, ub=max_capacity, name='tot_load')   # load of each courier at the end of the tour
    z = model.addVar(vtype=gp.GRB.INTEGER, lb=lb, ub=ub, name='z')                      # maximum travelled distance

    # Constraints

    # (1) Each node is visited once
    model.addConstrs(gp.quicksum(x[i,j,k] for k in K for i in V if i != j) == 1 for j in N)
    model.addConstrs(gp.quicksum(x[i,j,k] for k in K for j in V if j != i) == 1 for i in N)

    # (2) Each vehicle can reach and leave the depot once
    model.addConstrs(gp.quicksum(x[n+1,j,k] for j in N) == 1 for k in K)
    model.addConstrs(gp.quicksum(x[j,n+1,k] for j in N) == 1 for k in K)

    # (3) We reach and leave each node once
    model.addConstrs((gp.quicksum(x[i,j,k] for i in V if i!=j) - gp.quicksum(x[j,i,k] for i in V if i!=j)) == 0 for j in V for k in K)

    # (4) Load constraint
    model.addConstrs(tot_load[k] == gp.quicksum(x[i,j,k]*W[j] for i in V for j in N if j != i) for k in K)
    model.addConstrs(tot_load[k] <= C[k] for k in K)

    # (5) Subtour elimination
    model.addConstr(u[v] == 1)
    model.addConstrs((x[i,j,k]==1) >> (u[i]+1 == u[j]) for i in N for j in N for k in K if i!=j)

    # (Alternative: Big-M trick)
    # M = 1e5
    # model.addConstrs(u[i] + 1 <= u[j] + M*(1-x[i,j,k]) for i in N for j in N for k in K if i!=j)
    # model.addConstrs(u[i] + 1 >= u[j] - M*(1-x[i,j,k]) for i in N for j in N for k in K if i!=j)

    # (6) For each courier we have its distance
    model.addConstrs(gp.quicksum(x[i,j,k]*D[i-1,j-1] for i in V for j in V if j != i) == dist[k] for k in K)
    model.addConstrs(z >= dist[k] for k in K)

    # (Sym break 1)
    # (Couriers with more capacity deliver more weight than smaller or equal couriers)
    for k1 in K:
        for k2 in K:
            if k1 != k2 and is_bigger_mat[k1-1,k2-1]:
                model.addConstr(tot_load[k1] >= tot_load[k2])

    # (Sym break 2)
    # (Couriers with equal capacity do different paths)
    for k1 in K:
        for k2 in K:
            if k1 > k2 and is_equal_mat[k1-1,k2-1]:
                for i in V:
                    for j in V:
                        if i != j:
                            model.addConstr(x[i,j,k1] <= 1 - x[i,j,k2])

    # Objective: minimize the maximum distance
    model.setObjective(z, sense=gp.GRB.MINIMIZE)
    
    return model, x

def read_data(lines):
    m = int(lines[0].rstrip('\n'))                      # n. couriers
    n = int(lines[1].rstrip('\n'))                      # n. items
    v = n + 1                                           # n. verteces
    l = list(map(int, lines[2].rstrip('\n').split()))   # courier size
    s = list(map(int, lines[3].rstrip('\n').split()))   # item size
    d = np.zeros((n+1,n+1))                             # distance matrix
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

def get_sets(m, n, v, s, l, d):
    '''
    m:      n. of couriers
    n:      n. of items
    v:      n. of items + depot
    s:      list of item sizes
    l:      list of courier capacities
    d:      distance matrix
    '''
    K = [i+1 for i in range(m)]                                         # set of couriers
    N = [i+1 for i in range(n)]                                         # set of items
    V = N + [v]                                                         # set of items + depot
    A = [(i,j,k) for i in V for j in V for k in K if i!=j]              # set of 3D-matrix indexes
    W = { i: s[i-1] for i in N }                                        # dict of item weights
    C = { i: l[i-1] for i in K }                                        # dict of courier load    
    D = { i: dist for i, dist in np.ndenumerate(d) if i[0] != i[1] }    # indexed distance matrix
    return K, N, V, A, W, C, D

def main(argv):

    if len(argv) < 2:
        print('Insufficient arguments')
        print('Usage: docker run <image> <instXY.dat>')
        sys.exit(1)

    # Read file
    str_data = argv[1]
    lines = []
    with open('./inst/'+str_data) as data_file:
        for line in data_file:
            lines.append(line)
    
    # Read data
    m, n, v, l, s, d = read_data(lines)

    # Sets
    K, N, V, A, W, C, D = get_sets(m, n, v, s, l, d)

    # Preprocessing
    lowBound, upBound = low_up_bound(d, n, v)
    max_capacity = max(l)
    light_item = min(s)
    is_bigger_mat = get_is_bigger_matrix(m, l)
    is_equal_mat = get_is_equal_matrix(m, l)

    # The Model
    model, x = get_model(A, N, V, K, C, W, D, n, v, lowBound, upBound, max_capacity, light_item, is_bigger_mat, is_equal_mat)

    # Model configs
    model.setParam('TimeLimit', 300)
    model.setParam('Heuristics', 1)
    model.setParam('MIPFocus', 1)

    # Optimizing
    start_time = time.time()
    model.optimize()
    elapsed_time = time.time() - start_time

    # Check results
    is_optimal = False
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found.")
        is_optimal = True
    
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Optimization has reached the time limit")
    
    else:
        print("Solver did not converge to an optimal or sub-optimal solution.")
        sys.exit(1)


    # Collecting the variables
    # List of nodes visited by each courier
    sol = []
    for k in K:
        active_arcs = [a[:-1] for a in A if x[a].x > 0.99 and a[-1] == k]
        tmp_nodes = []
        # Find first node
        for arc in active_arcs:
            if arc[0] == n+1:
                next_node = arc[1]
                tmp_nodes.append(next_node)
                break
        # Collect nodes
        depot_reached = False
        while not depot_reached:
            for arc in active_arcs:
                if arc[0] == next_node:
                    next_node = arc[1]
                    if next_node == n+1:
                        depot_reached = True
                        break
                    tmp_nodes.append(next_node)
        sol.append(tmp_nodes)
    
    # Dump to json
    to_json = {}
    to_json['gurobi'] = {}
    to_json['gurobi']['time'] = round(elapsed_time)
    to_json['gurobi']['optimal'] = is_optimal
    to_json['gurobi']['obj'] = model.objVal
    to_json['gurobi']['sol'] = sol
    
    str_data = sys.argv[1].split('.')[0] + '.json'
    with open('./res/' + str_data, 'w') as json_file:
        json.dump(to_json, json_file, indent=4)
    
    print(to_json)
    print('Solutions dumped to json in res folder')

if __name__ == '__main__':
    main(sys.argv)
