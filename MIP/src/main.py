'''
MCVRP using Gurobi

Language: Gurobi Python API
Solver: Gurobi
'''
import sys
import numpy as np
import time
import json
import gurobipy as gp

'''
A:  set of indexes corresponding to the 3D boolean matrix
    the shape is (n,n,m)
N:  set of the items
V:  set of the items plus depot
K:  set of the couriers
C:  dict of capacities for each courier
q:  list of weights for each item
d:  dict of point-to-point distance 
n:  number of items
'''
#A, N, V, K, C, q, d, n
def get_model(A, N, V, K, C, q, d, n):
    model = gp.Model('MCVRP')

    # dict of the distance matrix
    c = { i: v for i, v in np.ndenumerate(d) if i[0] != i[1] }

    # Variables
    x = model.addVars(A, vtype=gp.GRB.BINARY, name='x')                 # routes for each courier
    u = model.addVars(N, vtype=gp.GRB.CONTINUOUS, name ='u')            # helper vector for subtour elimination
    dist = model.addVars(K, vtype=gp.GRB.CONTINUOUS, name='dist')        # distance travelled by each courier
    z = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name='z')           # maximum travelled distance

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
    model.addConstrs(gp.quicksum(x[i,j,k]*q[j] for i in V for j in N if j != i) <= C[k] for k in K)

    # (5) Subtour elimination
    model.addConstrs((x[i,j,k]==1)>>(u[i]+q[j]==u[j]) for i in N for j in N for k in K if i!=j)
    model.addConstrs(q[i] <= u[i] for i in N)
    model.addConstrs((x[i,j,k]==1)>>(u[i] <= C[k]) for k in K for i in N for j in V if i!=j )

    # (6) For each courier we have its distance
    model.addConstrs(gp.quicksum(x[i,j,k]*c[i-1,j-1] for i in V for j in V if j != i) == dist[k] for k in K)
    model.addConstrs(z >= dist[k] for k in K)

    # Objective: minimize the maximum distance
    model.setObjective(z, sense=gp.GRB.MINIMIZE)
    
    return model, x

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
    m = int(lines[0].rstrip('\n'))                      # n. couriers
    n = int(lines[1].rstrip('\n'))                      # n. items
    l = list(map(int, lines[2].rstrip('\n').split()))   # courier size
    s = list(map(int, lines[3].rstrip('\n').split()))   # item size
    d = np.zeros((n+1,n+1))                             # distance matrix
    for i in range(n+1):
        d_row = list(map(int, lines[i+4].rstrip('\n').split()))
        for j in range(n+1):
            d[i,j] = d_row[j]
    d = d.astype(int)

    # Sets
    K = [i+1 for i in range(m)]         # set of couriers
    N = [i+1 for i in range(n)]         # set of items
    V = N + [n+1]                       # set of items + depot
    A = [(i,j,k) for i in V for j in V for k in K if i!=j] # set of 3D-matrix indexes
    q = { i: s[i-1] for i in N }        # dict of item weights
    C = { i: l[i-1] for i in K }        # dict of courier load

    # The Model
    model, x = get_model(A, N, V, K, C, q, d, n)

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
        print("Solver did not converge to an optimal or suboptimal solution.")
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
