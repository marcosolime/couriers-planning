import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


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

def symmetrize(matrix):
    '''
    Symmetrize a matrix by averaging values at symmetric positions

    Args:
    mat (numpy.ndarray): The input matrix.

    Returns:
    numpy.ndarray: The symmetrized matrix.
    '''

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Input matrix must be square.')

    v = matrix.shape[0]
    sym_matrix = np.zeros((v,v))
    for i in range(v):
        for j in range(i, v):
            sym_matrix[i, j] = (matrix[i, j] + matrix[j, i]) / 2
            sym_matrix[j, i] = sym_matrix[i, j]

    return sym_matrix

def main(argv):
    INST_LIM = 30   # no labels drawn beyond this number of verteces (to avoid mess)

    # Check paramters
    if len(argv) < 3:
        print('Error: Insufficient n. of parameters provided')
        print(f'Expected: 2, provided: {len(argv)-1}')
        print('Usage: docker run --rm <image-name> <instXY.dat> <resXY.json>')
        sys.exit(1)

    # Gather parameters
    str_data = str(argv[1])             # data file (.dat)
    str_res = str(argv[2])              # result file (.json) 

    # Read data file
    lines = []
    with open('./inst/'+str_data) as data_file:
        for line in data_file:
            lines.append(line)
    m, n, v, l, s, d = read_data(lines)

    # Symmetrize and reduce
    sym_d = symmetrize(d)
    mds = MDS(n_components=2, metric=False, dissimilarity='precomputed', normalized_stress=True)
    xy = mds.fit_transform(sym_d)

    # Read json
    res_json = {}
    lines = ''
    with open('./res/'+str_res) as res_file:
        for line in res_file:
            lines += line
    
    res_json = json.loads(lines)
    
    solver = ""
    if len(res_json.keys()) == 0:
        print("Error: result file appears to be empty.")
        sys.exit(1)
    elif len(res_json.keys()) > 1:
        solvers = list(res_json.keys())
        print(f'Info: many solutions detected ({len(res_json.keys())}). Choose among: {solvers}')
        while True:
            solver = input('Type solver to display: ')
            if solver not in solvers:
                print('Unknown solver, try again.')
                continue
            break
    else:
        solver = next(iter(res_json))

    routes = res_json[solver]['sol'] # [[1, 3, 4], [2, 5, 6]]

    connections = {}
    for i in range(m):
        connections[f'courier_{i+1}'] = []

    for c, route in enumerate(routes):

        # only one node
        if len(route) == 1:
            connections[f'courier_{c+1}'].append((v, route[0]))
            connections[f'courier_{c+1}'].append((route[0], v))
            continue

        # more than one node
        for i, node in enumerate(route):
            if i == 0:
                connections[f'courier_{c+1}'].append((v, node))
                connections[f'courier_{c+1}'].append((route[i], route[i+1]))
            elif i == len(route)-1:
                connections[f'courier_{c+1}'].append((node, v))
                break
            else:
                connections[f'courier_{c+1}'].append((route[i], route[i+1]))

    # Draw node on the map
    plt.scatter(xy[:, 0], xy[:, 1], c='blue', marker='o')
    
    # To avoid mess, draw labels only on small instances
    if n < INST_LIM:
        for i, point in enumerate(xy):
            label = f'{i+1}' if i<n else 'depot'
            plt.annotate(f'{label}', (point[0], point[1]), textcoords='offset points', xytext=(0,10), ha='center')

    # Link the nodes with dashed lines
    for c in range(m):
        arcs = connections[f'courier_{c+1}']
        color = np.random.rand(3)
        for arc in arcs:
            start_point = xy[arc[0]-1]
            end_point = xy[arc[1]-1]
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            plt.arrow(start_point[0], start_point[1], dx, dy, color=color, linestyle='dashed', head_width=0.03, head_length=0.03)

    # print(connections)
    plt.grid()
    plt.savefig(f'./out/out_{solver}.png')
    plt.show()
    print(f'Info: plot saved in out/out_{solver}.png')
    # Write distances and weights

    # Put legends with colors

    # Save to png


if __name__ == '__main__':
    main(sys.argv)