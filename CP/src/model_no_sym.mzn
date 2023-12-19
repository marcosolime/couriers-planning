include "globals.mzn";

% PARAMETERS
int: m;                        % couriers
int: n;                        % items

% SETS
set of int: COURIER = 1..m;
set of int: ITEM = 1..n;
set of int: LOC = 1..n+1;

% PARAMETERS
array[COURIER] of int: l;      % courier load size
array[ITEM] of 0..max(l): s;   % item size
array[LOC,LOC] of int: d;      % locations (clients + depot)

% NODES
set of int: NODE = 1..n+2*m;
set of int: START_NODE = n+1..n+m;
set of int: END_NODE = n+m+1..n+2*m;

% DISTANCE MATRIX
array[NODE,NODE] of int: dx = 
    array2d(NODE,NODE,[
        if i<=n /\ j<=n then d[i,j]         % original d matrix
        elseif i<=n /\ j>n then d[i,n+1]    % customer-depot
        elseif i>n /\ j<=n then d[n+1,j]    % depot-customer
        else d[n+1,n+1]                     % depot-depot (0)
        endif
    | i,j in NODE ]);

% OBJECTIVE BOUNDS
int: lowBound;                              % lower bound
int: upBound;                               % upper bound

% VARIABLES
array[NODE] of var NODE: next;                              % following nodes
array[NODE] of var NODE: back;                              % previous nodes
array[NODE] of var COURIER: courier;                        % couriers assignments
array[NODE] of var 0..max(l): int_load;                     % intermediate load of couriers
array[COURIER] of var lowBound..upBound: traveled;          % distance traveled by each courier
var int: obj;                                               % max distance to minimize

% CONSTRAINTS

/*
    Constraint 0:
    We link the starting nodes with the end nodes
*/
constraint forall(i in (n+2..n+m))(back[i] = i+m-1);
constraint back[n+1] = n+2*m;

/*
    Contraint 1:
    Once the couriers end the tour, go back to start
*/
constraint forall(i in (n+m+1..n+2*m-1))(next[i] = i-m+1);
constraint next[n+2*m] = n+1;

/*
    Contraint 2:
    Each start/end node is associated with its courier
*/
constraint forall(i in START_NODE)(courier[i] = i-n);
constraint forall(i in END_NODE)(courier[i] = i-n-m);

/*
    Constraint 3:
    Initially, each courier is empty
*/
constraint forall(i in START_NODE)(int_load[i] = 0);

/*
    Constraint 4:
    We link the following nodes with the previous nodes
*/
constraint forall(i in NODE) (next[back[i]] = i);
constraint forall(i in NODE) (back[next[i]] = i);

/*
    Constraint 5:
    We create an unique TSP-like circuit, 
    using a global constraint
*/
constraint circuit(next);
constraint circuit(back);

/*
    Constraint 6:
    We make sure the path's couriers are consistent.
*/
constraint forall(i in ITEM)(courier[back[i]] = courier[i]);
constraint forall(i in ITEM)(courier[next[i]] = courier[i]);

/*
    Constraint 7:
    We accumulate the item weights during the path
*/
constraint forall(i in START_NODE)(int_load[i] = int_load[next[i]]);
constraint forall(i in ITEM)(int_load[i]+s[i] = int_load[next[i]]);

/*
    Constraint 8:
    We check whether the the accumulated weight
    went over the courier capacity
*/
constraint forall(i in ITEM)(int_load[i] <= l[courier[i]]);
constraint forall(i in COURIER)(l[i] >= int_load[n+m+i]);

/*
    Constraint 9:
    Couriers with the same capacity do different routes.
    In other words, we do not want them to touch the same nodes.
*/
/*
function var int: is_equal(var COURIER: a, COURIER: b) = 
    if a==b then 1
    else 0
    endif;
constraint forall(i,j in COURIER)
    ((l[i]==l[j] /\ j>i) -> lex_less([is_equal(courier[k],i) | k in ITEM],[is_equal(courier[w],j) | w in ITEM]));
*/

/*
    Constraint 10:
    We avoid symmetric paths, that is distributing the same items but in opposite order.
    Actually, this is a wrong assumption if the matrix is not symmetric, that is 
    d[i,j] != d[j,i] for i,j in NODE. To let the model find optimal solutions 
    with non-symmetric matrices, we relax this constraint.
*/
/*
function var int: getback(var int: i, int: c) = 
    if courier[i] == c then back[i]
    else 0 endif;
function var int: getnext(var int: i, int: c) = 
    if courier[i] == c then next[i]
    else 0 endif;
constraint forall(c in COURIER) (lex_less([getback(j,c) | j in NODE], [getnext(i,c) | i in NODE]));
*/

/*
    Constraint 11:
    Couriers with more capacity deliver more weight than smaller or equal couriers
*/
/*
constraint forall(i,j in COURIER)
    ((l[i]>l[j] /\ i!=j) -> int_load[n+m+i] >= int_load[n+m+j]);
*/

% OBJECTIVE
constraint forall(i in COURIER)(traveled[i] = sum(j in NODE where courier[j]==i)(dx[j,next[j]]));
constraint obj = max(traveled);

/*
    Solve strategies:
    - With Gecode we use a combination of dom_w_deg and first_fail
    - With Chuffed we mostly keep the default search
    - Note: On small instances, restarting methods slow down the search
*/

/*
solve
:: seq_search(
    [int_search(next, dom_w_deg, indomain_random),
    int_search(back, dom_w_deg, indomain_random),
    int_search(courier, first_fail, indomain_random),
    int_search(int_load, first_fail, indomain_min),
    relax_and_reconstruct(next, 92)
    ])
minimize obj;
*/

/*
    restart_luby(n*n)
    restart_geometric(2.0, n*n)
*/

solve minimize obj;
