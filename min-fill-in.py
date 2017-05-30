#!/usr/bin/env python
#! python/bin/user
import sys
import sets
import random
import collections
import igraph
import signal
import time
import igraph
import itertools

EXACT_THRESHOLD = 15
CYCLE_LEN_THRESHHOLD = 6
BRANCH_FILL_IN = []

########## Start : Expected by OPTIL #############
class Killer:
  exit_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit)
    signal.signal(signal.SIGTERM, self.exit)

  def exit(self,signum, frame):
    try: 
      for x, y in Fill_in + BRANCH_FILL_IN:
        print x, y
    except ValueError:
      pass
    sys.exit()
    #self.exit_now = True
########## End : Expected by OPTIL #############

############ Start : Graph Algo ###################
def _is_clique(G, S):
    """
    Input : Graph G and list S of vertices
    Output : True if G[S] is complete graph. False otherwise
    """
    if not S:
        return True
    H = G.subgraph(S)
    if not H.is_simple():
        raise ValueError("Induced Graph needs to be simple")
    n = H.vcount() 
    return H.ecount() == (n * (n - 1) / 2)

def _no_non_edges(G, S):
    """
    Input : Graph G and list S of vertices
    Output : Number of non edges in G[S]
    """
    if not S:
        return 0
    H = G.subgraph(S)
    if not H.is_simple():
        raise ValueError("Induced Graph needs to be simple")
    n = H.vcount() 
    return (n * (n - 1) / 2) - H.ecount()
    

def _non_edge(G, S):
    """
    Input : Graph G and list S of vertices
    Output : u, v in S such that (u, v) is not an edge in G. 
    """
    #if not S:
    #    return 0
    H = G.subgraph(S)
    if not H.is_simple():
        raise ValueError("Induced Graph needs to be simple")
    n = H.vcount()
    vertices = [u for u in H.vs() if H.degree(u) == n - 2]
    return vertices
    
def _is_chordless_cycle(G, S):
    """
    Input : Graph G and list S of vertices
    Output : Returns True if G[S] is a simple cycle in G.
    """
    if not S:
        return False
    if type(S[0]) != igraph.Vertex:
        raise TypeError("Input list S should contain vertices")
    H = G.subgraph(S)
    if not H.is_simple():
        raise ValueError('Graph needs to be simple')
    if not H.is_connected() or max(H.degree()) != 2 or min(H.degree()) != 2:
        raise ValueError("Subset does not induce a cycle")
    return True

def _next_vertex_to_mark(G, Unnumbered, Numbered):
    """
    Input : Graph G and its two subsets Unnumbered and Numberd
    Output : A vertex in Unnumnered that can be marked in Maximumc Cardinality Search.

    Used in : maximum_cardinality_search
    """
    #if type(Unnumbered) != igraph.VertexSeq or type(Numbered) != igraph.VertexSeq:
    #    raise TypeError("Unnumbered and Numbered should be Vertex Sequence")
    if not Unnumbered:
        raise ValueError("Unnumbered Vertices should not be empty")
    
    max_nbd_in_marked = -1
    for u in Unnumbered:
        Nu_marked = [w for w in Numbered if G.are_connected(u, w)]
        if len(Nu_marked) > max_nbd_in_marked:
            max_nbd_in_marked = len(Nu_marked)
            next_vertex = u
    return next_vertex
                                    
def maximum_cardinality_search(G):
    """
    Input : Graph G
    Output : Ordering of V(G) with maximum cardinality search

    Number the vertices from n to 1 in decreasing order. As the next vertex to number, select the vertex adjacent to the largest number of previously numbered vertices, breaking ties arbitrarily.
    """
    if not G.is_simple():
        raise ValueError("G is not a simple graph")
    if G.vcount() <= 3:
        return G.vs()
    Unnumbered = set(G.vs())
    u = random.choice(list(Unnumbered))
    Unnumbered.remove(u)
    Numbered = set([u])
    ordering = [u]
    while Unnumbered:
        v = _next_vertex_to_mark(G, Unnumbered, Numbered)
        Unnumbered.remove(v)
        Numbered.add(v)
        ordering.extend([v])
    ordering.reverse()
    return ordering
            
def is_chordal(G):
    """
    Input : Graph G
    Output : True if G is a chordal graph. False otherwise.
    """
    if G.vcount() < 2:
        return True
    if not G.is_simple():
        raise ValueError("G is not a simple graph") 
    mcs = maximum_cardinality_search(G)
    for i in range(len(mcs)):
        v = mcs[i]
        S = [w for w in mcs[i:] if G.are_connected(v, w)]
        if not _is_clique(G, S):
            return False
    return True

def _find_missing_edge(G, S):
    """
    Input : Graph G, subset S of V(G)
    Output : Missing edge in G[S] if it is not a clique, None otherwise
    """
    if not G.is_simple():
        raise ValueError("G has to be a simple graph")
    for u, w in itertools.product(S, repeat=2):
        if u != w and not G.are_connected(u , w):
            return (u, w)
    return None

def find_chordless_cycle(G):
    """
    Input : Graph G
    Output : Returns subset S of vertices of G such that G[S] is a chordless cycle of length at least 4. Returns None if G is chordal.
    """
    if G.vcount() <= 3:
        return None
    if not G.is_simple():
        raise ValueError("G is not a simple graph")
    H = G.copy()
    mcs = maximum_cardinality_search(H)
    for i in range(len(mcs)):
        v = mcs[i]
        S = [x for x in mcs[i:] if H.are_connected(v, x)]
        # If S is clique, move on to next vertex
        if _is_clique(H, S):
            continue
        (u, w) = _find_missing_edge(H, S)
        # Storing names of imporatant vertices before deletion
        v_name, u_name, w_name = v["name"], u["name"], w["name"]
        del_closed_nbd = [x["name"] for x in igraph.VertexSeq(H, H.neighbors(v))]
        del_closed_nbd.append(v_name)
        del_closed_nbd.remove(u_name)
        del_closed_nbd.remove(w_name)
        H.delete_vertices(del_closed_nbd)
        short_path = H.get_shortest_paths(u_name, to=w_name, mode='all', output='vpath')
        if short_path == None:
            raise ValueError("Something Wrong. Short path can not be none")
        chordless_cycle_name = [v_name] + [H.vs[x]["name"] for x in short_path[0]]
        chordless_cycle = [G.vs().find(name=vname) for vname in chordless_cycle_name]
        return chordless_cycle
    return None

def collect_chordless_cycles(G):
    """
    Input : Graph G
    Output : A, A_cycles, k_lower_bound
    List A of vertices which is hitting set of chodless cycles of G. List A_cycles contains these chordless cycles. k_lower_bound is lower bound on parameter k. 

    Ref : FOCS(1994) : Tractability of Para Problems on Chordal Graphs (k^3 kernel paper)
    """
    if G.vcount() <= 3:
        return [], [], 0
    if not G.is_simple():
        raise ValueError("G is not a simple graph")
    A_name = [] # Stores list of name of verteces
    A_cycles_name = [] # Stores list of cycles
    H = G.copy()
    k_lower_bound = 0 # Puts lower bound on parameter k
    chless_cycle_name = [u["name"] for u in find_chordless_cycle(H)] 
    while chless_cycle_name:
        A_name.extend(chless_cycle_name)
        A_cycles_name.append(chless_cycle_name)
        k_lower_bound += len(chless_cycle_name) - 3
        H.delete_vertices(chless_cycle_name)
        try:
            chless_cycle_name = [u["name"] for u in find_chordless_cycle(H)]
        except TypeError:
            chless_cycle_name = None
        #chless_cycle = find_chordless_cycle(H)

    A = [G.vs().find(name=vname) for vname in A_name]
    A_cycles = [[G.vs().find(name=vname) for vname in cyc_name] for cyc_name in A_cycles_name]
    return A, A_cycles, k_lower_bound

def find_shortest_chordless_cycle(G):
    """
    Input : Graph G
    Output : Returns list of vertices S of vertices of G such that G[S] is a shoretest chordless cycle of length at least 4. 
    Returns None if G is chordal.
    """
    A, A_cycles, k_lower_bound = collect_chordless_cycles(G)
    Len_cycles = [len(cycle) for cycle in A_cycles]
    if A_cycles:
        return A_cycles[Len_cycles.index(min(Len_cycles))]
    else:
        return None

def special_min_separator(G):
    """
    Input : Graph G
    Output : Clique_min_separator; Almost_Clique_min_separator

    Clique_min_separator : Clique minimal separator
    Almost_clique_min_separator : Minimal Separators which are missing one edge from being a clique
    """
    Clique_min_separator = set([])
    Almost_clique_min_separator = set([])
    mcs = maximum_cardinality_search(G)
    H = G.copy()
    last_vertex_nbd = range(len(mcs) + 2) # Max value. Changed at second point.
    for i in range(len(mcs)):
        v = mcs[i]
        S = [w for w in mcs[i + 1:] if H.are_connected(v, w)]
        for x, y in itertools.product(S, repeat=2):
            if x < y and not H.are_connected(v, w):
                H.add_edge(x, y)
        if len(last_vertex_nbd) <= len(S):
            if not last_vertex_nbd:
                # If last_vertex_nbd is empty, then continue
                continue
            if _is_clique(G, last_vertex_nbd):
                name_lvn = tuple([u["name"] for u in last_vertex_nbd])
                if name_lvn not in Clique_min_separator:
                    Clique_min_separator.add(name_lvn)
            elif _no_non_edges(G, last_vertex_nbd) == 1:
                name_lvn = tuple([u["name"] for u in last_vertex_nbd])
                if name_lvn not in Almost_clique_min_separator:
                    Almost_clique_min_separator.add(name_lvn)
        last_vertex_nbd = S
    return Clique_min_separator, Almost_clique_min_separator
############ End : Graph Algo ###################


############ Start : Kernalization ###################
def rr_deg1(G):
    """
    Input : Graph G
    Output : Nothing (Modifies the input graph)

    Applies reduction rule.
    Reduction Rule : Delete all vertices of deg at most 1
    """
    small_degree = [u for u in G.vs() if (G.degree(u) <= 1)]
    G.delete_vertices(small_degree)

def rr_deg2_cut_vertex(G):
    """
    Input : Graph G
    Output : Nothing (Modifies the input graph)

    Applies reduction rule.
    Reduction Rule : Delete vertex v in deg(v) = 2 and v is a cut vertex
    """
    cut_vertices = G.cut_vertices()
    G.delete_vertices([u for u in cut_vertices if G.degree(u) == 2])
    
def find_simplicial(G):
    """
    Input : Graph G
    Ouput : Vertex v such that N[v] is clique if it exits; None otherwise
    """
    for v in G.vs():
        if  _is_clique(G, G.neighbors(v)):
            return v
    return None

def rr_nbd_clique(G):
    """
    Input : Graph G
    Output : Nothing (Modifies the input graph)

    Applies reduction rule.
    Reductin Rule : Delete v for which N[v] is clique.
    """
    v = find_simplicial(G)
    while not v == None:
        G.delete_vertices(v)
        v = find_simplicial(G)

#Definiation Moplex: U \subseteq V(G) is moplex if U is inclusion wise maximal set for which following properties are satisfied
# 1. G[U] is a clique
# 2. N(U) is minimal separator
# 3. N[u] = N[v] for all u, v in U
#
# Ref : ICALP-2012 : Faster Para Algo for Min-Fill-In (3.0^k) algorithm paper

# A deg 2 vertex which is part of a cycle also satisfies all of above property and we call it single moplex.    
def rr_remove_single_moplex(G):
    """
    Input : Graph G
    Output : Set of edges which are part of a minimum fill in of G. It also modifies the input graph.

    Reduction Rule : Graph G on which rr_deg1(), rr_deg2_cut_vertex(), rr_nbd_clique has been applied exhaustively. 

    If deg(u) = 2 then u is single moplex as there is no edge between neighbors of u. If N(u) = {v, w} then this reduction rules add edge v, w and remove vertex u.

    It also adds edge (v, w) to global variable storing solution.
    """
    if not G.is_simple():
        raise ValueError("G needs to be simple")        

    fill_in = []
    try:
        min_deg = min(G.degree())
    except ValueError:
        return fill_in

    while min_deg <= 2:
        rr_deg1(G)
        rr_deg2_cut_vertex(G)
        # Now any degree 2 vertex is part of cycle
        try:
            u = G.vs().find(_degree=2)
        except ValueError:
            break
        # It returns vertex u
        v, w = G.neighbors(u)[0], G.neighbors(u)[1]
        if G.are_connected(v, w):
            G.delete_vertices(u)
        else:
            fill_in.append((G.vs[v]["name"], G.vs[w]["name"]))
            G.add_edge(G.vs[v]["name"], G.vs[w]["name"])
            G.delete_vertices(u)
        try:
            min_deg = min(G.degree())
        except ValueError:
            break
    return fill_in

def rr_max_degree(G):
    """
    Input : Graph G
    Output : Nothing (Modifies the input graph)

    Applies reduction rule.
    Reductin Rule : Delete v for which deg(v) = n - 1.
    """
    if G.vcount() == 0:
        return
    while max(G.degree()) == G.vcount()-1:
        try:
            v = G.vs().find(_degree = G.vcount()-1)
        except ValueError:
            break
        G.delete_vertices(v)


def is_kernel_applicable(G):
    """
    Input : Graph G
    Output : True if any kernalization rule is applicable
    """
    if G.vcount() == 0:
        return False

    return find_simplicial(G) or min(G.degree()) <= 2 or max(G.degree()) == G.vcount() - 1 
        
def kernel_without_para(G):
    """
    Input : Graph G
    Output : It does not return anything but changes the input graph.

    Apply simple reduction rules without knowledge of parameter k 
    """
    rr_deg1(G)
    rr_nbd_clique(G)
    rr_deg2_cut_vertex(G)
    rr_max_degree(G)

def kernel_add_edges(G):
    """
    Input : Graph G
    Output : List of Minimum Fill-in edges that can safely be added to G.

    There exists a minimum fill-in which contains all these edges. Function "kernel_without_para(G)" should have already been applied to G.
    """
    #Above three rules must be applied before applying following rule
    fill_in = []
    if not G.vcount():
        return fill_in
    while is_kernel_applicable(G):
        try:
            kernel_without_para(G)
            fill_in.extend(rr_remove_single_moplex(G))
        except ValueError:
            break
    return fill_in
############ End : Kernalization ###################


############ Start : Exact Algo ###################
def reachable(G, S, v):
    """
    Input : Graph G, S subset of V(G), a vertex v in V(G) - S
    Output : Set W subset of V(G) - S - v such that for every w in W there exists a path from v to w whose interval vertices are contained S. Notice that W also includes neighbors of v which are not in S.
    """
    if not S:
        raise ValueError("S can not be empty")
    if v in S:
        raise ValueError("Malformed instance. v should not be in S")
    X = set(list(S) + [v])
    H = G.subgraph(list(S) + [v])
    if not H.is_connected():
        for vl in H.clusters():
            Unames = [u["name"] for u in igraph.VertexSeq(H, vl)]
            if v["name"] in Unames:
                X = [igraph.VertexSeq(G).find(name=u_na) for u_na in Unames]
                break
    NbdX = set([])
    for x in X:
        NbdX.update(igraph.VertexSeq(G, G.neighbors(x)))
    NbdX.difference_update(X)
    return NbdX

def exact_trangulate_order(G):
    """
    Input : Graph G (connected, simple)
    Output : (MF, order) where MF is minimum fill in edges and order is Ordering on vertices which will produce minumum fill in.
    
    Uses 2^n exact algorithm to solve an instance.
    If number of vertices in G is at most 15 then it returns quick result.
    """
    if not G.is_simple():
        raise ValueError("G needs to be a simple graph")
    if not G.is_connected():
        raise ValueError("Apply it on connected componenets")
    if G.vcount() < 3:
        return G.vs()
    Nlist = range(G.vcount())
    DP_table = {}
    DP_table_new = {}
    for i in Nlist:
        DP_table_new[frozenset([i])] = [G.degree(i), [i]]
    for i in range(2, len(Nlist) + 1):
        DP_table = DP_table_new.copy()
        DP_table_new.clear()
        for L in itertools.combinations(Nlist, i):
            MF_L = len(Nlist) ** 2
            for x in L:
                L_temp = list(L)
                L_temp.remove(x)
                S = igraph.VertexSeq(G, L_temp)
                v = igraph.VertexSeq(G)[x]
                MF_temp = DP_table[frozenset(L_temp)][0] + len(reachable(G,S,v))
                if MF_L > MF_temp:
                    MF_L = MF_temp
                    order = DP_table[frozenset(L_temp)][1][:]
                    order.append(x)
            DP_table_new[frozenset(L)] = [MF_L, order]

    MF, order = DP_table_new[DP_table_new.keys()[0]]
    return MF - G.ecount(), order

def order_fill_in(G, orderID):
    """
    Input : Graph G, order on vertex IDs
    Output : Fill in edges with respect to this order.
    """
    H = G.copy()
    H.simplify()
    order = H.vs(orderID)
    minimal_fill_in = []
    for i in range(len(order)):
        v = order[i]
        S = [w for w in order[i:] if H.are_connected(v, w)]
        for x, y in itertools.product(S, repeat=2):
            if x < y and not H.are_connected(x, y):
                H.add_edge(x["name"], y["name"])
                minimal_fill_in.extend([(x["name"], y["name"])])
    return minimal_fill_in

def exact_trangulate(G):
    """
    Input : Graph G
    Output : Set of minimum Fill In edges
    """
    if not G.is_simple():
        raise ValueError("G needs to be a simple graph")
    if not G.is_connected():
        raise ValueError("Apply it on connected componenets")
    if G.vcount() < 3:
        return []
    if  is_chordal(G):
        return []

    MF, order = exact_trangulate_order(G)
    Fill_in_edges = order_fill_in(G, order)
    return Fill_in_edges
############ End : Exact Algo ###################


############ Start : Branching ##################
def _guess_chord(G, A):
    """
    Input : Graph G, list of vertices A which forms a cycle.
    Output : Two vertices u, w which is a chord in A.

    For every pair of non adjacent vertices, program checks vertex disjoint paths between them and returns chord between u, w if it is maximum for these two. In case of ties, returns u, w which are at maximum distance aparat.
    """
    H = G.copy()
    min_ver_conn = 0
    dist_between = 0
    alen = len(A)
    for u, w in itertools.product(A, repeat=2):
        if u >= w or H.are_connected(u, w):
            continue
        ver_conn = H.vertex_connectivity(u.index, w.index) 
        if ver_conn >= min_ver_conn:
            min_ver_conn =  ver_conn
            u_sp, v_sp = u, w
    return u_sp, v_sp
    
def guess_branch1(G):
    """
    Input : Graph G
    Output : Fill In edges of G.

    At each smallest cycle [a, b, c, d], the algorithm adds edge ac if number of vertex disjoint paths between a, c are higher than that of b, d. It adds edge b, d otherwise.

    Finds all vertex disjoint cycles and branch on each of them.

    Solution it returns need not be optimal.
    """
    if not G.is_simple():
        raise ValueError("G should be simple graph")
    #if not G.is_connected():
    #    raise ValueError("G should be connected for better results")
    if is_chordal(G):
        return []
    sol = []
    H = G.copy()
    while not is_chordal(H):
        kernel_without_para(H)
        temp_sol = kernel_add_edges(H)
        sol.extend(temp_sol)
        A, A_cycles, k_lw = collect_chordless_cycles(H)
        if not A:
            break
        for cycle in A_cycles:
            u, w = _guess_chord(H, cycle)
            H.add_edge(u, w)
            sol.extend([(u["name"], w["name"])])
    return sol

def guess_branch2(G):
    """
    Input : Graph G
    Output : Fill In edges of G.

    At each smallest cycle [a, b, c, d], the algorithm adds edge ac if number of vertex disjoint paths between a, c are higher than that of b, d. It adds edge b, d otherwise.
    
    Find smallest chordless cycle, branch on it and finds new smallest chordless cycle in modified graph.

    Solution it returns need not be optimal.
    """
    if not G.is_simple():
        raise ValueError("G should be simple graph")
    #if not G.is_connected():
    #    raise ValueError("G should be connected for better results")
    if is_chordal(G):
        return []
    sol = []
    H = G.copy()
    while not is_chordal(H):
        kernel_without_para(H)
        temp_sol = kernel_add_edges(H)
        sol.extend(temp_sol)
        A = find_shortest_chordless_cycle(H)
        if not A:
            break
        u, w = _guess_chord(H, A)
        H.add_edge(u, w)
        sol.extend([(u["name"], w["name"])])
    return sol


def find_all_cycle_trangulation(G, A):
    """
    Input : Graph G, list of vertices A
    Output : List of list of edges which are possible trangulation of cycle A.

    Works only when len(A) \in {4, 5}
    """
    #TODO Check G[A] is chordless cycle and vertices in A are in correct order.
    if len(A) == 4:
        # name of vertices
        u0 = A[0]["name"]; u1 = A[1]["name"]; u2 = A[2]["name"]; u3 = A[3]["name"]  
        return [[(u0, u2)], [(u1, u3)]]

    if len(A) == 5:
        # name of vertices
        u0 = A[0]["name"]; u1 = A[1]["name"]; u2 = A[2]["name"]; u3 = A[3]["name"]; u4 = A[4]["name"]
        return [ [(u0, u2), (u0, u3)],
                 [(u1, u3), (u1, u4)],
                 [(u0, u2), (u2, u4)],
                 [(u0, u3), (u1, u3)],
                 [(u1, u4), (u2, u4)]]

    if len(A) == 6:
        u0 = A[0]["name"]; u1 = A[1]["name"]; u2 = A[2]["name"]; u3 = A[3]["name"]; u4 = A[4]["name"]; u5 = A[5]["name"]
        return [[(u0, u2), (u0, u3), (u0, u4)], [(u1, u3), (u1, u4), (u1, u5)], [(u2, u4), (u2, u5), (u2, u0)],\
                [(u3, u5), (u3, u0), (u3, u1)], [(u4, u0), (u4, u1), (u4, u2)], [(u5, u1), (u5, u2), (u5, u3)],\
                [(u0, u2), (u2, u5), (u3, u5)], [(u1, u4), (u1, u5), (u2, u4)], [(u0, u3), (u0, u4), (u1, u3)],\
                [(u0, u2), (u0, u4), (u2, u4)], [(u1, u3), (u1, u5), (u3, u5)], [(u0, u2), (u0, u3), (u3, u5)],\
                [(u1, u5), (u2, u4), (u2, u5)], [(u0, u4), (u1, u3), (u1, u4)]]


def _branch_on_chords(G, A):
    """
    Input : Graph G, cycle A
    Output : Chords of A to branch on.
    Set (A[1], A[-1]); (A[0], A[2]); ... (A[0], A[-2])
    """
    all_chords = [(A[1], A[-1])]
    for i in range(2, len(A) - 1):
        all_chords.append((A[0], A[i]))
    return all_chords

def branch(G):
    """
    Input : Graph G
    Output : Minimum Fill In edges of G

    Branch on all possible trangulation when lenght of cycle is at most 6. Branch on all possible chords in cycle when cardinality is higher.
    """
    #TODO : Some feasibility check on upper bound
    if not G.is_simple():
        raise ValueError("G should be simple graph")
    if not G.is_connected():
        raise ValueError("G should be connected for better results")
    if is_chordal(G):
        return []

    sol1 = minimal_trangulation(G)
    sol2 = guess_branch1(G)
    sol3 = guess_branch2(G)
    # Dict to store solution optained by minimal_trangulation, branch_guess1, branch_guess2
    greedy_sol = ({len(sol1): sol1, len(sol2): sol2, len(sol3): sol3})
    # Initial solution
    min_sol = greedy_sol[min(greedy_sol.keys())]
    global BRANCH_FILL_IN
    BRANCH_FILL_IN = min_sol
    global_upper_bound = len(min_sol)
    Instances = collections.deque([(G, [])])
    while Instances:
        (H_temp, par_sol_temp) = Instances.pop()
        H = H_temp.copy()
        par_sol = par_sol_temp[:]
        # Apply kernelization rule. If we find some new edge that can be added, add it to par_sol
        kernel_without_para(H)
        temp_sol = kernel_add_edges(H)
        par_sol.extend(temp_sol)
        # Check it H has become chordal. If so, update appropriate values.
        if is_chordal(H):
          #We can update global_upper_bound
          if len(par_sol) < global_upper_bound:
                global_upper_bound = len(par_sol)
                min_sol = par_sol
                BRANCH_FILL_IN = min_sol
          continue
        
        #If H is not a solution but par_sol is larger than upper_bound than move to next Instance
        if len(par_sol) > global_upper_bound:
            continue
        # Apply exact algorithm if feasible
        if H.vcount() <= EXACT_THRESHOLD:
            temp_sol = exact_trangulate(H)
            sol = par_sol + temp_sol
            if len(sol) < global_upper_bound:
                global_upper_bound = len(sol)
                min_sol = sol
                BRANCH_FILL_IN = min_sol
            continue
        # Find chordless cycle.
        A = find_shortest_chordless_cycle(H)
        if len(par_sol) + len(A) - 3 > global_upper_bound:
            continue
        if len(A) > CYCLE_LEN_THRESHHOLD:
            # Either there is edge incident on A[0] or there is edge A[1], A[-1].
            all_chords = _branch_on_chords(H, A)
            for chord in all_chords:
                u, w = chord
                H_temp1 = H.copy()
                H_temp1.add_edge(u, w)
                temp_sol = par_sol + [(u["name"], w["name"])] 
                Instances.append((H_temp1, temp_sol))
            continue            
        All_cycle_trangulation = find_all_cycle_trangulation(H, A)
        for cycle_trang in All_cycle_trangulation:
            H_temp1 = H.copy()
            H_temp1.add_edges(list(cycle_trang))
            temp_sol = par_sol + list(cycle_trang)
            Instances.append((H_temp1, temp_sol))
    return min_sol

  
########### End : Branching #####################


########### Start : Trangulation #################
def minimal_trangulation(G):
    """
    Input : Graph G
    Output : Mimial trangulation of G.

    This is used to upper bound value of $k$.
    """
    H = G.copy()
    H.simplify()
    minimal_fill_in = []
    mcs = maximum_cardinality_search(H)
    for i in range(len(mcs)):
        v = mcs[i]
        S = [w for w in mcs[i:] if H.are_connected(v, w)]
        for x, y in itertools.product(S, repeat=2):
            if x < y and not H.are_connected(x, y):
                H.add_edge(x["name"], y["name"])
                minimal_fill_in.extend([(x["name"], y["name"])])
    return minimal_fill_in


def trangulation(G):
    """
    Input: Graph G
    Output : Minimum Fill In edges of G

    Main function to trangulate the graph.
    """
    Solution = []
    if not G.vcount() or not G.ecount:
        return Solution

    if G.maxdegree() < 2:
        return Solution

    if is_chordal(G):
        return Solution
    
    # We run trangulation algorithm on each biconnected componenet
    if G.cut_vertices() or not G.is_connected():
        for bi_conn in G.biconnected_components():
            H = G.subgraph(bi_conn)
            kernel_without_para(H)
            sol_temp1 = kernel_add_edges(H)
            Solution.extend(sol_temp1)
            if H.vcount() <= EXACT_THRESHOLD:
                sol_temp2 = exact_trangulate(H)
                Solution.extend(sol_temp2)
            else:
                sol_temp2 = branch(H)
                Solution.extend(sol_temp2)
        return Solution

    H = G.copy()
    kernel_without_para(H)
    sol_temp1 = kernel_add_edges(H)
    Solution.extend(sol_temp1)
    if H.vcount() <= EXACT_THRESHOLD:
        sol_temp2 = exact_trangulate(H)
        Solution.extend(sol_temp2)
    else:
        sol_temp2 = branch(H)
        Solution.extend(sol_temp2)
        
    return Solution

########### End : Trangulation ##################


########### Start : Sparse Graphs ###############

def decompose_at_separator(G, S):
    """
    Input : Graph G, list of vertex IDs S
    Output : List of graphs G[C_1 \cup S], G[C_2 \cup S] ... G[C_q \cup S] where C_1, C_2, ... , C_q are connected components of G\S.
    """
    graph_list = []
    S_name = [G.vs()[s]["name"] for s in S]
    H = G.copy()
    H.delete_vertices(S)
    for comp in H.components():
        comp_vertex_names = [H.vs()[u]["name"] for u in comp]
        vertex_temp = S_name + comp_vertex_names
        G_temp = G.subgraph(vertex_temp)
        graph_list.append(G_temp)
    return graph_list

def decompose_at_separator_name(G, S_names):
    """
    Input : Graph G, list of vertex names S_names
    Output : List of graphs G[C_1 \cup S_names], G[C_2 \cup S_names] ... G[C_q \cup S_names] where C_1, C_2, ... , C_q are connected components of G\S_names.
    """
    #TODO FIX THIS
    try:
        S = [G.vs.find(s_name).index for s_name in S_names]
    except ValueError:
        #return False
        pass
    return decompose_at_separator(G, S)


def decompose_graph(G):
    """
    Input : Graph G
    Output : List of graphs, Fill_in_edges that we need to add.

    List of graphs such which are obtained by decompositing graph G at clique minimal separator and almost clique minimal separator.
    """
    # cms == clique_minmial_separator 
    all_cms_names, all_almost_cms_names = special_min_separator(G)
    Fill_in_temp = []
    # Check whether we can add an edge in almost_cms. Apply Lemma 6 in "Faster Para Algo For Minimum Fill-In" by Bodlaeder et al.
    # If edge is added to almost_cms, it will be treated as cms
    for almost_cms in all_almost_cms_names:
        Comm_nbds = set(range(G.vcount()))
        for u_name in almost_cms:
            u = G.vs.find("name"==u_name)
            Nbdu = set(G.neighbors(u))
            Comm_nbds &= Nbdu
        if Comm_nbds:
            try:
              x, y = _non_edge(G, almost_cms)
              Fill_in_temp.extend([(x["name"], y["name"])])
              G.add_edge(x["name"], y["name"])
              all_cms_names.add(almost_cms)
            except ValueError:
              pass
            
    graph_list = [G] # Initialize this list
    for cms in all_cms_names:
        for H in graph_list:
          H_names = set([u["name"] for u in H.vs()])
          #cms_inH = set(cms) & H_names 
          if cms <= H_names :
            graph_list.remove(H)
            # If Graph H is get decomposeed then remove it from list. 
            graph_list.extend(decompose_at_separator_name(H, cms))
    return graph_list, Fill_in_temp


########### End : Sparse Graphs #################

########### Star : Minimum fill In #############

g = igraph.Graph()
Fill_in = [] # Store the global solution
killer = Killer()

while True:
    try:
        str = raw_input()
    except EOFError:
        break
    if str == "":
        break
    if str[0] == "#":
        continue
    x, y = str.split()
    try:
        g.vs.find(name=x)
    except ValueError:
        g.add_vertex(name=x)
    try:
        g.vs.find(name=y)
    except ValueError:
        g.add_vertex(name=y)        
    g.add_edge(x, y)

g.simplify() # Remove multiple edges

fill_in_temp1 = kernel_add_edges(g)
Fill_in.extend(fill_in_temp1)
if is_chordal(g):
    for x, y in Fill_in:
        print x, y
    sys.exit() 

GL, fill_in_temp3 = decompose_graph(g)
Fill_in.extend(fill_in_temp3)
GL.sort(key = lambda g_temp : g_temp.vcount())
for g_temp in GL:
    fill_in_temp2 = trangulation(g_temp)
    Fill_in.extend(fill_in_temp2)

try:
    for x, y in Fill_in:
        print x, y
except ValueError:
    pass
sys.exit()
########### End : Minimum fill In #############
