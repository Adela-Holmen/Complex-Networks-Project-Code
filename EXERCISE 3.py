import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def generate_random_graph(N, pi):
    degrees = []
    for _ in range(N):
        if np.random.rand() < pi:
            degrees.append(4)
        else:
            degrees.append(1)

    total_degree = sum(degrees) 
    if total_degree % 2 != 0:
        #Flip the degree of one vertex (in this case, the first one) so that the sum becomes even. 
        #One could also resample the entire sequence, but this guarantees that the sum becomes even and is also computationally less expensive.
        if degrees[0] == 1:
            degrees[0] = 4 
        else:
            degrees[0] = 1 
        total_degree = sum(degrees)

    G = nx.configuration_model(degrees)
    G = nx.Graph(G) #convert to simple graph
    G.remove_edges_from(nx.selfloop_edges(G)) #remove parallel edges and self-loops
    return G

#3-core algorithm: iteratively remove vertices of degree < 3 and their incident edges. 
#3-core is the first graph having no removable vertices.
def compute_3_core(G):
    while True:
        to_remove = [node for node, degree in G.degree() if degree < 3]
        if not to_remove:
            break
        G.remove_nodes_from(to_remove)
    return G

#Self-consistency equation
def self_consistency_eq(S, pi):
    return S - pi * (1 - (1 - S)**2)


#Find the analytical critical threshold pi_c. Solve S=0 in the self-consistency equation.
#S=0 is the point just before the 3-core begins to emerge. The first non-zero value of S corresponds to pi_c.
def analytical_threshold():
    result = fsolve(lambda pi: self_consistency_eq(1e-5, pi), 0.5)
    return result[0]

#compute analytical value of the 3-core size S for a given pi
def analytical_S(pi):
    S = fsolve(lambda S: self_consistency_eq(S, pi), 0.5)[0]
    return max(0, min(S, 1))  #clamp S between 0 and 1 (the 3-core size S represents the fraction of nodes in the graph)


N = 10000  #number of nodes
pis = np.linspace(0, 1, 50)  #range of pi values
core_sizes = []  #simulated 3-core sizes
analytical_S_vals = []  #analytical 3-core sizes

for pi in pis:
    G = generate_random_graph(N, pi)
    core = compute_3_core(G)
    core_sizes.append(core.number_of_nodes() / N) #(the number of nodes that remain in this 3-core) / (total number of nodes in the original graph) = fraction of nodes in 3-core
    analytical_S_vals.append(analytical_S(pi))

sim_threshold = pis[next(i for i, x in enumerate(core_sizes) if x > 0)] #find the simulated threshold: the first value of pi where the 3-core size becomes nonzero

print(f"Analytical Threshold πc: {analytical_threshold():.4f}") 
print(f"Simulated Threshold πc: {sim_threshold:.4f}")

plt.plot(pis, core_sizes, label='Simulated', marker='o')
plt.plot(pis, analytical_S_vals, label='Analytical S', linestyle='--')
plt.axvline(x=analytical_threshold(), color='r', linestyle='--', label='Analytical Threshold')
plt.xlabel('π (Fraction of degree 4 nodes)')
plt.ylabel('Fraction of nodes in 3-core')
plt.title('3-core Size vs π')
plt.legend()
plt.grid(True)
plt.show()
