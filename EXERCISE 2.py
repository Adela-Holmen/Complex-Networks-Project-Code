import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import networkx as nx

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
    G = nx.Graph(G) 
    G.remove_edges_from(nx.selfloop_edges(G)) #remove parallel edges and self-loops
    return G

# EXERCISE 2 (Connected components and largest component functions remain the same)
def connected_components(G): #a connected component is a maximal set of nodes where each pair is connected by a path
    components = list(nx.connected_components(G))  #use NetworkX to find connected components
    return components

def largest_component_size(components): #find largest connected component
    return max(len(comp) for comp in components)

def average_lcc_fraction(N, pi, num_realizations=100):
    lcc_sizes = [] #list to store sizes of the largest components from different realizations
    for _ in range(num_realizations):
        G = generate_random_graph(N, pi)  #generate random graph for each realization
        comps = connected_components(G)  #get connected components
        lcc_sizes.append(largest_component_size(comps))  #get size of largest component
    return np.mean(lcc_sizes) / N  #fraction of nodes in the largest component

def find_u(pi, tol=1e-6, max_iter=1000):
    u = 0.0   #initial guess for u, the size of the giant component
    for i in range(max_iter): #iteratively solve for u using a fixed-point iteration
        new_u = ((1 - pi) + 4 * pi * u**3) / (1 + 3 * pi) #update u using the fixed-point equation
        if abs(new_u - u) < tol: #check if the difference between the new and old u is below the tolerance
            return new_u #if the difference is small enough, return the result
        u = new_u #update u for the next iteration
    return u #return the result after max_iter iterations

def analytic_S(pi): #calculate the size of the giant component using the analytic formula
    u = find_u(pi) #find the value of u corresponding to the current pi
    return 1 - ((1 - pi) * u + pi * u**4) #return the size of the giant component


pi_values = np.linspace(0, 1, 50) # #generate 50 values of pi between 0 and 1

#choose a value for N (number of vertices) and the number of graph realizations for averaging
N = 10000
num_realizations = 10
N_values = [100, 500, 1000, 5000]

analytic_results = [analytic_S(pi) for pi in pi_values] #calculate the analytic giant component size for each value of pi
simulation_results = [average_lcc_fraction(N, pi, num_realizations) for pi in pi_values] #calculate the simulated giant component size for each value of pi for N=10000

lcc_sizes_for_different_N = {} #Store the largest component sizes for different N values
for n in N_values: #calculate the simulated giant component size for each value of pi for each value of N_values
    lcc_sizes_for_different_N[n] = [average_lcc_fraction(n, pi, num_realizations) for pi in pi_values]

plt.figure(figsize=(10, 6))
plt.plot(pi_values, analytic_results, label="Analytic Giant Component", lw=2) #plot the analytic result
plt.plot(pi_values, simulation_results, 'o', label="Simulated Largest Component (N=10000)", markersize=5) #plot the simulated result for N=10000 (for reference)
for n in N_values:
    plt.plot(pi_values, lcc_sizes_for_different_N[n], label=f"Simulated Largest Component (N={n})", linestyle='--') #plot the simulated results for different N values
plt.axvline(1/9, color='gray', linestyle='--', label=r'$\pi_c = 1/9$') #add a vertical line for the critical value of pi (1/9)
plt.xlabel('π')
plt.ylabel('Fraction in Giant Component')
plt.title(f'Giant Component vs. π for Various N Values')
plt.legend()
plt.show()
