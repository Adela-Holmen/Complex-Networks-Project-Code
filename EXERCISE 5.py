import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N = 50   #number of spins (vertices)
M = 1000  #number of sampled configurations
T = 2.5  #temperature (above critical temperature Tc ~ 2.27 for 2D Ising)
J = 1.0  #interaction strength (coupling constant that represents interaction strength between spins)
pi = 0.5  #fraction of nodes with degree 4

#generate a configuration model graph with the given degree distribution
degrees = [4 if np.random.rand() < pi else 1 for _ in range(N)]
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
G.remove_edges_from(nx.selfloop_edges(G))

true_edges = set(G.edges()) #extract true edges as a set of (i, j) pairs

#function to compute energy change if spin at index 'i' is flipped
def delta_energy(spins, i):
    neighbors = list(G.neighbors(i))
    dE = 2 * J * spins[i] * sum(spins[j] for j in neighbors) #calculate change in energy that would result from flipping spin i. (sum of spins of all neighbours j) * (change in S_i: -2 if S_i=1 & +2 if S_i=-1)
    return dE

#Metropolis-Hastings Sampling Function
#Generate M samples from the Ising model in the paramagnetic phase
def metropolis_sample(G, T, M):
    spins = np.random.choice([-1, 1], size=N)  #initial random state - disordered state
    samples = np.zeros((M, N))  #store configurations

    for m in range(M):
        for _ in range(N):  #N updates per sample (sweep)
            i = np.random.randint(0, N)  #pick a random spin
            dE = delta_energy(spins, i) #calculate energy change
            
            if dE < 0 or np.random.rand() < np.exp(-dE / T): #Metropolis acceptance criterion
                spins[i] *= -1  #spin flip
            
        samples[m, :] = spins  # Store sampled configuration
    
    return samples

#generate Ising model configurations
spins = metropolis_sample(G, T, M)

#compute mean-field magnetizations for each spin
def mean_field_approximation(G, T, J, max_iter=100, tol=1e-4):
    N = len(G)
    m = np.zeros(N)  #initial magnetizations (0 for all spins)
    
    for _ in range(max_iter):
        m_new = np.zeros_like(m)
        
        for i in range(N): #loop over each spin (each node in the graph)
            neighbor_magnetizations = np.sum([m[j] for j in G.neighbors(i)]) #sum of the magnetizations of the neighboring spins
            m_new[i] = np.tanh(J / T * neighbor_magnetizations)  #mean-field approximation: the magnetization for spin i is updated based on the neighboring spins and the interaction strength (J) and temperature (T)
        
        if np.max(np.abs(m_new - m)) < tol: #check convergence
            break
        
        m = m_new
    
    return m

m_mean_field = mean_field_approximation(G, T, J) #compute mean-field magnetizations

#compute connected correlations
mean_spins = np.mean(spins, axis=0)  # ⟨S_i⟩ for each spin i
correlations = {}
for i in range(N):
    for j in range(i + 1, N):
        mean_SiSj = np.mean(spins[:, i] * spins[:, j])  # ⟨S_i S_j⟩
        correlations[(i, j)] = mean_SiSj - (mean_spins[i] * mean_spins[j])  #compute c_ij

#sort correlations in descending order (strongest first)
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

#compute Positive Predictive Value (PPV)
num_predictions = list(range(1, len(sorted_correlations) + 1)) #creates a list of numbers from 1 to the length of the sorted_correlations list
true_positive_counts = np.cumsum([(i, j) in true_edges or (j, i) in true_edges for (i, j), _ in sorted_correlations]) #check if (i, j) or (j, i) is in the true_edges set (edges can be undirected, so both (i, j) and (j, i) are valid)
#np.cumsum: calculates the cumulative sum of boolean values (True: 1, False:0). true_positive_counts: cumulative count of true positives.
ppv_values = true_positive_counts / num_predictions  #PPV = true positives / number of predictions

#infer the graph based on mean-field magnetizations
def infer_graph_from_magnetizations(magnetizations, threshold=0.1):
    inferred_graph = nx.Graph() #create an empty graph (undirected)
    N = len(magnetizations)
    
     #iterate over all pairs of nodes (i, j) in the graph
    for i in range(N):
        for j in range(i + 1, N): #avoid adding edges twice by only considering i < j
            if np.abs(magnetizations[i] - magnetizations[j]) < threshold: #if the absolute difference between the magnetizations of nodes i and j is less than the threshold
                inferred_graph.add_edge(i, j) #add an edge between nodes i and j in the inferred graph
    
    return inferred_graph

inferred_graph_mf = infer_graph_from_magnetizations(m_mean_field, threshold=0.1) #infer the graph using mean-field magnetizations
inferred_edges = set(inferred_graph_mf.edges()) #inferred couplings (edges based on the inferred graph)

connected_correlations = []
unconnected_correlations = []
connected_couplings = []
unconnected_couplings = []

for (i, j), c_ij in correlations.items():
    if (i, j) in true_edges or (j, i) in true_edges:
        connected_correlations.append(c_ij)
    else:
        unconnected_correlations.append(c_ij)

#categorize the inferred couplings based on the true edges
for (i, j) in inferred_edges:
    if (i, j) in true_edges or (j, i) in true_edges:
        #this pair is correctly inferred as connected
        connected_couplings.append(1)
    else:
        #this pair is incorrectly inferred as connected
        unconnected_couplings.append(1)

#for unconnected pairs, if no edges exist in the inferred graph, they should be correctly classified as unconnected
for (i, j) in true_edges:
    if (i, j) not in inferred_edges and (j, i) not in inferred_edges:
        #this pair should be unconnected in the inferred graph
        connected_couplings.append(0)  #false negative, not inferred
    else:
        #this pair is unconnected but inferred as connected (false positive)
        unconnected_couplings.append(0)

#plot PPV curve
plt.figure(figsize=(8, 5))
plt.plot(num_predictions, ppv_values, marker='o', linestyle='-', color='b', markersize=3)
plt.xlabel("Number of Predictions (Strongest Correlated Pairs)")
plt.ylabel("Positive Predictive Value (PPV)")
plt.title("PPV vs Number of Predictions")
plt.grid(True)
plt.show()

#plot histograms for connected and unconnected correlations
fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].hist(connected_correlations, bins=20, color='g', alpha=0.7, label='Connected Correlations')
axes[0].set_title("Histogram of Connected Correlations")
axes[0].set_xlabel("Correlation Value")
axes[0].set_ylabel("Frequency")
axes[1].hist(unconnected_correlations, bins=20, color='r', alpha=0.7, label='Unconnected Correlations')
axes[1].set_title("Histogram of Unconnected Correlations")
axes[1].set_xlabel("Correlation Value")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

#plot histograms for inferred couplings
fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].hist(connected_couplings, bins=20, color='g', alpha=0.7, label="1: True Positive\n0: False Negative")
axes[0].set_title("Histogram of Inferred Couplings (Connected Pairs)")
axes[0].set_xlabel("Inferred Coupling (1 for connected, 0 for not)")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[1].hist(unconnected_couplings, bins=20, color='r', alpha=0.7, label="1: False Positive\n0: True Negative")
axes[1].set_title("Histogram of Inferred Couplings (Unconnected Pairs)")
axes[1].set_xlabel("Inferred Coupling (1 for connected, 0 for not)")
axes[1].set_ylabel("Frequency")
axes[1].legend()
plt.tight_layout()
plt.show()

#plot true configuration model graph and inferred graph
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].set_title("True Configuration Model Graph")
nx.draw(G, with_labels=True, node_size=500, font_size=10, edge_color='g', alpha=0.6, ax=axes[0])
axes[1].set_title("Inferred Graph (Mean-Field Approximation)")
inferred_graph_mf = nx.Graph()
inferred_graph_mf.add_edges_from(sorted_correlations[:len(true_edges)])
nx.draw(inferred_graph_mf, with_labels=True, node_size=500, font_size=10, edge_color='r', alpha=0.6, ax=axes[1])
plt.show()
