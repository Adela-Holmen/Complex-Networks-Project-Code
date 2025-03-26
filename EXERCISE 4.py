import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_configuration_model(N, pi):
    """Generate a random graph using the configuration model with an even degree sum."""
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
    G = nx.Graph(G)  #convert to simple graph
    G.remove_edges_from(nx.selfloop_edges(G)) #remove parallel edges and self-loops
    return G

def metropolis_step(G, spins, beta): #perform a single Metropolis update
    i = np.random.choice(list(G.nodes))
    dE = 2 * spins[i] * sum(spins[j] for j in G.neighbors(i)) #calculate change in energy that would result from flipping spin i. (sum of spins of all neighbours j) * (change in S_i: -2 if S_i=1 & +2 if S_i=-1)
    if dE < 0 or np.random.rand() < np.exp(-beta * dE): #accept the spin flip if dE<0 or if dE>0 with probability exp(-beta * dE)
        spins[i] *= -1 #spin flip

def has_equilibrated(magnetization, window=500, threshold=0.01): #check if the system has reached equilibrium based on magnetization fluctuations
    if len(magnetization) < window: #there needs to be enough magnetization data points to make a reliable assessment 
        return False
    recent = magnetization[-window:]
    return np.std(recent) < threshold #threshold: standard deviation limit below which the system is considered equilibrated

def simulate_ising(G, T, steps, max_equilibration=5000): #simulate the Ising model using Metropolis-Hastings with adaptive equilibration
    beta = 1 / T
    spins = {i: 1 for i in G.nodes}  #initialize all spins to +1 - ordered state
    magnetization = []
    
    #equilibration phase
    for _ in range(max_equilibration):
        metropolis_step(G, spins, beta)
        M = sum(spins.values()) / len(spins)
        magnetization.append(M)
        if has_equilibrated(magnetization):
            break  #stop early if equilibrium is reached
    
    #measurement phase
    magnetization.clear()  #reset magnetization list for measurement
    for _ in range(steps):
        metropolis_step(G, spins, beta)
        if _ % (steps // 100) == 0:  #record periodically
            M = sum(spins.values()) / len(spins)
            magnetization.append(M)
    
    return magnetization

def binder_cumulant(magnetization): #find binder cumulant to compare estimated phase transition point
    M2 = np.mean(np.array(magnetization) ** 2)
    M4 = np.mean(np.array(magnetization) ** 4)
    return 1 - M4 / (3 * M2 ** 2)

def belief_propagation(G, T, max_iter=1000, tol=1e-5):
    beta = 1 / T
    N = len(G.nodes)
    
    #initialize messages and beliefs to random values.
    messages = { (i, j): np.random.uniform(-0.1, 0.1) for i in G.nodes for j in G.neighbors(i)}
    beliefs = {i: 0 for i in G.nodes} #spins are equaly liekly to be -1 or +1 at the start - disordered state
    
    for _ in range(max_iter):
        #update messages
        max_diff = 0  #track convergence
        for i in G.nodes:
            for j in G.neighbors(i):
                #calculate the incoming messages to node i from all neighbors of node i (excluding neighbor j)
                incoming_messages = 0 
                for k in G.neighbors(i):
                    if k != j:
                        incoming_messages += messages[(k, i)]
                
                #update the message from i to j
                new_message = np.tanh(beta * incoming_messages) #tanh ensures message stays between -1 and +1. Beta controls thermal fluctuations.
                max_diff = max(max_diff, np.abs(new_message - messages[(i, j)])) #find max_diff for convergence check
                messages[(i, j)] = new_message
        
        #update beliefs
        for i in G.nodes:
            #update the belief at node i (it's the sum of all incoming messages). 
            belief = 0 #belief = local estimation of spin at node i, influenced by incoming messages from all neighbors.
            for j in G.neighbors(i):
                belief += messages[(j, i)]  #sum the messages from all neighbors
            beliefs[i] = np.tanh(beta * belief) #belief comstrained between -1 and +1
        
        #check for convergence
        if max_diff < tol:
            break
    
    #compute the magnetization from beliefs
    magnetization = sum(beliefs.values()) / N
    return magnetization

#population dynamics algorithm to compute effective field distribution
def population_dynamics(T, pi, num_pop, num_iterations=1000, tolerance=1e-6):
    beta = 1 / T  
    population = np.random.uniform(-1, 1, num_pop)  #initial population of fields (random starting points) 
    #(not all zero: prevent stagnation, more realistic, avoid trivial solutions, better chance of converging to a true equilibrium)
    
    #convergence loop (we will stop when the change in effective field is less than tolerance)
    for _ in range(num_iterations):
        new_population = np.zeros(num_pop)

        for i in range(num_pop):
            k = 4 if np.random.rand() < pi else 1  #choose the degree (4 or 1)
            neighbors = np.random.choice(population, k, replace=True)  #pick random neighbors

            #effective field update using equation from lectures
            new_effective_field = (1 / (2 * beta)) * np.sum(
                np.log(np.cosh(beta * (neighbors + 1))) - np.log(np.cosh(beta * (neighbors - 1)))
            )
            new_population[i] = new_effective_field  #update effective field

        
        population = new_population  #update population
        if np.mean(np.abs(new_population - population)) < tolerance: #check for convergence
            break

    
    return population #steady-state effective field distribution

N = 1000
pi_values = [0.2, 0.5, 0.8]
T_values = np.linspace(1.5, 3.5, 10)
steps = 20000
pi_values_pop_dyn = np.linspace(0, 1, 5)  
T_values_pop_dyn = np.linspace(0.5, 3.5, 50)  
num_pop = 10000  #population size (equivalent to number of spins)
magnetizations = {} #dict
binder_values = {} #dict

for pi in pi_values:
    magnetizations[pi] = {}
    binder_values[pi] = []
    for T in T_values:
        G = generate_configuration_model(N, pi)
        magnetizations[pi][T] = simulate_ising(G, T, steps)
        binder_values[pi].append(binder_cumulant(magnetizations[pi][T]))

#run simulations for both BP and MCMC
bp_magnetizations = {}
mc_magnetizations = {}

for pi in pi_values:
    bp_magnetizations[pi] = []
    mc_magnetizations[pi] = []
    
    for T in T_values:
        #Belief Propagation
        G = generate_configuration_model(N, pi)
        bp_magnetization = belief_propagation(G, T)
        bp_magnetizations[pi].append(bp_magnetization)
       
        #Monte Carlo (MCMC)
        G = generate_configuration_model(N, pi)
        mc_magnetization = simulate_ising(G, T, steps)[-1]  #last value after simulation
        mc_magnetizations[pi].append(mc_magnetization)

#compute phase diagram: shows how m changes with T for different pi
phase_diagram = {} #dictionary
for pi in pi_values_pop_dyn:
    magnetizations_pop_dyn = []
    for T in T_values_pop_dyn:
        fields = population_dynamics(T, pi, num_pop)
        magnetization_pop_dyn = np.mean(np.tanh(fields/T))  #mean magnetization
        magnetizations_pop_dyn.append(magnetization_pop_dyn)
    phase_diagram[pi] = magnetizations_pop_dyn

#histograms
for pi in pi_values:
    plt.figure(figsize=(12, 8))
    for j, T in enumerate(T_values):
        plt.subplot(2, 5, j + 1)
        bins = max(10, int(len(magnetizations[pi][T]) ** (1/3))) #cube-root rule: N^1/3.
        plt.hist(magnetizations[pi][T], bins=bins, density=True)
        plt.title(f"pi = {pi}, T = {T:.2f}")
        plt.xlabel("Magnetization M")
        plt.ylabel("Probability Density")
    plt.tight_layout()
    plt.show()

    #Binder cumulant
    plt.figure(figsize=(8, 6))
    plt.plot(T_values, binder_values[pi], label=f"pi = {pi}")
    plt.xlabel("Temperature T")
    plt.ylabel("Binder Cumulant")
    plt.title(f"Binder Cumulant vs Temperature for pi = {pi}")
    plt.legend()
    plt.show()
    
    #estimate phase transition temperature
    #Tc = T_values[np.argmax(np.array(binder_values[pi]))]
    #print(f"Estimated Tc for pi = {pi}: {Tc:.2f}")

#plot BP vs MCMC magnetization
for pi in pi_values:
    plt.figure(figsize=(8, 6))
    plt.plot(T_values, bp_magnetizations[pi], label="BP", linestyle='--', color='b')
    plt.plot(T_values, mc_magnetizations[pi], label="MCMC", color='r')
    plt.xlabel("Temperature T")
    plt.ylabel("Magnetization M")
    plt.title(f"Comparison of BP and MCMC for pi = {pi}")
    plt.legend()
    plt.show()

    #find phase transition temperature based on BP
    Tc_bp = T_values[np.argmax(np.diff(bp_magnetizations[pi])) + 1]  #approximation of T_c from BP
    print(f"Estimated phase transition temperature Tc for BP (pi = {pi}): {Tc_bp:.2f}")
    
    #find phase transition temperature based on MCMC
    Tc_mc = T_values[np.argmax(np.diff(mc_magnetizations[pi])) + 1]  #approximation of T_c from MCMC
    print(f"Estimated phase transition temperature Tc for MCMC (pi = {pi}): {Tc_mc:.2f}")

#plot phase diagram using population dynamics
plt.figure(figsize=(8, 6))
for pi, magnetizations_pop_dyn in phase_diagram.items():
    plt.plot(T_values_pop_dyn, magnetizations_pop_dyn, label=f'π = {pi:.2f}')
plt.xlabel('Temperature T')
plt.ylabel('Magnetization')
plt.legend()
plt.title('Ensemble-Averaged Phase Diagram')
plt.show()
