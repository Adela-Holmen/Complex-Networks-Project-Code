import numpy as np
import random

#EXERCISE 1

def generate_graph(N, pi):
    #Assign a degree to each vertex.
    #Degree 1 with probability 1-pi. Degree 4 with probability pi.
    #The assignment is made the following way: The random() method returns a random floating number uniformly distributed between 0 and 1. 
    #If the number is smaller than pi, the degree is 4. Otherwise, the degree is 1. All degrees are added to the list of degrees.
    degrees = []
    for i in range(N):
        if random.random() < pi:
            degrees.append(4)
        else:
            degrees.append(1)
    
    #Check that the total degree sum is even. Why should it be even? Because of the "handshaking" lemma: 2E = sum(degrees). 
    #In any undirected graph,  each edge connects two vertices. When you sum the degrees of all vertices, you count each edge twice. 
    #Because 2E is always even, the sum of all the vertex degrees must be even. 
    #In config model: if the total number of stubs were odd, it would be impossible to pair them all up into edges (since one stub would be left unpaired). Then we can't create the graph.
    total_degree = sum(degrees)
    if total_degree % 2 != 0:
        #Flip the degree of oone vertex (in this case, the first one) so that the sum becomes even. 
        #One could also resample the entire sequence, but this guarantees that the sum becomes even and is also computationally less expensive.
        if degrees[0] == 1:
            degrees[0] = 4 #in this case, the sum of the degrees increases by 3. ood + odd = even
        else:
            degrees[0] = 1 #in this case, the sum of the degrees decreases by 3. ood - odd = even
        total_degree = sum(degrees) #recalculate the sum of the degrees. 
    
    #Create the list of stubs. (Create a list where each vertex appears as many times as its assigned degree.)
    #Looping over the degrees list. vertex = index of vertex. degree = degree of the vertex.
    #For each vertex, the inner loop runs degree times. In each iteration of the inner loop, the current vertex is appended to the stubs list. 
    #e.g. if a vertex has degree 4, its index will be added 4 times to the list. If vertex 1 has degree 4: [1,1,1,1], and if vertex 2 has degree 1: [1,1,1,1,2], etc.
    stubs = []
    for vertex, degree in enumerate(degrees): 
        for _ in range(degree):
            stubs.append(vertex)
    
    #Initialize an empty graph structure. Which data structure should be used?
    #use a set: Set items are unordered, unchangeable, and do not allow duplicate values. Using a set ensures that each edge is stored only once.
    #Use a set to store edges as sorted tuples (min(u,v), max(u,v)). Why store them this way?
    #Always storing edges as (min(u, v), max(u, v)) makes it easy to check if an edge has already been added, regardless of the order in which the two vertices were paired.
    edges = set()
    
    #Pair stubs at random.
    #Removing stubs one by one until none remain. Each iteration attempts to form one edge by removing two stubs.
    while stubs:
        #Randomly select the first stub and remove it.
        i = random.randrange(len(stubs)) #selects a random index from the current list of stubs.
        v = stubs.pop(i) #removes the stub at that index and assigns it to v. v = vertex associated with this stub.
        
        #Randomly select a second stub from the remaining ones.
        j = random.randrange(len(stubs)) #A second random index is selected from the remaining stubs.
        w = stubs.pop(j) #The stub at that index is removed and assigned to w. w = vertex associated with this second stub.
        
        #Check for forbidden edges: self-loops or multiple edges.
        if v == w or ((min(v, w), max(v, w)) in edges): #v==w: avoid self-loops. ((min(v, w), max(v, w)) in edges): avoid multiple edges.
            #If a forbidden edge is found, they are reinserted and a new pairing is tried.
            stubs.append(v) #reinsert stub v into the stubs list.
            stubs.append(w) #reinsert stub w into the stubs list.
            random.shuffle(stubs) #shuffle the stubs to avoid getting stuck.
            continue
        
        #If the edge fulfils the criteria, it is added to the set of edges.
        edges.add((min(v, w), max(v, w)))
    
    return edges, degrees #returning the set "edges" contains the set of all edges of our graph.

