
import networkx as nx
import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from numpy import sqrt
from enum import Enum

class execution_type(Enum):
    static_part1 = 0
    static_part2 = 1
    static = 2
    dynamic = 3
    
class weight_edges(Enum):
    linear = 0
    variable = 1

class genetic_graph:

    def __init__(self,G,configuration=weight_edges.variable):
        self.G = G
        self.configuration = configuration
        self.createNodes()
        self.createEdges()
        operations = graph_operations(self.G)
        operations.computeClusters()
        nx.draw(self.G)
        plt.show()
    
    def createNodes(self):
        nx.set_node_attributes(self.G,'concept',None)
        rnd.seed()
        value = 0
        for i in self.G.nodes_iter():
            self.G.node[i]['id'] = value
            self.G.node[i]['concept'] = rnd.randint(0,9)
            value = value + 1
   
    def createEdges(self):
        value = 0
        for i in self.G.edges_iter():
            self.G.edge[i[0]][i[1]]['id'] = value
            if self.configuration == weight_edges.variable:
                self.G.edge[i[0]][i[1]]['weight'] = 1
            value = value + 1
        
class graph_operations:
    def __init__(self,G,option=execution_type.dynamic):
        self.nodes = []
        self.G = G
        self.option = option
        self.createHubs()
        
    def createHubs(self):
        self.hub_vertexes = []
        self.non_hub_vertexes = []
        self.HVSs = []
        self.clusters = []    
        self.num_percentage_vertexes = 20
        self.num_hub_vertexes = int(self.G.number_of_nodes() * self.num_percentage_vertexes/100.0)
        self.hub_score = 1
        self.no_hub_score = 0.5
        
    def get_non_hub_vertexes(self):
        return self.non_hub_vertexes
    
    def get_HVSs(self):
        return self.HVSs
         
    def getSalienceRanking(self):
        for i in self.G.nodes_iter():
            new_salience = node_values(self.G.node[i]['id'],len(self.G.neighbors(i)))
            self.nodes.append(new_salience)
        self.nodes = sorted(self.nodes, key = lambda node_values: node_values.get_value())       
        return self.nodes
    
    def computeClusters(self):
        # Obtener los HVS.
        self.initHVS()
        self.generateHVSs()
        # Unir los HVS que presenten una conectividad interna menor que la que tienen entre si.
        if self.option == execution_type.static_part1 or self.option == execution_type.static:
            hvs_connectivity = HVSConnectivityGenetic(self)
            connectivity = hvs_connectivity.evolution()
            self.assignSolutiontoConnectivity(connectivity)
        else:
            self.interconnectHVSs()
        print("HVSs",self.HVSs)
        
        if self.option == execution_type.static_part2 or self.option == execution_type.static:
            hvs_internal = HVSInternalGenetic(self)
            internal = hvs_internal.evolution()
            self.assignSolutiontoHVS(internal)
        else:
            self.similarityHVSs()
        print("HVSs",self.HVSs)    
        # Extraer los que tienen solo un elemento y pasarlos a la lista de non hub vertices.
        self.extractNodesWithOneVertex()
        print("HVSs",self.HVSs)
        # Asignar los 'non hub vertices' a los clusters
        non_hub = NonHubGenetic(self)
        solution = non_hub.evolution();
        self.assignSolutiontoClusters(solution)
        #self.assignNonHubToClusters()
        print("Clusters:")
        for i in range(len(self.clusters)):
            print(self.clusters[i])
            
    def assignSolutiontoConnectivity(self,solution):
        for i in range(0,len(solution)):
            connectivity = solution[i].get_value()[0]
            if connectivity != -1:
                new_connectivity = solution[i].get_value()[1]
                position = solution[i].get_iden()
                self.HVSs[new_connectivity].append(self.HVSs[position][connectivity])
                self.HVSs[position].pop(connectivity) 
        i = 0
        while i in range(0,len(self.HVSs)):
            if len(self.HVSs[i]) == 0:
                self.HVSs.pop(i)
            else:
                i = i + 1        
        
    def assignSolutiontoHVS(self,solution):
        pops = []
        for i in range(0,len(solution)):
            connection = solution[i].get_value()
            if connection != -1:
                position = solution[i].get_iden()
                self.HVSs[position].extend(self.HVSs[connection])
                pops.append(connection)
        for i in range(0,len(pops)):
            self.HVSs.pop(i)   
            
    def assignSolutiontoClusters(self,solution):
        for i in range(len(self.HVSs)):
            self.clusters.append(self.HVSs[i])
        for i in range(0,len(solution)):
            chromosome = solution[i]
            iden = chromosome.get_iden()
            cluster = chromosome.get_value()
            if cluster != -1:
                self.clusters[cluster].append(iden)
            
    def initHVS(self):
        # Obtener los 'n' hub vertices y los 'N-n' no hub vertices.
        ranking = self.getSalienceRanking()
        stop = len(ranking) - self.num_hub_vertexes - 2
        for i in range(len(ranking)-1,stop,-1):
            self.hub_vertexes.append(ranking[i].get_iden())
        print("hubs:",self.hub_vertexes)
        
        start = len(ranking) - self.num_hub_vertexes - 2
        for i in range(start,0,-1):
            self.non_hub_vertexes.append(ranking[i].get_iden())
    
    def generateHVSs(self):   
        # Inicialmente, creamos un HVS por cada Hub Vertice.
        for i in range(len(self.hub_vertexes)):
            iden = self.hub_vertexes[i]
            hvs = []
            hvs.append(iden)
            self.HVSs.append(hvs)
            
    def interconnectHVSs(self):
        # Para cada hub vertice, comprobar si existe un HVS distinto al que pertenece
        #   con el que presente una mayor conectividad que al suyo propio.
        change = True
        while(change):
            change = False
            i = 0
            while (i < len(self.HVSs)):
                vertexes = self.HVSs[i]
                j = 0
                while (j < len(vertexes)):
                    iden = vertexes[j] 
                    intraconnection = self.getConnectionWithHVS(iden,self.HVSs[i])
                    interconnection = self.getMaxConnectionWithHVSs(iden,intraconnection)
                    if interconnection[0] != -1 and interconnection[1] != 0: # Existe otro HVS con el que se encuentra mas conectado.
                        # Cambiar al vertice de HVS.
                        change = True
                        self.HVSs[i].pop(j)
                        self.HVSs[interconnection[0]].append(iden)
                        print iden
                    else:
                        j = j + 1
                if len(vertexes) == 0:
                    self.HVSs.pop(i)
                else:
                    i = i + 1

    def similarityHVSs(self):
        change = True
        while(change):
            change = False
            pops = []
            for i in range(len(self.HVSs)):
                hvs1 = self.HVSs[i]
                j = i
                while (j < len(self.HVSs)):
                    hvs2 = self.HVSs[j]
                    intra_sim1 = self.getIntraSimilarity(hvs1)
                    intra_sim2 = self.getIntraSimilarity(hvs2)
                    inter_sim = self.getInterSimilarity(hvs1,hvs2)
                    if (inter_sim > intra_sim1 or inter_sim > intra_sim2):
                        # Unir ambos HVSs.
                        print ("entra")
                        self.HVSs[i].extend(hvs2)
                        pops.append(j)
                        change = True
                    j = j + 1
            for i in pops:
                print("entra")
                self.HVSs.pop(i)
        
    # Funcion que devuelve el nodo del grafo que tiene el identificador indicado.
    def getNodeFromIden(self,iden):
        result = None
        for i in self.G.nodes_iter():
            node = self.G.node[i]
            if iden == node['id']:
                result = node
                break
        return result                         
    
    # Funcion que devuelve el HVS con el que un concepto presenta una mayor conectividad, si esta supera su conectividad interna.
    def getMaxConnectionWithHVSs(self,iden,intraconnection):
        max_connection = 0.0
        max_position = -1
        result = []
        result.append(-1)
        result.append(-1)
        for i in range(len(self.HVSs)):
            connection = self.getConnectionWithHVS(iden,self.HVSs[i]);
            if (connection > max_connection):
                max_connection = connection
                max_position = i
        if (max_connection > intraconnection):
            result[0] = max_position
            result[1] = max_connection
        else:
            result[0] = -1;
            result[1] = -1;
        return result
    
    # Funcion que devuelve la conectividad de un concepto con respecto a un HVS.      
    def getConnectionWithHVS(self,iden,vertexes):
        node = self.getNodeFromIden(iden)
        print(node)
        neighbors = self.G.neighbors(node['id'])
        connection = 0.0
        for i in range(len(neighbors)):
            neighbor_iden = neighbors[i]
            if neighbor_iden in vertexes:
                neighbor = self.getNodeFromIden(neighbor_iden)
                if self.G.has_edge(node['id'],neighbor['id']):
                    edge_data = self.G.get_edge_data(node['id'],neighbor['id'])
                    connection = edge_data['weight']
                    break
        return connection   
    
    # Funcion que calcula la similitud (conectividad) entre los conceptos de un HVS.
    def getIntraSimilarity(self,vertexes):
        similarity = 0.0;
        for i in range(len(vertexes)):
            iden = vertexes[i]
            node = self.getNodeFromIden(iden)
            neighbors = self.G.neighbors(node['id'])
            for j in range(len(neighbors)):
                neighbor_iden = neighbors[j]
                if neighbor_iden in vertexes:
                    neighbor = self.getNodeFromIden(neighbor_iden)
                    if self.G.has_edge(node['id'],neighbor['id']):
                        edge_data = self.G.get_edge_data(node['id'],neighbor['id'])
                        weight = edge_data['weight']
                        similarity = similarity + weight
        return similarity
    
    # Funcion que calcula la similitud (conectividad) entre dos HVSx.
    def getInterSimilarity(self,hvs1,hvs2):
        similarity = 0.0;
        for i in range(len(hvs1)):
            iden = hvs1[i]
            node = self.getNodeFromIden(iden)
            neighbors = self.G.neighbors(node['id'])
            for j in range(len(neighbors)):
                neighbor_iden = neighbors[j]
                if neighbor_iden in hvs2:
                    neighbor = self.getNodeFromIden(neighbor_iden)
                    if self.G.has_edge(node['id'],neighbor['id']):
                        edge_data = self.G.get_edge_data(node['id'],neighbor['id'])
                        weight = edge_data['weight']
                        similarity = similarity + weight
        return similarity 
    
    # Metodo que elimina los HVSs con conectividad 1.
    def extractNodesWithOneVertex(self):
        i = 0
        while (i < len(self.HVSs)):
            vertexes = self.HVSs[i]
            if len(vertexes) <= 1:
                self.non_hub_vertexes.append(vertexes[0])
                self.HVSs.remove(vertexes)
            else:
                i = i + 1   
       
    # Dado un nodo, devuelve el HVS al que mas se asemeja, y a cuyo cluster.             
    def getMoreSimilarHVS(self,iden):
        max_position = -1
        max_similarity = 0.0
        for i in range(len(self.HVSs)):
            similarity = 0.0
            vertexes = self.HVSs[i]
            for j in range(len(vertexes)):
                hv = vertexes[j]
                hvnode = self.getNodeFromIden(hv)
                node = self.getNodeFromIden(iden)
                pos = self.find(node,hvnode)
                if (pos != -1):
                    edge_data = self.G.get_edge_data(node['id'],self.G.node[pos]['id'])
                    weight = edge_data['weight']
                    similarity = similarity + weight
                if (similarity > max_similarity):
                    max_position = i
                    max_similarity = similarity
        return max_position
    
    def find(self,node1,node2):
        result = -1
        processed = []
        itr = nx.all_neighbors(self.G,node1['id'])
        for i in itr:
            if i not in processed:
                processed.append(i)
                if self.G.node[i]['concept'] == node2['concept']:
                    result = self.G.node[i]['id']
                    break
        return result
    
class HVSConnectivityGenetic():
    def __init__(self,graph_operations,limit=800,size=16,margin_crossover=0.6,prob_crossover=0.9,
        margin_mutation=0.1,prob_mutation=0.4):
        rnd.seed(0)
        self.counter = 0
        self.graph_operations = graph_operations
        self.target = len(self.graph_operations.get_HVSs())
        self.limit = limit
        self.size = size
        self.margin_crossover = margin_crossover
        self.prob_crossover = prob_crossover
        self.margin_mutation = margin_mutation
        self.prob_mutation = prob_mutation
        self.children = []
        
    def init_population(self):
        population = []
        for _ in range(0,self.size):
            chromosome = self.init_chromosome()
            population.append(chromosome)
        return population
    
    def init_chromosome(self):
        chromosome = []
        for i in range(0,len(self.graph_operations.get_HVSs())):
            value = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
            relation = node_values(i,[value,-1])
            chromosome.append(relation)
        return chromosome
         
    def fitness(self,chromosome):
        accurancy = 0
        for i in range(0,len(chromosome)):
            vertexes = self.graph_operations.get_HVSs()[i]
            j = 0
            found = False
            while (j < len(vertexes) and not found):
                iden = vertexes[j]
                intraconnection = self.graph_operations.getConnectionWithHVS(iden,self.graph_operations.get_HVSs()[i])
                interconnection = self.graph_operations.getMaxConnectionWithHVSs(iden,intraconnection)
                if interconnection[0] != -1 and interconnection[1] != 0:
                    if chromosome[i].get_value()[0] == j:
                        found = True
                        chromosome[i].set_value([chromosome[i].get_value()[0],interconnection[0]])
                        accurancy = accurancy + 1
                    else:
                        j = j + 1
                else:
                    j = j + 1
            if found == False:
                if chromosome[i].get_value()[0] == -1:
                    accurancy = accurancy + 1    
        return accurancy
    
    def get_optimal(self,position):
        vertexes = self.graph_operations.get_HVSs()[position]
        result = -1
        inter = -1
        j = 0
        found = False
        while (j < len(vertexes) and not found):
            iden = vertexes[j]
            intraconnection = self.graph_operations.getConnectionWithHVS(iden,self.graph_operations.get_HVSs()[position])
            interconnection = self.graph_operations.getMaxConnectionWithHVSs(iden,intraconnection)
            if interconnection[0] != -1 and interconnection[1] != 0:
                result = j
                inter = interconnection[0]
                found = True
            else:
                j = j + 1 
        return result,inter
    
    def calculate_fitness(self,population):
        values = []
        for i in population:
            fit = self.fitness(i)
            values.append(fit) 
        return values
    
    def completed_evolution(self,values):
        for i in values:
            if i == self.target:
                return True
        return False
    
    def get_max_values(self,values):
        best1 = 0;best2 = 0
        position1 = -1;position2 = -1
        for i in range(0,len(values)):
            if values[i] > best1:
                best2 = best1
                best1 = values[i]
                position1 = i
            elif values[i] > best2:
                best2 = values[i]
                position2 = i
        return position1,position2
    
    def get_fittest_individuals(self,population,values):
        position1,position2 = self.get_max_values(values)
        fittest1 = population[position1]
        fittest2 = population[position2]
        return fittest1,fittest2
    
    def new_individual(self,parent1,parent2):
        child = self.crossover(parent1,parent2)
        self.mutation(child)
        return child
        
    def log_individual(self,child):
        self.children.append(child)
        
    def reproduce(self,parent1,parent2):
        self.children = []
        pool = mp.Pool()
        for _ in range(0,self.size):
            pool.apply_async(self.new_individual, args = (parent1,parent2, ),callback = self.log_individual)
        pool.close()
        pool.join()   
    
    def crossover(self,parent1,parent2):
        child = []
        cross = rnd.randint(0,1)
        # El hijo es una mezcla de los progenitores.
        if self.prob_crossover > cross:
            space = int (len(parent1) * self.margin_crossover)
            margin = rnd.randint(int(space / 2),space)
            for i in range(0,margin):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                #value2 = parent1[i].get_value2()
                new_part = node_values(iden,value)
                child.append(new_part)
            for i in range(margin,len(parent2)):
                iden = parent2[i].get_iden()
                value = parent2[i].get_value()
                #value2 = parent2[i].get_value2()
                new_part = node_values(iden,value)
                child.append(new_part)
        else:
            # El hijo es una copia exacta del mejor progenitor.
            for i in range(len(parent1)):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                #value2 = parent1[i].get_value2()
                new_part = node_values(iden,value)
                child.append(new_part)
        return child
    
    def mutation(self,chromosome):
        mutate = rnd.randint(0,1)
        if self.prob_mutation > mutate:
            # El hijo presenta mutaciones en sus genes.
            margin = int(len(chromosome) * self.margin_mutation)
            for _ in range(0,margin):
                position = rnd.randint(0,len(chromosome) - 1)
                optimal,interconnection = self.get_optimal(position)
                if optimal == chromosome[position].get_value()[0]:
                    randomization = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
                    chromosome[position].set_value([randomization,-1])
                else:
                    chromosome[position].set_value([optimal,interconnection])
                  
    def get_worst(self,values):
        target = self.target
        position = -1
        for i in range(0,len(values)):
            if values[i] < target:
                target = values[i]
                position = i 
        return position
    
    def natural_selection(self,population,values):    
        for child in self.children:
            position = self.get_worst(values)
            if position != -1:
                fit = self.fitness(child)
                if fit > values[position]:
                    population.pop(position)
                    population.append(child)
                    values.pop(position)
                    values.append(fit)
    
    def evolution(self):
        completed = False
        population = self.init_population()
        fitness_values = self.calculate_fitness(population)
        while (self.counter < self.limit and not completed):                        
            if (self.completed_evolution(fitness_values)):
                completed = True
            else:
                parent1,parent2 = self.get_fittest_individuals(population,fitness_values)
                self.reproduce(parent1,parent2)
                self.natural_selection(population,fitness_values)   
                if self.counter % 10 == 0:
                    print("huuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuub")
                    print("La precision en la generacion",self.counter,"es de",max(fitness_values),"sobre",self.target)
                    result = ""
                    for i in range(0,len(parent1)):
                        result = result + parent1[i].__str__() + " "
                    print(result)
                fitness_values = self.calculate_fitness(population)
                self.counter = self.counter + 1

        parent,_ = self.get_fittest_individuals(population,fitness_values)
        return parent
    
class HVSInternalGenetic():
    
    def __init__(self,graph_operations,limit=800,size=16,margin_crossover=0.6,prob_crossover=0.9,
        margin_mutation=0.1,prob_mutation=0.4):
        rnd.seed(0)
        self.counter = 0
        self.graph_operations = graph_operations
        self.target = len(self.graph_operations.get_HVSs())
        self.limit = limit
        self.size = size
        self.margin_crossover = margin_crossover
        self.prob_crossover = prob_crossover
        self.margin_mutation = margin_mutation
        self.prob_mutation = prob_mutation
        self.children = []
        
    def init_population(self):
        population = []
        for _ in range(0,self.size):
            chromosome = self.init_chromosome()
            population.append(chromosome)
        return population
    
    def init_chromosome(self):
        chromosome = []
        for i in range(0,len(self.graph_operations.get_HVSs())):
            value = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
            relation = node_values(i,value)
            chromosome.append(relation)
        return chromosome
    
    def fitness(self,chromosome):
        accurancy = 0
        for i in range(0,len(chromosome)):
            hvs1 = self.graph_operations.get_HVSs()[i]
            j = i
            found = False
            while (j < len(chromosome) and not found):
                hvs2 = self.graph_operations.get_HVSs()[j]
                intra_sim1 = self.graph_operations.getIntraSimilarity(hvs1)
                intra_sim2 = self.graph_operations.getIntraSimilarity(hvs2)
                inter_sim = self.graph_operations.getInterSimilarity(hvs1,hvs2)
                if (inter_sim > intra_sim1 or inter_sim > intra_sim2):
                    if (chromosome[i].get_value() == j):
                        found = True
                        accurancy = accurancy + 1
                    else:
                        j = j + 1
                else:
                    j = j + 1
            if found == False:
                if chromosome[i].get_value() == -1:
                    accurancy = accurancy + 1
        return accurancy
    
    def calculate_fitness(self,population):
        values = []
        for i in population:
            fit = self.fitness(i)
            values.append(fit) 
        return values
         
    def completed_evolution(self,values):
        for i in values:
            if i == self.target:
                return True
        return False
    
    def get_max_values(self,values):
        best1 = 0;best2 = 0
        position1 = -1;position2 = -1
        for i in range(0,len(values)):
            if values[i] > best1:
                best2 = best1
                best1 = values[i]
                position1 = i
            elif values[i] > best2:
                best2 = values[i]
                position2 = i
        return position1,position2
    
    def get_fittest_individuals(self,population,values):
        position1,position2 = self.get_max_values(values)
        fittest1 = population[position1]
        fittest2 = population[position2]
        return fittest1,fittest2
    
    def new_individual(self,parent1,parent2):
        child = self.crossover(parent1,parent2)
        self.mutation(child)
        return child
        
    def log_individual(self,child):
        self.children.append(child)
        
    def reproduce(self,parent1,parent2):
        self.children = []
        pool = mp.Pool()
        for _ in range(0,self.size):
            pool.apply_async(self.new_individual, args = (parent1,parent2, ),callback = self.log_individual)
        pool.close()
        pool.join()  
    
    def crossover(self,parent1,parent2):
        child = []
        cross = rnd.randint(0,1)
        # El hijo es una mezcla de los progenitores.
        if self.prob_crossover > cross:
            space = int (len(parent1) * self.margin_crossover)
            margin = rnd.randint(int(space / 2),space)
            for i in range(0,margin):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
            for i in range(margin,len(parent2)):
                iden = parent2[i].get_iden()
                value = parent2[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
        else:
            # El hijo es una copia exacta del mejor progenitor.
            for i in range(len(parent1)):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
        return child
    
    def get_optimal(self,position):    
        result = -1
        found = False
        hvs1 = self.graph_operations.get_HVSs()[position]
        j = position
        while (j < len(self.graph_operations.get_HVSs()) and not found):
            hvs2 = self.graph_operations.get_HVSs()[j]
            intra_sim1 = self.graph_operations.getIntraSimilarity(hvs1)
            intra_sim2 = self.graph_operations.getIntraSimilarity(hvs2)
            inter_sim = self.graph_operations.getInterSimilarity(hvs1,hvs2)
            if (inter_sim > intra_sim1 or inter_sim > intra_sim2):
                result = j
                found = True
            else:
                j = j + 1
        return result
    
    def mutation(self,chromosome):
        mutate = rnd.randint(0,1)
        if self.prob_mutation > mutate:
            # El hijo presenta mutaciones en sus genes.
            margin = int(len(chromosome) * self.margin_mutation)
            for _ in range(0,margin):
                position = rnd.randint(0,len(chromosome) - 1)
                optimal = self.get_optimal(position)
                if optimal == chromosome[position].get_value():
                    randomization = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
                    chromosome[position].set_value(randomization)
                else:
                    chromosome[position].set_value(optimal)
    
    def get_worst(self,values):
        target = self.target
        position = -1
        for i in range(0,len(values)):
            if values[i] < target:
                target = values[i]
                position = i 
        return position
    
    def natural_selection(self,population,values):    
        for child in self.children:
            position = self.get_worst(values)
            if position != -1:
                fit = self.fitness(child)
                if fit > values[position]:
                    population.pop(position)
                    population.append(child)
                    values.pop(position)
                    values.append(fit)
                print("fit", fit)

    def evolution(self):
        completed = False
        population = self.init_population()
        fitness_values = self.calculate_fitness(population)
        while (self.counter < self.limit and not completed):                     
            if (self.completed_evolution(fitness_values)):
                completed = True
            else:
                parent1,parent2 = self.get_fittest_individuals(population,fitness_values)
                self.reproduce(parent1,parent2)
                self.natural_selection(population,fitness_values)   
                if self.counter % 10 == 0:
                    print("La precision en la generacion",self.counter,"es de",max(fitness_values),"sobre",self.target)
                    result = ""
                    for i in range(0,len(parent1)):
                        result = result + parent1[i].__str__() + " "
                    print(result)
                fitness_values = self.calculate_fitness(population)
                self.counter = self.counter + 1

        parent,_ = self.get_fittest_individuals(population,fitness_values)
        return parent
    
class NonHubGenetic():

    def __init__(self,graph_operations,limit=800,size=16,margin_crossover=0.6,prob_crossover=0.9,
        margin_mutation=0.1,prob_mutation=0.4,artificial_mutation = True,mutation_accurancy=0.2):
        rnd.seed(0)
        self.graph_operations = graph_operations
        self.target = len(self.graph_operations.get_non_hub_vertexes())
        self.counter = 0
        self.limit = limit
        self.size = size
        self.margin_crossover = margin_crossover
        self.prob_crossover = prob_crossover
        self.margin_mutation = margin_mutation
        self.prob_mutation = prob_mutation
        self.artificial_mutation = artificial_mutation
        self.mutation_accurancy = mutation_accurancy
        self.children = []
    
    def init_population(self):
        population = []
        for _ in range(0,self.size):
            chromosome = self.init_chromosome()
            population.append(chromosome)
        return population
    
    def init_chromosome(self):
        chromosome = []
        for node in self.graph_operations.get_non_hub_vertexes():
            value = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
            info = node_values(node,value)
            chromosome.append(info)
        return chromosome
    
    def fitness(self, chromo):
        accurancy = 0
        for value in chromo:
            position = self.graph_operations.getMoreSimilarHVS(value.get_iden())
            if position == value.get_value():
                accurancy = accurancy + 1
        return accurancy
    
    def calculate_fitness(self,population):
        values = []
        for i in population:
            fit = self.fitness(i)
            values.append(fit) 
        return values
        
    def completed_evolution(self,values):
        for i in values:
            if i == self.target:
                return True
        return False
    
    def get_max_values(self,values):
        best1 = 0;best2 = 0
        position1 = -1;position2 = -1
        for i in range(0,len(values)):
            if values[i] > best1:
                best2 = best1
                best1 = values[i]
                position1 = i
            elif values[i] > best2:
                best2 = values[i]
                position2 = i
        return position1,position2
    
    def get_fittest_individuals(self,population,values):
        position1,position2 = self.get_max_values(values)
        fittest1 = population[position1]
        fittest2 = population[position2]
        return fittest1,fittest2
    
    def new_individual(self,parent1,parent2):
        child = self.crossover(parent1,parent2)
        self.mutation(child)
        return child
        
    def log_individual(self,child):
        self.children.append(child)
        
    def reproduce(self,parent1,parent2):
        self.children = []
        pool = mp.Pool()
        for _ in range(0,self.size):
            pool.apply_async(self.new_individual, args = (parent1,parent2, ),callback = self.log_individual)
        pool.close()
        pool.join()  
        
    def get_worst(self,values):
        target = self.target
        position = -1
        for i in range(0,len(values)):
            if values[i] < target:
                target = values[i]
                position = i 
        return position

    def crossover(self,parent1,parent2):
        child = []
        cross = rnd.randint(0,1)
        # El hijo es una mezcla de los progenitores.
        if self.prob_crossover > cross:
            space = int (len(parent1) * self.margin_crossover)
            margin = rnd.randint(int(space / 2),space)
            for i in range(0,margin):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
            for i in range(margin,len(parent2)):
                iden = parent2[i].get_iden()
                value = parent2[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
        else:
            # El hijo es una copia exacta del mejor progenitor.
            for i in range(len(parent1)):
                iden = parent1[i].get_iden()
                value = parent1[i].get_value()
                new_part = node_values(iden,value)
                child.append(new_part)
        return child
    
    def mutation(self,chromosome):
        mutate = rnd.randint(0,1)
        if self.prob_mutation > mutate:
            # El hijo presenta mutaciones en sus genes.
            margin = int(len(chromosome) * self.margin_mutation)
            for _ in range(0,margin):
                position = rnd.randint(0,len(chromosome) - 1)
                iden = chromosome[position].get_iden()
                optimal = self.graph_operations.getMoreSimilarHVS(iden)
                if self.artificial_mutation == True:
                    prob = rnd.randint(0,1)
                    if self.mutation_accurancy >= prob:
                        chromosome[position].set_value(optimal)
                    else:
                        randomization = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
                        chromosome[position].set_value(randomization)
                else:
                    # Arreglo para trampear la mutacion y obtener la solucion.
                    if chromosome[position].get_value() == optimal:
                        print("TRAMPAAAAA")
                        #while True:
                        #    randomization = rnd.randint(-1,len(self.graph_operations.get_HVSs()) - 1)
                        #    if randomization != optimal:
                        #        chromosome[position].set_value(randomization)
                        #        break;
                    else:
                        print("TRAMPAAAAA2")
                    #    chromosome[position].set_value(optimal)
    
    def natural_selection(self,population,values):    
        for child in self.children:
            position = self.get_worst(values)
            if position != -1:
                fit = self.fitness(child)
                if fit > values[position]:
                    population.pop(position)
                    population.append(child)
                    values.pop(position)
                    values.append(fit)
                    
    def evolution(self):
        completed = False
        population = self.init_population()
        fitness_values = self.calculate_fitness(population)
        old_max_value = 0
        max_value = max(fitness_values)
        original_mutation_accurancy = self.mutation_accurancy
        while (self.counter < self.limit and not completed):                        
            if (self.completed_evolution(fitness_values)):
                completed = True
            else:
                parent1,parent2 = self.get_fittest_individuals(population,fitness_values)
                self.reproduce(parent1,parent2)
                self.natural_selection(population,fitness_values)   
                if self.counter % 10 == 0:
                    if self.artificial_mutation == True:
                        if old_max_value >= max_value:
                            self.mutation_accurancy = self.mutation_accurancy + 0.1
                        elif self.mutation_accurancy > original_mutation_accurancy:
                            self.mutation_accurancy = self.mutation_accurancy - 0.1
                    print("assssssssssssssssssssssssss")
                    print("La precision en la generacion",self.counter,"es de",max_value,"sobre",self.target)
                    result = ""
                    for i in range(0,len(parent1)):
                        result = result + parent1[i].__str__() + " "
                    print(result)
                fitness_values = self.calculate_fitness(population)
                old_max_value = max_value
                max_value = max(fitness_values)
                self.counter = self.counter + 1

        parent,_ = self.get_fittest_individuals(population,fitness_values)
        return parent
    
class node_values:
    
    def __init__(self,iden,value):
        self.iden = iden
        self.value = value
    
    def get_iden(self):
        return self.iden
    
    def set_iden(self,iden):
        self.iden = iden
    
    def get_value(self):
        return self.value
    
    def set_value(self,value):
        self.value = value
        
    def __str__(self):
        iden = str(self.iden)
        value = str(self.value)
        return iden + ':' + value

def square_matrix_from_vector(matrix,square):
    matrix = matrix.reshape((square,square))
    return matrix

def read_graph():
    M = []
    rows = 0
    columns = 0
    csv_filename = "./examples/matriz.csv"
    for r in open(csv_filename):
        line = r.strip().split(",")
        for j in range(0,len(line)):
            line[j] = float(line[j])
        M += [line]
        rows = rows + 1
        columns = len(line)
    square = int(sqrt(columns))
    matrix = np.matrix(M)
    b = abs(square) - abs(int(square))
    if rows != columns:
        if b == 0:
            matrix = square_matrix_from_vector(matrix,square)
        else:
            raise Exception('Matrix cannot be converted to square.')                    
    return nx.from_numpy_matrix(matrix)

def main():
    G = read_graph()
    print (nx.info(G))
    genetic_graph(G)
    
if __name__ == '__main__':
    main()