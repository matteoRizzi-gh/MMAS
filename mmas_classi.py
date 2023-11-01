import random as rd
import math
from geopy.distance import geodesic
import matplotlib.pyplot as plt

class Graph:
    
    #definizione del costruttore 
    def __init__(self, city_num):
        self.city_num = city_num
        self.coordinates = []  # Lista per le coordinate delle città
        
        #creazione matrice delle distanze
        self.distance_matrix=[[0]*city_num for _ in range(city_num)]

    #aggiungere un nodo ad un grafo, si Assume che TSP sia simmetrico ovvero lenght(A,B) = lenght(B,A)
    #USARE SE SI CREA UN GRAFO DA 0
    def new_node(self, city_1, city_2, distance):
        self.distance_matrix[city_1][city_2] = distance
        self.distance_matrix[city_2][city_1] = distance
     
   #USARE SE SI HA UN ISTANZA IMPORTATA
    def add_node(self, city, x, y):
        # Aggiungi le coordinate della città alla lista
        self.coordinates.append((x, y))

 
    def euclidean_distance_matrix(self):
        # Calcola le distanze euclidee tra tutte le coppie di città
        for i in range(self.city_num):
            for j in range(i + 1, self.city_num):
                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance  # Poiché il TSP è simmetrico


# Calcola la distanza geografica tra due città in radianti
    def geo_distance(self, city_1, city_2):
        # Coordinate geografiche delle città
        lat1, lon1 = self.coordinates[city_1]
        lat2, lon2 = self.coordinates[city_2]

        # Calcola la distanza utilizzando la formula di Geodesic
        distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers

        return distance


 

    def geo_distance_matrix(self):
        for i in range(self.city_num):
            for j in range(i + 1, self.city_num):
                # Calcola la distanza geografica tra le città i e j utilizzando la formula di Geodesic
                distance = self.geo_distance(i, j)
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance  # Poiché il TSP è simmetrico



    # Calcola la distanza totale di un percorso utilizzando la distanza geografica
    def get_geographical_distance(self, trail):
        length = 0
        for i in range(len(trail) - 1):
            
            current_city = trail[i]
            next_city = trail[i + 1]
            distance = self.geo_distance(current_city, next_city)
            length += distance
            print(i, " ", trail[i]+1, " ",trail[i+1]+1," ", length)
        # distanza geografica tra l'ultimo e il primo punto per chiudere il ciclo
        length += self.geo_distance(trail[-1], trail[0])

        return length


      
    # totale del percorso compiuto  
    def get_distance(self, trail):
        lenght=0
        for i in range(len(trail)-1):
            current_city=trail[i] 
            succ_city = trail[i+1]
            
            #sommo le distanza tra tutti gli archi del percorso
            lenght += self.distance_matrix[current_city][succ_city]  
        #manca solo l'arco che collega l'ultimo vertice del percorso con quello di partenza
        lenght +=self.distance_matrix[trail[-1]][trail[0]] 
        return lenght



class Ants:
    #costruttore per le formiche arificiali
    def __init__(self, distance_matrix, alpha, beta ):
        self.distance_matrix = distance_matrix
        self.alpha = alpha  #parametro relativo dei feromoni
        self.beta = beta  #parametro relativo informazioni euristiche (visibilità)
        self.tour = []  # Tour compiuto dalla formica



    #probabilità della formica di passare attraverso l'arco (u,v)
    def edge_prob(self, u, v, pheromone): #funzione 3
        
        tau_uv = pheromone[u][v]
        eta_uv = 1.0 / self.distance_matrix[u][v] #visibilità

        denominator = 0.0
        for w in range(len(pheromone[u])):
            if w != u:
                #tutti gli archi collegati w con u
                tau_uw = pheromone[u][w]
                #visibilità delle città u,w
                eta_uw = 1.0 / self.distance_matrix[u][w]
                denominator += (tau_uw ** self.alpha) * (eta_uw ** self.beta)
        prob = (tau_uv**self.alpha)*(eta_uv**self.beta) / denominator
        return prob

 

    #la città è scelta rispetto al vettore di probabilità in modo "casuale", 
    #non la migliore perché si vuole esplorare lo spazio delle soluzioni in 
    #maniera efficiente prima di ricavare l'effettiva soluzione ottimale
    def city_choice(self, prob):
       #calcolo della probabilità totale tra le città e i vertici
        total_prob = sum(p[1] for p in prob)
        choice = rd.uniform(0, total_prob) #generato casualmenet per scegliere la città successiva
        prob.sort(key=lambda x: x[1]) #prob contiene le coppie città, probablità e viene ordinaro rispetto alla probabilità
        cumulative_prob = 0 #accumulatore
        #scorro il vettore prob e sommo le probabilità, quando la somma è maggiore di choice la formica sceglie la città
        #corrispoondente alla città a cui siamo arrivati durante questa iterazione
        for city, probability in prob:
            cumulative_prob += probability
            if cumulative_prob >= choice:
                next_city = city
                return next_city



    #costruzione di un percorso casuale delle formiche
    def construct_tour(self, pheromone):
        
        city_num = len(pheromone)
        
        #la prima città, ovvero quella da cui parte la formica, è scelta casualmente
        start_city = rd.randint(0, city_num -1)
        self.tour = [start_city] #inizializzazione del vettore che mantiene la successione delle citt' visitate dalla formica
        
        while len(self.tour) < city_num:
            current_city = self.tour[-1]
            #calcolo della probabilità per le città non visitate
            prob = []
            for city in range(city_num):
                if city not in self.tour:
                    probability = self.edge_prob(current_city, city, pheromone)
                    prob.append((city, probability))
            
            #scelta della prossima città rispetto al vettore probabilità
            next_city = self.city_choice(prob)
            self.tour.append(next_city)





    #calcolo della lunghezza del percorso compiuto dalla formica
    def tour_length(self):
        lenght=0
        for i in range(len(self.tour)-1):
            current_city=self.tour[i]
            succ_city = self.tour[i+1]
            
            #sommo le distanza tra tutti gli archi del percorso
            lenght += self.distance_matrix[current_city][succ_city]  
        #manca solo l'arco che collega l'ultimo vertice del percorso con quello di partenza
        lenght += self.distance_matrix[self.tour[-1]][self.tour[0]] 
        return lenght



#Oss. posso togliere max_pheromone dall'istanza del costruttore ed imporlo uguale ad 1.0

class MMAS:
    def __init__(self, city_num, alpha, beta, rho, q0, distance_matrix, max_iteration,max_trail, min_trail, max_pheromone, stagnation_threshold, num_ants):
        self.city_num = city_num
        self.alpha = alpha  # peso dato ai feromoni
        self.beta = beta  # peso dato alla visibilità
        self.rho = rho  # evaporation rate
        self.q0 = q0  # probabilità di scegliere la strada rispetto al valore di feromone (se vicino a 0) o in base alla distanza(se vicino a 1)
        self.max_iteration = max_iteration  # Numero di iterazioni
        self.max_trail = max_trail  # tau_max sugli archi
        self.min_trail = min_trail  # tau_min sigli archi
        self.distance_matrix = distance_matrix
        self.stagnation_threshold=stagnation_threshold #limite per verificare stagnazione
        self.max_pheromone = max_pheromone #feromone iniziale
        self.num_ants=num_ants;
        self.pheromone=[[max_pheromone]*city_num for _ in range(city_num)]
        #self.pheromone_delta = [[0.0] * city_num for _ in range(city_num)]  #per approccio global-best, devo salvare tutte le informazioni di tutte le formiche per ogni iterazione
        self.convergence_data = []  #lista vuota per i dati di convergenza

    
    #probabilità della formica di passare attraverso l'arco (u,v)
    def edge_prob(self, u, v): #funzione 3
        
        tau_uv = self.pheromone[u][v]
        eta_uv = 1.0 / self.distance_matrix[u][v] #visibilità

        denominator = 0.0
        for w in range(len(self.pheromone[u])):
            if w != u:
                #tutti gli archi collegati w con u
                tau_uw = self.pheromone[u][w]
                #visibilità delle città u,w
                eta_uw = 1.0 / self.distance_matrix[u][w]
                denominator += (tau_uw ** self.alpha) * (eta_uw ** self.beta)
        prob = (tau_uv**self.alpha)*(eta_uv**self.beta) / denominator
        return prob
        


    #definizione del percorso costruito dalle formiche
    def construction_tour(self, ant):
        city_num = self.city_num
        start_city = rd.randint(0, city_num - 1) #città iniziale casuale
        unvisited_cities = set(range(city_num)) #vittà ancora da visitare
        unvisited_cities.remove(start_city) #dato che la formica si trova sulla prima città, posso rimuoverla da wuelle non visitate
        current_city = start_city
        ant.tour = [current_city]
        

        #itero finché ci sono città non visitate
        while unvisited_cities:
            #calcolo del vettore delle probabilità di attraversare gli archi collegati all'attuale città
            prob=[]
            for city in unvisited_cities:
                probability = self.edge_prob(current_city, city)
                prob.append((city, probability))

             # Introduzione di una perturbazione casuale che faccia saltare una formica ad una città casuale, così da favorire l'esplorazione
            if rd.random() < 0.1:  #10% di probabilità di perturbazione
                non_visited = [city for city in unvisited_cities]
                next_city = rd.choice(non_visited)
            else:
                #se un valore casuale è inferiore a q0, allora la formica si comporta deterministicamente
                #e sceglie la città successiva rispetto alla massima probabilità tra gli archi
                 if rd.random() < self.q0:
                    massima = max(prob, key=lambda x: x[1])
                    next_city = [city for city, probability in prob if probability == massima[1]][0]

            #altrimenti vi è un esplorazione stocastica dei vertici
                 else:
                    #calcolo della probabilità totale tra le città e i vertici
                    total_prob = sum(p[1] for p in prob)

                    choice = rd.uniform(0, total_prob) #generato casualmenet per scegliere la città successiva
                    prob.sort(key=lambda x: x[1])#prob contiene le coppie città, probablità e viene ordinaro rispetto alla probabilità
                    cumulative_prob = 0 #accumulatore
                    #scorro il vettore prob e sommo le probabilità, quando la somma è maggiore di choice la formica sceglie la città
                    #corrispoondente alla città a cui siamo arrivati durante questa iterazione e uso break per uscire dal cilco
                    for city, probability in prob:
                        cumulative_prob += probability
                        if cumulative_prob >= choice:
                            next_city = city
                            break
            #aggiorno i vettori tour, città ancora da visitare e la città corrente
            ant.tour.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city


    def update_pheromone(self,  best_tour_length, best_tour):
        delta_best = 1.0 / best_tour_length
        current_tour=best_tour.copy();
        for _ in range(self.city_num-1):
            i=current_tour[0]
            for j in range(self.city_num): 
            #delta_best viene sommato solo agli archi del percorso migliore
                if j==current_tour[1]:
                    delta_pheromone = delta_best
                else:
                    delta_pheromone=0.0
                
                self.pheromone[i][j]=(1-self.rho)*self.pheromone[i][j] + delta_pheromone
                self.pheromone[i][j] = max(self.min_trail, min(self.max_trail, self.pheromone[i][j]))
                self.pheromone[j][i] = max(self.min_trail, min(self.max_trail, self.pheromone[i][j]))
            current_tour.remove(i)
        self.pheromone[-1][0]=(1-self.rho)*self.pheromone[-1][0] + delta_pheromone
        self.pheromone[0][-1] = max(self.min_trail, min(self.max_trail, self.pheromone[-1][0]))
        self.pheromone[-1][0] = max(self.min_trail, min(self.max_trail, self.pheromone[-1][0]))
                    
                   
                
                


    #calcolo della lunghezza del percorso compiuto dalla formica
    def tour_length(self, tour):
        length=0
        for i in range(len(tour)-1):
            current_city=tour[i]
            succ_city = tour[i+1]
            
            #sommo le distanza tra tutti gli archi del percorso
            length += self.distance_matrix[current_city][succ_city]  
        #manca solo l'arco che collega l'ultimo vertice del percorso con quello di partenza
        length +=self.distance_matrix[tour[-1]][tour[0]] 
        return length
    

    def calculate_branching_factor(self, ant):
        city_num = self.city_num
        total_branching_factor = 0.0
        for i in range(city_num):
            current_city = ant.tour[i]
            for j in range(city_num):
                if i != j:
                    next_city = ant.tour[j]
                    prob_ij = self.edge_prob(current_city, next_city)
                    total_branching_factor += prob_ij
                    
        return total_branching_factor / (city_num * (city_num - 1))
    

    def smooth_pheromone(self, branching_factor):
        
        #print(branching_factor)

        if branching_factor < self.stagnation_threshold:
            # Calcola la differenza tra trail_max e i valori correnti dei feromoni e aggiorna proporzionalmente
            for i in range(self.city_num):
                for j in range(self.city_num):
                    if i != j:
                        delta_pheromone = self.max_trail - self.pheromone[i][j]
                        self.pheromone[i][j] += delta_pheromone
                        # I valori dei feromoni devono entro i limiti min_trail e max_trail
                        self.pheromone[i][j] = max(self.min_trail, min(self.max_trail, self.pheromone[i][j]))


     #aggiornamento dati di convergenza
    def update_convergence_data(self, tour_length):
        self.convergence_data.append(tour_length)
        

    #risoluzione effettiva del problema
    def solve(self):
        best_tour = None
        best_tour_length = float('inf')

        for iteration in range(self.max_iteration):
            ants = [Ants(self.distance_matrix, self.alpha, self.beta) for _ in range(self.num_ants)]
            for ant in ants:
                # costruzione del percorso
                self.construction_tour(ant)

                # calcolo della lunghezza del percorso
                tour_length = self.tour_length(ant.tour)

                # aggiornamento del percorso più breve se viene trovato (iteration-best)
                if tour_length < best_tour_length:
                    best_tour_length = tour_length
                    best_tour = ant.tour[:]
                
            #verifica di stagnamento
            if iteration % 10==0:
                branching_factor = self.calculate_branching_factor(ant)
                if branching_factor < self.stagnation_threshold:
                    self.smooth_pheromone(branching_factor)
            # Aggiornamenot livelli di feromoni
            self.update_pheromone(best_tour_length, best_tour)
            
            #print("iterazione ", iteration)
            #print("Best Tour Length:", best_tour_length)
            #print("Best Tour:", best_tour)
            
            self.update_convergence_data(best_tour_length)
        #for row in self.pheromone:
         #      print(row)
          #     print(" ")
        return best_tour_length, best_tour
    
    def plot_convergence(self, convergence_data):
       iterations = list(range(1, len(convergence_data) + 1))
       plt.plot(iterations, convergence_data, marker='o', linestyle='-', color='b')
       plt.xlabel('Iteration')
       plt.ylabel('Best Tour Length')
       plt.title('Convergence Analysis')
       plt.grid(True)
       plt.show()
