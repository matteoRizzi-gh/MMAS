from mmas_classi import Graph, Ants, MMAS
import random as rd
import time
import math
import matplotlib.pyplot as plt
start_time = time.time()

# Definisci il nome del file
nome_file = "swiss42.tsp"

# Inizializza una lista vuota per la matrice delle distanze
distance_matrix = []

with open(nome_file, 'r') as file:
    leggi = False
    # Itera sulle linee del file
    for linea in file:
        # Rileva la linea "EDGE_WEIGHT_SECTION" per iniziare a leggere i dati
        if "EDGE_WEIGHT_SECTION" in linea:
            leggi = True
            continue
        elif "EOF" in linea:
            leggi = False    
        
        elif leggi:
            valori = linea.strip().split()
            # Converti i valori in numeri interi e li aggiungi alla matrice
            valori_numerici = [int(valore) for valore in valori]
            distance_matrix.append(valori_numerici)



#with open('italy.tsp', 'r') as file:
#    lines = file.readlines()

#coordinates = {}
#read_coordinates = False

#for line in lines:
 #   line = line.strip()
  #  if line == "NODE_COORD_SECTION":
   #     read_coordinates = True
    #elif line == "EOF":
    #    read_coordinates = False
    #elif read_coordinates:
     #   parts = line.split()
      #  city_number = int(parts[0])
       # x = float(parts[1])
      #  y = float(parts[2])
       # coordinates[city_number] = (x, y)


#city_num = len(coordinates)
city_num=42
my_graph = Graph(city_num)

#for city, (x, y) in coordinates.items():
#    my_graph.add_node(city, x, y)
    
#for i in coordinates:
#    print(i, " ", coordinates[i])
#my_graph.euclidean_distance_matrix() 


#tour=[0, 13,12,11,6,5, 14,4,10,8,9,18,19, 20,15,2,1,16,21,3,17,7, 0]
#tour=[4, 14, 5, 6, 11, 12, 13, 0, 7, 21, 3, 17, 16, 1, 2, 15, 20, 19, 18, 9, 8, 10]
#tour=[1, 16, 3, 17, 21, 7, 0, 13, 12, 11, 6, 5, 14, 4, 10, 8, 9, 18, 19, 20, 15, 2]
#tour=[0,13,12,11,6,5,14,4,10,8,9,15,2,1,3,7]
#tour=[0,21,7,25,30,27,2,35,34,19,1,28,20,15,49,33,29,8,48,9,38,32,44,14,43,41,39,18,40,12,24,13,23,42,6,22,47,5,26,50,45,11,46,17,3,16,36,4,37,10,31]
#tour=[5,2,0,1,3,4,7,10,9,13,17,14,16,15,11,12,8,6]

my_graph.distance_matrix = distance_matrix


#print(my_graph.get_geographical_distance(tour))



#for j in range(i, city_count):
     #  if i==j:
    #        graph.new_node(i,j, 0.0)
   #    else:
  #          number= rd.uniform(1, 100)
 #           graph.new_node(i,j, number)


#for i in range(city_count):
 # for j in range(i + 1, city_count):
  #   graph.new_node(i, j, distanze[i][j])



for row in distance_matrix:
    print(row)




  
all_tour=[]
x_vector=[]

for i in range(0, 2):

    #a partire dal primo tour possibile creato dalla classe ant, costruisco iterativamente i tour 
    #rispetto all'iteration-best
    param_combinations = [
    #(1.0, 2.0, 0.97, 0.5, 30),  # Combinazione 1
    #(1.0, 3.0, 0.95, 0.6, 50),   # Combinazione 2
    #(1.0, 4.0, 0.99, 0.4, 40),  # Combinazione 3
    (1.0, 5.0, 0.98, 0.4, 60),  # Combinazione 4
    #(1.0, 5.0, 0.98, 0.5, 60),
    #(1.0, 5.0, 0.98, 0.6, 60),
    #(1.0, 2.5, 0.96, 0.55, 35)  # Combinazione 5
]

    for params in param_combinations:
        alpha, beta, rho, q0, num_ants = params
    
        mmas_solver = MMAS(
        city_num=city_num, alpha=alpha, beta=beta, rho=rho, q0=q0, distance_matrix=distance_matrix, 
            max_iteration=1000, max_trail=1.0, min_trail=0.05, max_pheromone=1.0, stagnation_threshold=0.5, num_ants=num_ants
        )

        best_tour_length, best_tour = mmas_solver.solve()
        print("Parameters: alpha={}, beta={}, rho={}, q0={}, num_ants={}".format(alpha, beta, rho, q0, num_ants))
        print("Best Tour Length:", best_tour_length)
        print("Best Tour:", best_tour)
        #mmas_solver.plot_convergence(mmas_solver.convergence_data)
        print(" ")
    all_tour.append(best_tour_length)
    x_vector.append(i)
    


end_time = time.time()
execution_time = end_time - start_time

print("Tempo impiegato:", execution_time, "secondi")

