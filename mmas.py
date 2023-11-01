from mmas_classi import Graph, Ants, MMAS
import random as rd
import time
import math
import matplotlib.pyplot as plt
start_time = time.time()

with open('ulysses22.tsp', 'r') as file:
    lines = file.readlines()

coordinates = {}
read_coordinates = False

for line in lines:
    line = line.strip()
    if line == "NODE_COORD_SECTION":
        read_coordinates = True
    elif line == "EOF":
        read_coordinates = False
    elif read_coordinates:
        parts = line.split()
        city_number = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        coordinates[city_number] = (x, y)


city_num = len(coordinates)
my_graph = Graph(city_num)

for city, (x, y) in coordinates.items():
    my_graph.add_node(city, x, y)
    
for i in coordinates:
    print(i, " ", coordinates[i])
my_graph.geo_distance_matrix()

tour=[0, 13,12,11,6,5, 14,4,10,8,9,18,19, 20,15,2,1,16,21,3,17,7, 0]



distance_matrix = my_graph.distance_matrix

print(my_graph.get_geographical_distance(tour))



#for j in range(i, city_count):
     #  if i==j:
    #        graph.new_node(i,j, 0.0)
   #    else:
  #          number= rd.uniform(1, 100)
 #           graph.new_node(i,j, number)


#for i in range(city_count):
 # for j in range(i + 1, city_count):
  #   graph.new_node(i, j, distanze[i][j])




# for row in distance_matrix:
  # print(row)



#mi serve un istanza ant per costruire il primo tour possibile
#ants_1 = Ants(distance_matrix, alpha=1.0, beta=2.0)
#pheromone_matrix = [[1.0] * city_num for _ in range(city_num)]


#ants_1.construct_tour(pheromone_matrix)
#tour_lenght= ants_1.tour_length()


#print("Tour Length: ", tour_lenght)
  
all_tour=[]
x_vector=[]

for i in range(0, 10):

    #a partire dal primo tour possibile creato dalla classe ant, costruisco iterativamente i tour 
    #rispetto all'iteration-best

    mmas_solver=MMAS(
        city_num=city_num, alpha=1.0, beta=1.0, rho=0.1, q0=0.6, distance_matrix=distance_matrix, 
        max_iteration=750, max_trail=1.0, min_trail=0.01, max_pheromone =1.0, stagnation_threshold= 0.5, num_ants=22
        )

    best_tour_length, best_tour = mmas_solver.solve()
    print("Best Tour Length:", best_tour_length)
    print("Best Tour:", best_tour)
    #mmas_solver.plot_convergence(mmas_solver.convergence_data)
    print(" ")
    all_tour.append(best_tour_length)
    x_vector.append(i)
    


end_time = time.time()
execution_time = end_time - start_time

print("Tempo impiegato:", execution_time, "secondi")

plt.plot(x_vector , all_tour , marker='o', linestyle='-')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Tour Path')
plt.show()
