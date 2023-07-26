import numpy as np
import networkx as nx
import sys 
import pygraphviz as pgv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.colors
import pandas as pd 
plt.style.use('ggplot')

T1_2_0=716
T1_3_0=440
T1_4_0=358
T3_7_0=143
T3_6_0=199
T2_3_0=11
T2_5_0=380
T4_8_0= 284
T5_9_0,T5_10_0 =71, 96
T8_11_0=205
T6_10_0=32
T11_12_0=110
T10_12_0 =50
T7_12_0= 56
T9_13_0 =32
T12_13_0= 47

asc_=[]
Dc_=[]
Redundancy_=[]
a_=[]
C_link_density = []
n_roles = []
T_overall_list = []
matrix_list = []
Z_axis=[]
T_3_6 = []
a_log_a = []


initial_matrix = np.array([[0., T1_2_0, T1_3_0, T1_4_0, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., T2_3_0, 0., T2_5_0, 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., T3_6_0, T3_7_0, 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., T4_8_0, 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., T5_9_0, T5_10_0, 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., T6_10_0, 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T7_12_0, 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T8_11_0, 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T9_13_0],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T10_12_0, 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T11_12_0, 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., T12_13_0],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
Di = nx.DiGraph()
T = np.round(initial_matrix, decimals=1)
for i, row in enumerate(T, start=1):
    row_result = []
    for j, value in enumerate(row, start=1):
        if value != 0:
            row_result.append([i, j, value])
            Di.add_edge(i, j, weight=value)

for i in range(1, len(T)+1):
    for j in range(1, len(T)+1):
        if i ==len(T):
            break
        if i != j and i==1:
            try:
                shortest_path = nx.shortest_path(Di, i, j)
                shortest_path_weights = nx.get_edge_attributes(Di, 'weight')
                shortest_path_weights = [shortest_path_weights[(shortest_path[l], shortest_path[l+1])] for l in range(len(shortest_path)-1)]

                all_pathways = nx.all_simple_paths(Di, i, j)
                allpath = [path for path in all_pathways]
                all_path_weights = nx.get_edge_attributes(Di, 'weight')
                
                for x in allpath:
                    all_path_weight = []
                    for l in range(len(x)-1):
                        edge = (x[l], x[l+1])
                        if edge in all_path_weights:
                            weight = all_path_weights[edge]
                            all_path_weight.append(weight)
                        else:
                            pass
                        
                    #in order to design a redistribution rule we have all the possble pathways listed 
#                     print(f'possible pathways from {i} to {j} and weight:', x, all_path_weight)
                    pass

            except Exception as e:
                print(e)
#unused
nested_list_rules = [[[1, 2], 325], [[1, 2, 3], 11], [[1, 2, 3, 6], 0], [[1, 2, 3, 6, 10], 0], [[1, 2, 3, 6, 10, 12], 0], [[1, 2, 3, 6, 10, 12, 13], 0], [[1, 2, 3, 7], 0], [[1, 2, 3, 7, 12], 0], [[1, 2, 3, 7, 12, 13], 0], [[1, 2, 5], 213], [[1, 2, 5, 9], 39], [[1, 2, 5, 9, 13], 32], [[1, 2, 5, 10], 78], [[1, 2, 5, 10, 12], 18], [[1, 2, 5, 10, 12, 13], 0], [[1, 3], 98], [[1, 3, 6], 167], [[1, 3, 6, 10], 0], [[1, 3, 6, 10, 12], 32], [[1, 3, 6, 10, 12, 13], 0], [[1, 3, 7], 87], [[1, 3, 7, 12], 56], [[1, 3, 7, 12, 13], 0], [[1, 4], 74], [[1, 4, 8], 79], [[1, 4, 8, 11], 95], [[1, 4, 8, 11, 12], 63], [[1, 4, 8, 11, 12, 13], 47]]
#rule used 
rule = {(1, 2): 325, (1, 2, 3): 11, (1, 2, 3, 6): 0, (1, 2, 3, 6, 10): 0, (1, 2, 3, 6, 10, 12): 0, (1, 2, 3, 6, 10, 12, 13): 0, (1, 2, 3, 7): 0, (1, 2, 3, 7, 12): 0, (1, 2, 3, 7, 12, 13): 0, (1, 2, 5): 213, (1, 2, 5, 9): 39, (1, 2, 5, 9, 13): 32, (1, 2, 5, 10): 78, (1, 2, 5, 10, 12): 18, (1, 2, 5, 10, 12, 13): 0, (1, 3): 98, (1, 3, 6): 167, (1, 3, 6, 10): 0, (1, 3, 6, 10, 12): 32, (1, 3, 6, 10, 12, 13): 0, (1, 3, 7): 87, (1, 3, 7, 12): 56, (1, 3, 7, 12, 13): 0, (1, 4): 74, (1, 4, 8): 79, (1, 4, 8, 11): 95, (1, 4, 8, 11, 12): 63, (1, 4, 8, 11, 12, 13): 47}



pathways = []
flow_values = []
def path_to_reduce(start , end):
    if (start , end) not in Di.edges: 
        print(' There is no connection between these two points in the network')
        sys.exit()
    else: 
        print(Di.edges)
        for key, value in (sorted(rule.items())):
            if start in key and end in key and value!=0 and list(key).index(start)==list(key).index(end)-1:
                
                pathways.append({key: value})
        
        print('paths involved')
        print(pathways)
        start_end = start,end  
        return start_end 

    
def redistribution(start_end):
    distribution_pathways = []
    for count, x in enumerate(pathways):
        for key_b, value_b in x.items():
             for key, value in rule.items():
                if key[0]==key_b[0] and key[-1]==key_b[-1]:

                    distribution_pathways.append({key:value})
    print('paths used for/to redistribute')
    print(distribution_pathways)
    
    # Grouping dictionaries based on similar start and end points and removing any connection associated with pathway(s) to be reduced   
    data = []
    for dict1 in distribution_pathways:
        unique = True
        for dict2 in pathways:
            if dict1.keys() == dict2.keys():
                unique = False
                break
        if unique:
            data.append(dict1)
#     print(data)

    groups = {}
    for item in data:
        key = (list(item.keys())[0][0], list(item.keys())[0][-1])
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    # Finding shortest key within each group and obtain all other dictionaries with similar nodes leading to the shortest paths
    shortest_keys = []
    other_dict = []
    for group_items in groups.values():
        key_numbers = []
        shortest_key = min(group_items, key=lambda x: len(list(x.keys())[0]))
        key_tuple = next(iter(shortest_key))
        shortest_keys.append(shortest_key)
#         print(shortest_keys)
        num = (key_tuple[1:-1])
        for x in num:
            for key, value in rule.items():
                if key[-1]==x :
                    all_items_contained = all(item in key_tuple for item in key)
                    if all_items_contained:
                        other_dict.append({key: value})
                        
#     print(other_dict)
    final_matrices = []
    update_to_rule = []
    for item in shortest_keys:
        key = list(item.keys())[0]
        value = item[key]
        
        # Update value in data based on items_to_remove
        start = key[0]
        end = key[-1]
        for remove_item in pathways:
            remove_key = list(remove_item.keys())[0]
            remove_value = remove_item[remove_key]
            #accessing items with identical last key index and increasing/reducing all paths associated
            if remove_key[-1] == end:
                for removed_value in range(remove_value, -1,-1):
                    if removed_value > 0:
                        x = remove_value-removed_value
                        y= 0
                        y+=1+x
                        z = -x+y
                        item[key] += z 
                    
                    for items in shortest_keys:
                        update_to_rule.append({remove_key:removed_value})
                        update_to_rule.append(items)

                    for dict2 in update_to_rule :
                        for key2, value2 in dict2.items():
                            for dict1 in [rule]:
                                for key1 in dict1.keys():
                                    if key1 == key2:
                                        dict1[key1] = value2
                    # Print the updated data
                    print("final Updated data:")
                    for single_rule in rule:
                        pass
                        print(single_rule, rule[single_rule])
                    
                    final_matrix = np.zeros_like(initial_matrix) 
                    print(rule)
                    for k, v  in rule.items():
                        l = list(k)
                        while len(l) !=1:
                            a,b = l[0:2]
                            final_matrix[a-1,b-1] += v
                            del l[0]

                    final_matrices.append(final_matrix)
                    print('first')
#                     print(x,y)
                    print(final_matrix)
                    
                            
    for count, final_matrix in enumerate(final_matrices):
        Di = nx.DiGraph()
        edge_list = []
        for i, row in enumerate(final_matrix, start=1):
            row_result = []
            for j, value in enumerate(row, start=1):
                if value != 0:
                    row_result.append([i, j, value])
                    Di.add_edge(i, j, weight=value)

            if row_result:
                edge_list.append(row_result)        
        edge_ = [inner_list for outer_list in edge_list for inner_list in outer_list]
        # create a graph object
        G = pgv.AGraph(directed=True)

         # add nodes to the graph
        G.add_nodes_from(range(1, len(final_matrix)))

        edges= sorted(edge_, key=lambda x: x[0])
        for edge in edges:
            G.add_edge(edge[0:2])

        # specify layout for the graph
        pos = G.layout("dot")

        # add node labels and edge weights
        for node in G.nodes():
            node.attr["label"] = str(node)

        for item1 in G.edges():
            for item2 in edge_:
                if int(item1[0]) == item2[0] and int(item1[1]) == item2[1]:
                    item1.attr["label"] = item2[2]

        # show the graph
        G.draw(f"OneDrive/Desktop/constrained_network/network_graph_{start_end}_{count}.png", prog="dot")
                       
        T = final_matrix
        rows, cols = T.shape

        # Compute the row and column sums
        row_sums = T.sum(axis=1)
        col_sums = T.sum(axis=0)
       
        
        T_sum = T.sum()

        # Compute the ascendancy values
        ascendancy_list = [[T[i][j] * (math.log(T[i][j]) + math.log(T_sum) - math.log(row_sums[i]) - math.log(col_sums[j])) if T[i][j] > 0 else 0 for j in range(cols)] for i in range(rows)]
        ascendancy = sum(map(lambda x: sum(x), ascendancy_list))
        
        
        # Compute the redundancy values
        redundancy_list = [[T[i][j] * (math.log(T[i][j]) + math.log(T[i][j]) - math.log(row_sums[i]) - math.log(col_sums[j])) if T[i][j] > 0 else 0 for j in range(cols)] for i in range(rows)]
        redundancy = -1*sum(map(lambda x: sum(x), redundancy_list))
       
        
        # Compute the Development_capacity values
        Development_capacity_list = [[T[i][j] * (math.log(T[i][j]) - math.log(T_sum)) if T[i][j] > 0 else 0 for j in range(cols)] for i in range(rows)]
        Development_capacity = -1*sum(map(lambda x: sum(x), Development_capacity_list))

        X = (ascendancy/T_sum)             
        psi = (redundancy/T_sum )
        Dc = ascendancy + redundancy 
        a = ascendancy/Dc
        e = math.e
        c = e**(psi/2)
        n = e**X

        # visualisation of results
#         print( '\n total_Ascendancy (A):', ascendancy, 
#           '\n total_Redundancy', '(' + (chr(966)) + '):', redundancy ,
#           '\n total_Capacity (C):', Development_capacity,
#           '\n X:', (X) ,
#           '\n psi:', '(' + (chr(968)) + '):', psi,
#           '\n sum of Ascendancy and Redundancy: ', Dc,
#           '\n a (A/C) : ', a,
#           '\n Connectivity:',  c,
#           '\n number of roles:', n)

        asc_.append(ascendancy)
        Dc_.append(Dc)
        Redundancy_.append(redundancy)
        a_.append(a)
        Z_axis.append(count/len(final_matrices))
        C_link_density.append(c)
        n_roles.append(n)
        T_overall_list.append(T_sum)
        data = list(zip(C_link_density, n_roles,T_overall_list,asc_,Redundancy_,Dc_,a_))
        

        AC_vals = np.linspace(0, 1, 100)
        C_vals = np.nan_to_num(-AC_vals*np.log(AC_vals) , nan=0) 

        # Find the maximum robustness value and corresponding AC ratio
        log_vals = np.nan_to_num(-a*np.log(a) , nan=0) 
        a_log_a.append(log_vals)

        # Define parameter values
        Cmin = 0
        Cmax = 1
 
        # create a DataFrame from the list of tuples
        df = pd.DataFrame(data, columns=[ 'Link Density','Number of roles', 'T..', 'Ascendancy', 'Redundancy','Capacity','a' ])
#     print(df)
     
     # Create the plot A
    max_C = max(a_)
    max_AC= max(C_vals)
    fig, ax = plt.subplots()
    ax.plot(AC_vals, C_vals, 'b-')
    ax.plot(a_, a_log_a, 'ro')

    # Set axis limits
    ax.set_xlim([Cmin, Cmax])
    ax.set_ylim([0, max_C + 0.1])  # Updated to extend plot higher on the y-axis
    
    # Add labels and title
    ax.set_xlabel('a (Ascendancy/Development Capacity)')
    ax.set_ylabel(' (Robustness) [-a*log(a)]')
    ax.set_title('Robustness Curve')
#     ax.axhline(y=max_AC, linestyle='--', color='green')
#     ax.axvline(x=max_C, linestyle='--', color='gray')
    plt.savefig(f'OneDrive/Desktop/transport_curve_decomposing_pathways/Robustness_Curve_{count}_{start_end}.png')


      # Create the plot B
    max_C = max(a_)
    max_AC= max(C_vals)
    fig, ax = plt.subplots()
    
    ax.plot(Z_axis,a_, 'ro')

    # Set axis limits
    ax.set_xlim([Cmin, Cmax])
    ax.set_ylim([0, max_C + 0.1])  # Updated to extend plot higher on the y-axis
    
    # Add labels and title
    ax.set_ylabel('a (Ascendancy/Development Capacity)')
    ax.set_xlabel(f'{start_end}')
    ax.set_title('Degree of Order')
    plt.savefig(f'OneDrive/Desktop/transport_curve_decomposing_pathways/_Curve_a_T36_{count}_{start_end}.png')
    

    # 2D plots of roles vs connectivity 
    fig, ax = plt.subplots()
#     ax.plot( n_roles_E,C_link_density_E, 'bo')
    plt.scatter(n_roles,C_link_density,c=Z_axis, cmap='YlOrRd')
    plt.colorbar()
    ax.set_xlabel('Number of roles')
    ax.set_ylabel('Link Density')
    ax.set_title('Transport Curve')
    plt.savefig(f'OneDrive/Desktop/transport_curve_decomposing_pathways/_plot_roles_connectivity_2D_{count}_{start_end}.png')
    
     # 3D plots of roles vs connectivity 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter =ax.scatter(n_roles,C_link_density, Z_axis,c = Z_axis, cmap='PuBuGn')
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel('number of roles')
    ax.set_ylabel('link_density')
    ax.set_zlabel(f'T{start_end[0]}-{start_end[1]}/T{start_end[0]}-{start_end[1]}_0')
    plt.savefig(f'OneDrive/Desktop/transport_curve_decomposing_pathways/roles_connectivity_3D_{count}_{start_end}.png')
    
            
   
output = path_to_reduce(1,3) 
# print(output)
redistribution(output)