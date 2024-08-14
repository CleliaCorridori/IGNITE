import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


    
# # ------ ADJACENCY MATRIX ------
def to_adj_matrix(matrix, thr=0):
    """Converts an interaction matrix to an adjacency matrix.
    
    Values with absolute value below the threshold are set to 0.
    Negative values are set to -1, and positive values to +1.
    
    Args:
        matrix (np.ndarray): The interaction matrix.
        thr (float): The threshold for determining significance of values.
    
    Returns:
        np.ndarray: The adjacency matrix.
    """
    threshold = thr * np.max(np.abs(matrix))
    adj_matrix = np.zeros_like(matrix)

    adj_matrix[matrix < -threshold] = -1
    adj_matrix[matrix > threshold] = 1

    return adj_matrix

# # ------ NETWORK VISUALIZATION -------

def visualize_graphTrue(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Funzione per visualizzare la rete delle interazioni note da una lista """
    # Crea un grafo diretto dalla matrice di adiacenza
    G = nx.DiGraph(adj_matrix.T)
    # Rinomina i nodi con i nuovi nomi
    G = nx.relabel_nodes(G, {i: node_name for i, node_name in enumerate(node_names)})

    # Crea un set di nodi unici coinvolti nelle interazioni
    involved_nodes = set()
    for interaction in interactions:
        node1, node2, _ = interaction.split(" ")
        involved_nodes.add(node1)
        involved_nodes.add(node2)

    # Filtra solo i nodi coinvolti nelle interazioni
    G = G.subgraph(involved_nodes)

    # Disegna il grafo usando il layout circolare
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)

    # Crea un dizionario di mappatura nodo-colore
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes() if node in involved_nodes}

    # Disegna i nodi con colori diversi
    nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node) for node in G.nodes()], node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=18)

    # Disegna gli archi con FancyArrowPatch
    total_edges_drawn = 0
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if node1 in G and node2 in G:  # Controlla se entrambi i nodi sono nel grafo
            if direction == "1":
                edge_color = 'r'
            else:
                edge_color = 'b'
            # Disegno personalizzato della freccia con FancyArrowPatch
            arrow = FancyArrowPatch(posA=pos[node1], posB=pos[node2],
                                    arrowstyle='simple,head_length=15,head_width=10,tail_width=3',
                                    color=edge_color,
                                    connectionstyle='arc3,rad=-0.05',
                                    mutation_scale=3,
                                    lw=1.0,
                                    alpha=0.7,
                                    shrinkA=0.5, shrinkB=0.5)
            plt.gca().add_patch(arrow)
            total_edges_drawn += 1
    print(f"Total edges drawn: {total_edges_drawn}")
    # plt.title(title, fontsize=34)


def visualize_graphSel(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Funzione per visualizzare la rete delle interazioni correttamente inferite da una lista """
    # Crea un grafo diretto dalla matrice di adiacenza
    G = nx.DiGraph(adj_matrix.T)
    # Rinomina i nodi con i nuovi nomi
    G = nx.relabel_nodes(G, {i: node_name for i, node_name in enumerate(node_names)})

    # Crea un set di nodi unici coinvolti nelle interazioni
    involved_nodes = set()
    for interaction in interactions:
        node1, node2, _ = interaction.split(" ")
        involved_nodes.add(node1)
        involved_nodes.add(node2)

    # Filtra solo i nodi coinvolti nelle interazioni
    G = G.subgraph(involved_nodes)

    # Disegna il grafo usando il layout circolare
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)

    # Crea un dizionario di mappatura nodo-colore
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes() if node in involved_nodes}

    # Disegna i nodi con colori diversi
    nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node) for node in G.nodes()], node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=18)

    # Disegna solo le interazioni date se esistono
    total_edges_drawn = 0
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if node1 in G and node2 in G:  # Controlla se entrambi i nodi sono nel grafo
            if direction == "1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)] > 0:
                edge_color = 'r'
                edges_to_draw = (node1, node2)
                
                # Disegno personalizzato della freccia con FancyArrowPatch
                arrow = FancyArrowPatch(posA=pos[edges_to_draw[0]], posB=pos[edges_to_draw[1]],
                                        arrowstyle='simple,head_length=15,head_width=10,tail_width=3',
                                        color=edge_color,
                                        connectionstyle='arc3,rad=-0.05',
                                        mutation_scale=3,
                                        lw=1.0,
                                        alpha=0.7,
                                        shrinkA=0.5, shrinkB=0.5)
                plt.gca().add_patch(arrow)
                total_edges_drawn += 1
            
            elif direction == "-1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)] < 0:
                edge_color = 'b'
                edges_to_draw = (node1, node2)

                # Disegno personalizzato della freccia con FancyArrowPatch
                arrow = FancyArrowPatch(posA=pos[edges_to_draw[0]], posB=pos[edges_to_draw[1]],
                                        arrowstyle='simple,head_length=15,head_width=10,tail_width=3',
                                        color=edge_color,
                                        connectionstyle='arc3,rad=-0.05',
                                        mutation_scale=3,
                                        lw=1.0,
                                        alpha=0.7,
                                        shrinkA=0.5, shrinkB=0.5)
                plt.gca().add_patch(arrow)
                total_edges_drawn += 1
    
    print(f"Total edges drawn: {total_edges_drawn}")



def visualize_graphSel_undirected(adj_matrix, node_names, naive_nodes, formative_nodes, committed_nodes, interactions=[], title=""):
    """ Funzione per visualizzare la rete delle interazioni correttamente inferite da una lista """
    # Crea un grafo diretto dalla matrice di adiacenza
    G = nx.DiGraph(adj_matrix.T)
    # Rinomina i nodi con i nuovi nomi
    G = nx.relabel_nodes(G, {i: node_name for i, node_name in enumerate(node_names)})

    # Crea un set di nodi unici coinvolti nelle interazioni
    involved_nodes = set()
    for interaction in interactions:
        node1, node2, _ = interaction.split(" ")
        involved_nodes.add(node1)
        involved_nodes.add(node2)

    # Filtra solo i nodi coinvolti nelle interazioni
    G = G.subgraph(involved_nodes)

    # Disegna il grafo usando il layout circolare
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)

    # Crea un dizionario di mappatura nodo-colore
    color_map = {node: "lightskyblue" if node in naive_nodes else "palegoldenrod" if node in formative_nodes else "salmon" if node in committed_nodes else "silver" for node in G.nodes() if node in involved_nodes}

    # Disegna i nodi con colori diversi
    nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node) for node in G.nodes()], node_size=2000)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=18)

    # Disegna solo le interazioni date se esistono
    total_edges_drawn = 0
    for interaction in interactions:
        node1, node2, direction = interaction.split(" ")
        if node1 in G and node2 in G:  # Controlla se entrambi i nodi sono nel grafo
            if direction == "1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)] > 0:
                edge_color = 'r'
                edges_to_draw = (node1, node2)
                
                # Disegno personalizzato della freccia con FancyArrowPatch
                arrow = FancyArrowPatch(posA=pos[edges_to_draw[0]], posB=pos[edges_to_draw[1]],
                                        arrowstyle='-',
                                        color=edge_color,
                                        connectionstyle='arc3,rad=-0.05',
                                        mutation_scale=3,
                                        lw=10.0,
                                        alpha=0.7,
                                        shrinkA=0.5, shrinkB=0.5)
                plt.gca().add_patch(arrow)
                total_edges_drawn += 1
            
            elif direction == "-1" and adj_matrix[list(node_names).index(node2), list(node_names).index(node1)] < 0:
                edge_color = 'b'
                edges_to_draw = (node1, node2)

                # Disegno personalizzato della freccia con FancyArrowPatch
                arrow = FancyArrowPatch(posA=pos[edges_to_draw[0]], posB=pos[edges_to_draw[1]],
                                        arrowstyle='-',
                                        color=edge_color,
                                        connectionstyle='arc3,rad=-0.05',
                                        mutation_scale=3,
                                        lw=10.0,
                                        alpha=0.7,
                                        shrinkA=0.5, shrinkB=0.5)
                plt.gca().add_patch(arrow)
                total_edges_drawn += 1
    
    print(f"Total edges drawn: {total_edges_drawn}")

    