import csv
import networkx as nx
import numpy as np

# PARSE THE NETWORK FORM CSV 
def parse_network(csv_file):
    G = nx.DiGraph()
    links = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            link_id = int(row[0])
            desc = row[1]
            
            # EXTRACT NODES FROM DESCRIPTION 
            parts = desc.split(' to ')
            from_node = int(parts[0].split()[-1])
            to_node = int(parts[1].split()[-1])
            G.add_edge(from_node, to_node, link_id=link_id)
            links[link_id] = (from_node, to_node)
    return G, links

# LINK PERFORMANCE FUNCTION
def link_cost(flow):
    return 15 + flow / 120

# OD PAIRS
od_pairs = {
    (1, 3): 8000,
    (2, 4): 6000
}

# FIND ALL SIMPLE PATHS FOR EACH OD PAIR
def get_paths(G, od_pairs):
    paths = {}
    for origin, dest in od_pairs:
        paths[(origin, dest)] = list(nx.all_simple_paths(G, origin, dest))
    return paths

# ALL-OR-NOTHING ASSIGNMENT
def all_or_nothing(G, od_pairs, paths):
    link_flows = {link: 0.0 for link in G.edges}
    for (o, d), demand in od_pairs.items():
        
        # COMPUTE SHORTEST PATH BASED ON FREE-FLOW COSTS 
        costs = {edge: link_cost(0) for edge in G.edges}
        nx.set_edge_attributes(G, costs, 'cost')
        try:
            path = nx.shortest_path(G, o, d, weight='cost')
            path_edges = list(zip(path[:-1], path[1:]))
            for edge in path_edges:
                link_flows[edge] += demand
        except nx.NetworkXNoPath:
            print(f"No path from {o} to {d}")
    return link_flows

# COMPUTE SHORTEST PATH TRAVEL TIME
def compute_sptt(G, od_pairs):
    sptt = {}
    for (o, d), demand in od_pairs.items():
        
        # USE FREE-FLOW COSTS FOR SHORTEST PATH 
        costs = {edge: link_cost(0) for edge in G.edges}
        nx.set_edge_attributes(G, costs, 'cost')
        try:
            path = nx.shortest_path(G, o, d, weight='cost')
            path_cost = sum(link_cost(0) for _ in zip(path[:-1], path[1:]))
            sptt[(o, d)] = path_cost * demand
        except nx.NetworkXNoPath:
            sptt[(o, d)] = 0
    return sum(sptt.values())

# FRANK-WOLFE ITERATION
def frank_wolfe(G, od_pairs, paths, max_iter=1000, gamma=0.001, lambda_thresh=0.001):
    # INITIAL ALL-OR-NOTHING
    x = all_or_nothing(G, od_pairs, paths)
    sptt = compute_sptt(G, od_pairs)
    for iteration in range(max_iter):
        # COMPUTE CURRENT COSTS
        costs = {edge: link_cost(flow) for edge, flow in x.items()}
        nx.set_edge_attributes(G, costs, 'cost')

        # AUXILIARY FLOWS (ALL-OR-NOTHING ON CURRENT COSTS)
        y = all_or_nothing(G, od_pairs, paths)

        numerator = 0
        denominator = 0
        for edge in G.edges:
            xi = x[edge]
            yi = y[edge]
            ci_x = link_cost(xi)
            ci_y = link_cost(yi)
            numerator += (xi - yi) * (ci_x - ci_y)
            denominator += (xi - yi)**2 * (1/120)

        if denominator == 0:
            alpha = 1
        else:
            alpha = numerator / denominator
            alpha = max(0, min(1, alpha))

        # UPDATE FLOWS
        x_new = {edge: (1 - alpha) * x[edge] + alpha * y[edge] for edge in G.edges}

        # COMPUTE TSTT AND RELATIVE GAP
        tstt = sum(link_cost(flow) * flow for flow in x_new.values())
        relative_gap = (tstt - sptt) / sptt if sptt > 0 else 0

        # CHECK CONVERGENCE
        max_rel_change = max(abs(x_new[edge] - x[edge]) / (x[edge] + 1e-6) for edge in G.edges)
        if max_rel_change < lambda_thresh or relative_gap < gamma:
            print(f"Converged At Iteration {iteration+1}, Max Rel Change: {max_rel_change}, Relative GAP: {relative_gap}")
            break

        x = x_new

        if iteration % 100 == 0:
            print(f"Iteration {iteration+1}, max rel change: {max_rel_change}, relative gap: {relative_gap}")

    return x, sptt

# MAIN
if __name__ == "__main__":
    G, links = parse_network('CE521_H2Nodes.csv')
    paths = get_paths(G, od_pairs)

    print("Network Parsed:")
    print(f"Nodes: {list(G.nodes)}")
    print(f"Links: {links}")
    print(f"OD Pairs: {od_pairs}")
    print(f"Paths: {paths}")

    # RUN FRANK-WOLFE
    final_flows, sptt = frank_wolfe(G, od_pairs, paths, gamma=0.001, lambda_thresh=0.001)

    # COMPUTE TRAVEL TIMES AND TOTAL SYSTEM TRAVEL TIME
    travel_times = {edge: link_cost(flow) for edge, flow in final_flows.items()}
    total_tstt = sum(flow * time for flow, time in zip(final_flows.values(), travel_times.values()))
    relative_gap = (total_tstt - sptt) / sptt if sptt > 0 else 0

    print("\nFinal Link Flows:")
    for link, flow in final_flows.items():
        print(f"Link {G.edges[link]['link_id']}: {flow:.2f} Vehicles")

    print("\nTravel Times:")
    for link, time in travel_times.items():
        print(f"Link {G.edges[link]['link_id']}: {time:.2f} Minutes")

    print(f"\nShortest Path Travel Time (SPTT): {sptt:.2f} Vehicle-Minutes")
    print(f"Total System Travel Time (TSTT): {total_tstt:.2f} Vehicle-Minutes")
    print(f"Relative Gap: {relative_gap:.6f}")