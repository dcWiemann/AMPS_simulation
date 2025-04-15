

def build_electrical_nodes(components, connections):
    """
    Builds a mapping of electrical nodes by merging connected terminals.

    Returns:
    - electrical_nodes: { electrical_node_id: set((component_id, terminal_id), ...) }
    """
    class UnionFind:
        """ Disjoint Set Union (Union-Find) with path compression """
        def __init__(self):
            self.parent = {}

        def find(self, item):
            if self.parent[item] != item:
                self.parent[item] = self.find(self.parent[item])  # Path compression
            return self.parent[item]

        def union(self, item1, item2):
            root1 = self.find(item1)
            root2 = self.find(item2)
            if root1 != root2:
                self.parent[root2] = root1  # Merge sets

        def add(self, item):
            if item not in self.parent:
                self.parent[item] = item  # Initialize parent to itself

    # Initialize Union-Find
    uf = UnionFind()

    # Step 1: Register each component terminal dynamically
    terminal_to_component = {}  # Maps terminal ID to component ID
    for comp in components:
        if "data" in comp and "terminals" in comp["data"]:
            for terminal in comp["data"]["terminals"]:
                terminal_id = (comp["id"], terminal["id"])  # Unique terminal identifier
                uf.add(terminal_id)
                terminal_to_component[terminal_id] = comp["id"]

    # Step 2: Merge terminals based on circuit connections
    for edge in connections:
        source_terminal = (edge['source'], edge['sourceHandle'])
        target_terminal = (edge['target'], edge['targetHandle'])
        uf.union(source_terminal, target_terminal)

    # Step 3: Assign unique electrical node IDs
    electrical_node_mapping = {}  # Maps Union-Find root -> electrical node ID
    electrical_nodes = {}
    electrical_node_id = 0

    for component_terminal in uf.parent.keys():
        root = uf.find(component_terminal)
        if root not in electrical_node_mapping:
            electrical_node_mapping[root] = electrical_node_id
            electrical_node_id += 1

        assigned_node = electrical_node_mapping[root]
        if assigned_node not in electrical_nodes:
            electrical_nodes[assigned_node] = set()

        electrical_nodes[assigned_node].add(component_terminal)

    return electrical_nodes



def build_circuit_components(components, electrical_nodes):
    """
    Extracts and structures only relevant information for circuit simulation.

    Returns:
    - circuit_components: {component_id: { "type": str, "value": float, "terminals": {terminal_id: electrical_node} } }
    """
    circuit_components = {}

    # Reverse map: electrical node -> (component_id, terminal_id)
    terminal_to_node = {term: node for node, terminals in electrical_nodes.items() for term in terminals}

    for comp in components:
        comp_id = comp["id"]
        comp_type = comp["data"]["componentType"]
        comp_value = comp["data"].get("value", None)  # Some components might not have a value
        terminals = {}

        # Extract terminal mappings
        for terminal in comp["data"].get("terminals", []):
            term_id = terminal["id"]
            term_key = (comp_id, term_id)
            if term_key in terminal_to_node:
                terminals[term_id] = terminal_to_node[term_key]  # Map terminal to electrical node

        # Store cleaned component data
        circuit_components[comp_id] = {
            "type": comp_type,
            "value": comp_value,
            "terminals": terminals
        }

    return circuit_components