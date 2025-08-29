from typing import List, Dict, Any

from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class ExplorationEngine:
    def __init__(self, graph: "KnowledgeGraph"):
        self.logger = get_logger(__file__)
        self.graph = graph

    def explore(self, initial_nodes: List[str]) -> List[Dict[str, Any]]:
        """
        Performs a single-step graph expansion.

        1. Retrieves the full data for the initial nodes.
        2. Finds the immediate neighbors of the initial nodes (1-hop expansion).
        3. Combines the initial nodes and their neighbors, ensuring no duplicates.
        """
        if not initial_nodes:
            return []

        self.logger.info(f"Starting exploration from {len(initial_nodes)} initial nodes.")

        # 1. Get the initial nodes' data from the graph
        initial_subgraph = self.graph.subgraph(initial_nodes)
        initial_node_data = initial_subgraph.get('nodes', [])

        # 2. Expand to find neighbors
        # The 'expansion' method returns a list of lists of neighbors.
        # We need to flatten it and extract the node data.
        neighbor_data_list = self.graph.expansion(
            frontier_ids=initial_nodes,
            excluded_ids=initial_nodes  # Exclude initial nodes from being "neighbors"
        )

        # Flatten the list of lists and get the 'node' dictionary from each item
        neighbor_nodes = [
            neighbor['node']
            for neighbors in neighbor_data_list
            for neighbor in neighbors
        ]

        self.logger.info(f"Found {len(neighbor_nodes)} neighbors.")

        # 3. Combine initial nodes and their neighbors
        combined_nodes = {node['id']: node for node in initial_node_data}
        for node in neighbor_nodes:
            if node['id'] not in combined_nodes:
                combined_nodes[node['id']] = node

        self.logger.info(f"Total nodes after expansion: {len(combined_nodes)}")

        return list(combined_nodes.values())
