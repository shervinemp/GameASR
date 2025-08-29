from typing import List, Dict, Any

from .knowledge_base import KnowledgeGraph
from ..common.utils import get_logger


class ExplorationEngine:
    """
    Manages the exploration of the knowledge graph from a set of starting nodes.

    This engine performs a multi-hop traversal to gather a rich, contextual
    subgraph of information surrounding the initial nodes, which can then be
    used by the generator.
    """
    def __init__(self, graph: "KnowledgeGraph"):
        self.logger = get_logger(__file__)
        self.graph = graph

    def explore(self, initial_node_ids: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Performs a multi-hop graph expansion from a set of initial node IDs.

        This method gathers a subgraph containing the initial nodes and all unique
        neighbor nodes up to `max_hops` away.

        Args:
            initial_node_ids (List[str]): A list of IDs for the starting nodes.
            max_hops (int, optional): The maximum number of hops to traverse from
                the initial nodes. Defaults to 2.

        Returns:
            List[Dict[str, Any]]: A list of unique node dictionaries found
                                  during the exploration, including the initial
                                  nodes and their neighbors.
        """
        if not initial_node_ids:
            return []

        self.logger.info(f"Starting {max_hops}-hop exploration from {len(initial_node_ids)} initial nodes.")

        # 1. Get the initial nodes' data from the graph to ensure they are included.
        initial_subgraph = self.graph.subgraph(initial_node_ids)
        initial_node_data = initial_subgraph.get('nodes', [])

        # 2. Expand to find neighbors up to max_hops away.
        # `expansion` now returns a list of lists, so we need to flatten it.
        neighbor_data_lists = self.graph.expansion(
            frontier_ids=initial_node_ids,
            excluded_ids=initial_node_ids,
            max_hops=max_hops
        )

        # Flatten the list of lists and get the node dictionary from each item
        neighbor_nodes = [
            item['node']
            for sublist in neighbor_data_lists
            for item in sublist
        ]

        self.logger.info(f"Found {len(neighbor_nodes)} unique neighbors within {max_hops} hops.")

        # 3. Combine initial nodes and their neighbors, ensuring no duplicates.
        combined_nodes = {node['id']: node for node in initial_node_data}
        for node in neighbor_nodes:
            if node['id'] not in combined_nodes:
                combined_nodes[node['id']] = node

        self.logger.info(f"Total unique nodes after expansion: {len(combined_nodes)}")

        return list(combined_nodes.values())
