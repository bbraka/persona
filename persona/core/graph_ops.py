import json
from persona.core.neo4j_database import Neo4jConnectionManager
from persona.llm.embeddings import generate_embeddings
from persona.models.schema import (
    NodeModel, RelationshipModel, NodesAndRelationshipsResponse, 
    CommunityStructure, Subgraph, Node
)
from typing import List, Dict, Any, Optional
from persona.llm.llm_graph import detect_communities
from collections import defaultdict
from server.logging_config import get_logger

logger = get_logger(__name__)

class GraphOps:
    def __init__(self, neo4j_manager: Optional[Neo4jConnectionManager] = None):
        """Initialize GraphOps with an optional Neo4j manager"""
        self.neo4j_manager = neo4j_manager if neo4j_manager is not None else Neo4jConnectionManager()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        """Initialize the connection if not already initialized"""
        if not self.neo4j_manager.driver:
            await self.neo4j_manager.initialize()

    async def clean_graph(self):
        # Clean the graph
        await self.neo4j_manager.clean_graph()

    async def add_nodes(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add nodes.")
            return

        # Create nodes with names, properties, types, and chunk_id
        node_dicts = [{
            "name": node.name,
            "type": node.type or "",
            "properties": node.properties or {},
            "chunk_id": getattr(node, 'chunk_id', None)
        } for node in nodes]
        await self.neo4j_manager.create_nodes(node_dicts, user_id)

        # Generate and add embeddings for new nodes
        await self.add_nodes_batch_embeddings(nodes, user_id)


    async def add_nodes_batch_embeddings(self, nodes: List[NodeModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add embeddings.")
            return

        # Generate embeddings for all nodes in one batch
        node_names = [node.name for node in nodes]
        embeddings = generate_embeddings(node_names)  # This will now embed the full narrative/label
        
        # Add embeddings to nodes
        for node_name, embedding in zip(node_names, embeddings):
            if embedding:
                await self.neo4j_manager.add_embedding_to_vector_index(node_name, embedding, user_id)
            else:
                logger.error(f"Failed to generate embedding for node: {node_name}")


    async def add_relationships(self, relationships: List[RelationshipModel], user_id: str):
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot add relationships.")
            return

        relationship_dicts = [rel.dict() for rel in relationships]
        await self.neo4j_manager.create_relationships(relationship_dicts, user_id)

    async def get_node_data(self, node_name: str, user_id: str) -> NodeModel:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node data.")
            return NodeModel(name=node_name, type=None, embedding=None)

        node_data = await self.neo4j_manager.get_node_data(node_name, user_id)
        if node_data:
            return NodeModel(
                name=node_data["name"],
                type=node_data.get("type"),
                properties=node_data.get("properties", {}),
                embedding=node_data.get("embedding")
            )
        return NodeModel(name=node_name, type=None, embedding=None)

    async def get_node_relationships(self, node_name: str, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get node relationships.")
            return []

        relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)
        return [RelationshipModel(source=rel["source"], target=rel["target"], relation=rel["relation"]) 
                for rel in relationships]

    async def text_similarity_search(self, query: str, user_id: str, limit: int = 5, threshold: float = 0.7, index_name: str = "embeddings_index") -> Dict[str, Any]:
        """
        Perform a similarity search on the graph based on a text query.
        
        Args:
            query: Text query to search for
            user_id: User ID to filter results by
            limit: Maximum number of results to return after filtering (default: 5)
            threshold: Minimum similarity score to include (0.0-1.0, default: 0.7)
            index_name: Name of the vector index to query
            
        Returns:
            Dictionary with query and filtered results
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        logger.debug(f"Generating embedding for query: '{query}' for user ID: '{user_id}'")
        query_embeddings = generate_embeddings([query])
        if not query_embeddings[0]:
            return {"query": query, "results": []}

        logger.debug(f"Performing similarity search for the query: '{query}' for user ID: '{user_id}'")
        # Fetch more results than limit to account for threshold filtering
        fetch_limit = max(limit * 3, 20)  # Fetch 3x limit or at least 20 results
        results = await self.neo4j_manager.query_text_similarity(query_embeddings[0], user_id, limit=fetch_limit)

        # Filter by threshold and apply limit
        filtered = [r for r in results if r["score"] >= threshold][:limit]
        return {
            "query": query,
            "results": [
                {
                    "nodeId": result["nodeId"],
                    "nodeName": result["nodeName"],
                    "score": result["score"]
                } for result in filtered
            ]
        }
    
    async def update_graph(self, graph_update: NodesAndRelationshipsResponse, user_id: str):
        """
        Update the graph with new nodes and relationships

        Args:
            graph_update: Contains nodes and relationships to add
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return

        if graph_update.nodes:
            await self.add_nodes(graph_update.nodes, user_id)
        if graph_update.relationships:
            await self.add_relationships(graph_update.relationships, user_id)
        if not graph_update.nodes and not graph_update.relationships:
            logger.debug("No nodes or relationships to update.")

    async def update_graph_with_bloom_transactional(
        self,
        graph_update: NodesAndRelationshipsResponse,
        user_id: str
    ) -> None:
        """
        Update graph with nodes, relationships, embeddings, and bloom levels in a single transaction.
        If any step fails, all changes are rolled back for data consistency.
        
        Args:
            graph_update: Contains nodes and relationships to add
            user_id: User ID
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update graph.")
            return
        
        if not graph_update.nodes and not graph_update.relationships:
            logger.debug("No nodes or relationships to update.")
            return
        
        # Prepare data for transaction
        nodes_data = []
        embeddings_data = []
        
        # Process nodes and embeddings
        for node in graph_update.nodes:
            nodes_data.append({
                "name": node.name,
                "type": node.type or "",
                "properties": node.properties or {}
            })
            
            # Add embedding data
            if node.embedding:
                embeddings_data.append({
                    "node_name": node.name,
                    "embedding": node.embedding
                })
        
        # Process relationships
        relationships_data = [{
            "source": rel.source,
            "target": rel.target,
            "relation": rel.relation
        } for rel in graph_update.relationships]
        
        # Calculate bloom levels for all affected nodes
        bloom_updates = []
        
        # Create a lookup map for new nodes to preserve their LLM-extracted properties
        new_nodes_map = {node.name: node.properties for node in graph_update.nodes}
        
        affected_nodes = set([node.name for node in graph_update.nodes])
        
        # Get neighbors of new nodes to recalculate their bloom levels too
        for node in graph_update.nodes:
            try:
                neighbors = await self.get_node_relationships(node.name, user_id)
                affected_nodes.update([rel.target for rel in neighbors])
                affected_nodes.update([rel.source for rel in neighbors])
            except Exception as e:
                logger.debug(f"Could not get neighbors for {node.name}: {e}")
        
        # Calculate bloom level for each affected node
        for node_name in affected_nodes:
            try:
                bloom_level = await self.calculate_bloom_level(node_name, user_id)
                
                # For new nodes, use properties from graph_update (LLM-extracted)
                # For existing nodes, fetch from DB to preserve their existing properties
                if node_name in new_nodes_map:
                    # New node: use LLM-extracted properties (discipline, confidence)
                    props = new_nodes_map[node_name]
                    current_props = props.copy() if props else {}
                else:
                    # Existing node: fetch current properties from DB
                    node_data = await self.get_node_data(node_name, user_id)
                    current_props = node_data.properties.copy() if node_data.properties else {}
                
                # Update with calculated bloom level (overwrites LLM bloom_level with topology-based)
                current_props['bloom_level'] = bloom_level
                
                bloom_updates.append({
                    "node_name": node_name,
                    "properties": current_props
                })
            except Exception as e:
                logger.warning(f"Could not calculate bloom level for {node_name}: {e}")
        
        # Execute everything in a single transaction
        await self.neo4j_manager.update_graph_transactional(
            nodes=nodes_data,
            relationships=relationships_data,
            embeddings_data=embeddings_data,
            bloom_updates=bloom_updates,
            user_id=user_id
        )        

    async def close(self):
        """Close the Neo4j connection"""
        logger.info("Closing Neo4j connection...")
        await self.neo4j_manager.close()

    async def get_all_nodes(self, user_id: str) -> List[NodeModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all nodes.")
            return []

        nodes = await self.neo4j_manager.get_all_nodes(user_id)
        return [NodeModel(
            name=node['name'],
            type=node.get('type'),
            properties=node.get('properties', {}),
            embedding=node.get('embedding')
        ) for node in nodes]

    async def get_all_relationships(self, user_id: str) -> List[RelationshipModel]:
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot get all relationships.")
            return []

        relationships = await self.neo4j_manager.get_all_relationships(user_id)
        return [RelationshipModel(source=rel['source'], target=rel['target'], relation=rel['relation']) for rel in relationships]
    
    async def create_user(self, user_id: str) -> None:
        await self.neo4j_manager.create_user(user_id)

    async def delete_user(self, user_id: str) -> None:
        await self.neo4j_manager.delete_user(user_id)

    async def user_exists(self, user_id: str) -> bool:
        return await self.neo4j_manager.user_exists(user_id)

    async def perform_similarity_search(
        self, 
        query: str, 
        embedding: List[float],
        user_id: str, 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Perform similarity search using pre-computed embedding
        """
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot perform similarity search.")
            return {"query": query, "results": []}

        try:
            logger.debug(f"Performing similarity search for: {query}")
            results = await self.neo4j_manager.query_text_similarity(embedding, user_id)
            logger.debug(f"Found {len(results)} similar nodes for '{query}'")
            
            return {
                "query": query,
                "results": [
                    {
                        "nodeId": result["nodeId"],
                        "nodeName": result["nodeName"],
                        "score": result["score"]
                    } for result in results
                ]
            }
        except Exception as e:
            logger.error(f"Error in similarity search for {query}: {str(e)}")
            return {"query": query, "results": []}
        

    async def get_ranked_subgraphs(self, user_id: str) -> List[Subgraph]:
        """Get all subgraphs in the graph, ranked by size and influence"""
        subgraphs = []
        visited_nodes = set()

        # Retrieve all nodes for the user
        all_nodes = await self.neo4j_manager.get_all_nodes(user_id)

        for node in all_nodes:
            node_name = node['name']
            if node_name not in visited_nodes:
                # Retrieve all relationships for this node
                relationships = await self.neo4j_manager.get_node_relationships(node_name, user_id)

                # Collect all connected nodes
                connected_nodes = {node_name}
                for rel in relationships:
                    connected_nodes.add(rel['source'])
                    connected_nodes.add(rel['target'])

                # Mark nodes as visited
                visited_nodes.update(connected_nodes)

                # Calculate central nodes based on degree
                central_nodes = await self._get_central_nodes(list(connected_nodes), relationships)

                subgraphs.append(Subgraph(
                    id=len(subgraphs),  # Assign a unique ID based on the order
                    nodes=list(connected_nodes),
                    relationships=relationships,
                    size=len(connected_nodes),
                    central_nodes=central_nodes
                ))

        # Sort subgraphs by size in descending order
        subgraphs.sort(key=lambda sg: sg.size, reverse=True)

        return subgraphs

    async def _get_central_nodes(self, nodes: List[str], relationships: List[Dict]) -> List[str]:
        """Calculate central nodes based on degree centrality"""
        if not nodes:  # Handle empty subgraph case
            return []
            
        degree_count = defaultdict(int)
        for rel in relationships:
            degree_count[rel['source']] += 1
            degree_count[rel['target']] += 1
        
        # If we have nodes with relationships, return the most central one
        if degree_count:
            sorted_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
            # Return only the node name (first element of the tuple)
            return [sorted_nodes[0][0]]
        
        # If no relationships exist, return the first node as representative
        return [nodes[0]]
    

    async def format_subgraphs_for_llm(self, subgraphs: List[Subgraph]) -> str:
        """Format subgraphs into a string for LLM input"""
        formatted = "# Graph Structure Analysis\n\n"
        
        for subgraph in subgraphs:
            formatted += f"\n## Subgraph {subgraph.id} (Size: {subgraph.size})\n"
            formatted += f"Central Nodes: {', '.join(subgraph.central_nodes)}\n"
            formatted += "\nNodes:\n"
            for node in subgraph.nodes:
                formatted += f"- {node}\n"
            
            formatted += "\nRelationships:\n"
            for rel in subgraph.relationships:
                formatted += f"- {rel['source']} {rel['relation']} {rel['target']}\n"
        
        return formatted

    async def make_communities(self, user_id: str, community_structure: CommunityStructure, subgraphs: List[Subgraph]) -> None:
        """Create community structure in the graph"""
        for header in community_structure.communityHeaders:
            # Create header node
            await self.neo4j_manager.create_nodes([{
                'name': header.header,
                'type': 'CommunityHeader'
            }], user_id)
            
            for subheader in header.subheaders:
                # Create subheader node
                await self.neo4j_manager.create_nodes([{
                    'name': subheader.subheader,
                    'type': 'CommunitySubheader'
                }], user_id)
                
                # Connect header to subheader
                await self.neo4j_manager.create_relationships([{
                    'source': header.header,
                    'target': subheader.subheader,
                    'relation': 'HAS_SUBHEADER'
                }], user_id)
                
                # Connect representative nodes from subgraphs to this subheader
                for subgraph_id in subheader.subgraph_ids:
                    if subgraph_id < len(subgraphs):
                        subgraph = subgraphs[subgraph_id]
                        for central_node in subgraph.central_nodes:
                            await self.neo4j_manager.create_relationships([{
                                'source': central_node,
                                'target': subheader.subheader,
                                'relation': 'BELONGS_TO'
                            }], user_id)

    async def community_detection(self, user_id) -> None:
        """Main orchestrator for community detection process"""
        # Get ranked subgraphs
        subgraphs = await self.get_ranked_subgraphs(user_id)
        
        # Format for LLM
        subgraphs_text = await self.format_subgraphs_for_llm(subgraphs)
        
        # Get community structure from LLM
        community_structure = await detect_communities(subgraphs_text)
        
        # Create community structure in graph
        await self.make_communities(user_id, community_structure, subgraphs)
        
    async def calculate_bloom_level(self, node_name: str, user_id: str) -> str:
            """Calculate Bloom's taxonomy level based on graph evidence using a single optimized query"""
            await self.initialize()
            
            if not self.neo4j_manager.driver:
                logger.error("Neo4j driver is not initialized.")
                return "Remember"
            
            # Combined query: get degree and contexts in one call
            query = """
            MATCH (n:NodeName {name: $node_name, UserId: $user_id})
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as degree
            OPTIONAL MATCH (n)-[*1..2]-(connected)
            WHERE connected.type IS NOT NULL
            RETURN degree, collect(DISTINCT connected.type) as contexts
            """
            
            async with self.neo4j_manager.driver.session() as session:
                result = await session.run(query, node_name=node_name, user_id=user_id)
                data = await result.data()
                
                if not data:
                    return "Remember"
                
                record = data[0]
                degree = record.get('degree', 0) or 0
                contexts = [c for c in record.get('contexts', []) if c]
                
                # Evidence-based bloom level
                if degree >= 6 and len(contexts) >= 3:
                    return "Analyze"
                elif degree >= 3 and len(contexts) >= 2:
                    return "Apply"
                elif degree >= 2:
                    return "Understand"
                else:
                    return "Remember"
    
    async def update_node_properties(self, node_name: str, user_id: str, properties: Dict[str, Any]) -> None:
        """Update properties of an existing node"""
        if not await self.user_exists(user_id):
            logger.warning(f"User {user_id} does not exist. Cannot update node properties.")
            return
        
        await self.initialize()
        
        if not self.neo4j_manager.driver:
            logger.error("Neo4j driver is not initialized.")
            return
        
        query = """
        MATCH (n:NodeName {name: $node_name, UserId: $user_id})
        SET n.discipline = $discipline,
            n.bloom_level = $bloom_level,
            n.confidence = $confidence
        RETURN n.name as name
        """
        async with self.neo4j_manager.driver.session() as session:
            result = await session.run(query, 
                node_name=node_name, 
                user_id=user_id, 
                discipline=properties.get("discipline", ""),
                bloom_level=properties.get("bloom_level", ""),
                confidence=properties.get("confidence", 0.0)
            )
            data = await result.data()
            if data:
                logger.debug(f"Updated properties for node: {node_name}")
            else:
                logger.warning(f"Node {node_name} not found for user {user_id}")

class GraphContextRetriever:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops

    async def get_rich_context(self, query: str, user_id: str, top_k: int = 5, max_hops: int = 2) -> str:
        """
        Get rich context from the graph based on a text query.
        """
        similar_nodes = await self.graph_ops.text_similarity_search(query=query, user_id=user_id, limit=top_k, index_name="embeddings_index")
        context = await self.crawl_graph(similar_nodes['results'], max_hops, user_id)
        return self.format_separated_context(context)

    async def crawl_graph(self, start_nodes, max_hops, user_id):
        """
        Crawl the graph to get rich context.
        """
        context = {}
        for node in start_nodes:
            logger.debug(f"Exploring node: {node}")
            await self.explore_node(node['nodeName'], context, max_hops, user_id)
        return context

    async def explore_node(self, node_name, context, hops_left, user_id):
        """
        Explore a node and its relationships up to max_hops away.
        """
        if node_name in context or hops_left < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)
        
        context[node_name] = {
            'properties': node_data.properties,
            'relationships': []
        }

        for rel in relationships:
            context[node_name]['relationships'].append(f"{rel.relation} -> {rel.target}")
            if hops_left > 0:
                await self.explore_node(rel.target, context, hops_left - 1, user_id)

    def format_separated_context(self, context):
        """
        Format the context into a readable string with graph structure and node details.
        """
        graph_structure = []
        node_descriptions = []

        for node, data in context.items():
            for relationship in data['relationships']:
                graph_structure.append(f"{node} -> {relationship}")
            
            node_descriptions.append(f"### {node}")
            if data.get('properties'):
                for key, value in data['properties'].items():
                    node_descriptions.append(f"{key}: {value}")
            node_descriptions.append("")

        formatted = "# User Knowledge Graph\n\n## Graph Structure\n```\n"
        formatted += "\n".join(graph_structure)
        formatted += "\n```\n\n## Node Details\n\n"
        formatted += "\n".join(node_descriptions)

        return formatted
    
    async def get_relevant_graph_context(self, nodes: List[Node], user_id: str, max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the given nodes.
        """
        context = "# Relevant Graph Context\n\n"
        
        # Check if there are any existing nodes in the graph
        existing_nodes = await self.graph_ops.get_all_nodes(user_id)
        if not existing_nodes:
            logger.debug("No existing nodes in graph (t=0). Skipping context retrieval.")
            return context
        
        logger.debug(f"Found {len(existing_nodes)} existing nodes in graph")
        
        # Start exploring from the provided nodes directly
        subgraph = {}
        for node in nodes:
            await self._explore_node(node_name=node.name, subgraph=subgraph, user_id=user_id, max_hops=max_hops)
        
        # Format the subgraph context
        if subgraph:
            context += "## Related Nodes and Relationships\n"
            for node_name, data in subgraph.items():
                context += f"\n### {node_name}\n"
                if data['relationships']:
                    context += "Relationships:\n"
                    for rel in data['relationships']:
                        context += f"- {rel}\n"
        
        return context

    async def _explore_node(self, node_name: str, subgraph: Dict, user_id: str, max_hops: int):
        """Helper method to explore node relationships for context building"""
        if node_name in subgraph or max_hops < 0:
            return
        
        node_data = await self.graph_ops.get_node_data(node_name, user_id)
        relationships = await self.graph_ops.get_node_relationships(node_name, user_id)
        
        subgraph[node_name] = {
            'properties': node_data.properties,
            'relationships': []
        }
        
        for rel in relationships:
            subgraph[node_name]['relationships'].append(
                f"{rel.source} {rel.relation} {rel.target}"
            )
            if max_hops > 0:
                next_node = rel.target if rel.source == node_name else rel.source
                await self._explore_node(next_node, subgraph, user_id, max_hops - 1)

    async def get_entire_graph_context(self, user_id: str) -> str:
        """
        Get graph context relative to the nodes, and user psyche.
        Currently gets the entire graph context, which is not efficient.
        """
        # TODO: This is not efficient, only get the context for the nodes that are relevant.
        # TODO: Use more efficient sophisticated context retrieval techniques like graph traversal, etc.
        # TODO: Use vector search to get the context for the nodes.
    
        nodes = await self.graph_ops.get_all_nodes(user_id)
        relationships = await self.graph_ops.get_all_relationships(user_id)
       
        context = "# Current Knowledge Graph\n\n## Nodes\n"
        for node in nodes:
            context += f"- {node.name} (type: {node.type})\n"
        
        context += "\n## Relationships\n"
        for rel in relationships:
            context += f"- {rel.source} {rel.relation} {rel.target}\n"
        
        return context

    async def get_graph_context(self, query: str, user_id: str):
        logger.debug(f"Getting graph context for query: {query}")
        query_embeddings = generate_embeddings([query])
        if not query_embeddings[0]:
            return {"query": query, "results": []}
        results = await self.graph_ops.perform_similarity_search(query=query, embedding=query_embeddings[0], user_id=user_id)
        return results