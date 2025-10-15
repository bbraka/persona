from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.llm.llm_graph import get_nodes, get_relationships, Node as LLMNode
from persona.llm.embeddings import generate_embeddings
from persona.models.schema import (
    NodeModel, RelationshipModel, GraphUpdateModel,
    UnstructuredData, NodesAndRelationshipsResponse, Node, Relationship
)
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import asyncio
from server.logging_config import get_logger

logger = get_logger(__name__)

class GraphConstructor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.graph_ops = None
        self.graph_context_retriever = None

    async def __aenter__(self):
        self.graph_ops = await GraphOps().__aenter__()
        self.graph_context_retriever = GraphContextRetriever(self.graph_ops)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.graph_ops is not None:
            await self.graph_ops.__aexit__(exc_type, exc_val, exc_tb)

    async def clean_graph(self):
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        await self.graph_ops.clean_graph()

    async def ingest_unstructured_data_to_graph(self, data: UnstructuredData):
        """
        Ingest unstructured data into the graph.
        This process now includes:
        1. Extracting meaningful, self-contained nodes from the content
        2. Finding strong, justified relationships between new nodes
        3. Selectively connecting with existing nodes only when truly relevant
        """
        if self.graph_ops is None or self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        text = self.preprocess_data(data)
        
        # Extract new nodes from the content
        new_nodes = await self.extract_nodes(text)
        if not new_nodes:
            logger.info("No new nodes generated from the unstructured data.")
            return

        # Get existing graph context
        existing_context = await self.graph_context_retriever.get_rich_context(text, self.user_id)
        
        # Generate relationships - now more selective
        relationships = []
        
        # Phase 1: Core relationships between new nodes
        new_node_relationships = await self.generate_relationships(new_nodes)
        relationships.extend(new_node_relationships)
        
        # Phase 2: Only connect with existing nodes if there's strong relevance
        if existing_context and len(new_nodes) > 0:
            mixed_relationships = await self.generate_cross_relationships(new_nodes, existing_context)
            # Filter for stronger relationships (we might want to add a confidence score)
            relationships.extend(mixed_relationships)
        
        # Create the graph update - nodes with type information and PKG properties
        # Generate embeddings for each node
        node_texts = [node.name for node in new_nodes]
        embeddings = generate_embeddings(node_texts)

        # Build properties dict from PKG fields
        nodes = []
        for node, embedding in zip(new_nodes, embeddings):
            properties = {}
            if node.discipline:
                properties["discipline"] = node.discipline
            if node.bloom_level:
                properties["bloom_level"] = node.bloom_level
            if node.confidence is not None:
                properties["confidence"] = node.confidence
            nodes.append(NodeModel(name=node.name, type=node.type, properties=properties, embedding=embedding))
        
        relationships = [RelationshipModel(
            source=rel.source,
            target=rel.target,
            relation=rel.relation
        ) for rel in relationships]
        
        graph_update = NodesAndRelationshipsResponse(
            nodes=nodes,
            relationships=relationships
        )
        
        # Update the graph with transaction support (atomic: nodes + relationships + embeddings + bloom levels)
        # If any step fails, all changes are rolled back
        try:
            await self.graph_ops.update_graph_with_bloom_transactional(graph_update, self.user_id)
            logger.info(f"Successfully ingested {len(nodes)} nodes and {len(relationships)} relationships with bloom levels")
        except Exception as e:
            logger.error(f"Failed to ingest data (transaction rolled back): {e}")
            raise
            
    def preprocess_data(self, data: UnstructuredData) -> str:
        """
        Preprocess the data, combine relevant fields into a single string.
        """
        preprocessed = f"{data.title}\n{data.content}\n"
        if data.metadata:
            preprocessed += "\n".join([f"{k}: {v}" for k, v in data.metadata.items()])
        return preprocessed

    async def extract_nodes(self, text: str) -> List[Node]:
        """
        Extract nodes from the unstructured text.
        """
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=[])
        llm_nodes = await get_nodes(text, graph_context)
        # Convert LLM nodes to schema nodes
        return [Node(name=node.name, type=node.type, discipline=getattr(node, 'discipline', ''), bloom_level=getattr(node, 'bloom_level', ''), confidence=getattr(node, 'confidence', 0.0)) for node in llm_nodes]

    async def generate_relationships(self, nodes: List[Node], context_description: str = "") -> List[Relationship]:
        """
        Generate core relationships between nodes.
        Only creates relationships that are strongly justified.
        """
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=nodes)
        # Convert schema nodes to LLM nodes
        llm_nodes = [LLMNode(name=node.name, type=node.type, discipline=node.discipline, bloom_level=node.bloom_level, confidence=node.confidence) for node in nodes]
        llm_relationships, _ = await get_relationships(llm_nodes, graph_context)  # Ignore the ID mapping
        # Convert LLM relationships to schema relationships
        return [Relationship(source=rel.source, target=rel.target, relation=rel.relation) for rel in llm_relationships]

    async def generate_cross_relationships(self, new_nodes: List[Node], existing_context: str) -> List[Relationship]:
        """
        Generate relationships between new and existing nodes.
        Only creates relationships when there's a strong, meaningful connection.
        """
        # Convert schema nodes to LLM nodes
        llm_nodes = [LLMNode(name=node.name, type=node.type, discipline=node.discipline, bloom_level=node.bloom_level, confidence=node.confidence) for node in new_nodes]
        llm_relationships, _ = await get_relationships(llm_nodes, existing_context)  # Ignore the ID mapping
        # Convert LLM relationships to schema relationships
        return [Relationship(source=rel.source, target=rel.target, relation=rel.relation) for rel in llm_relationships]

    async def discover_new_relationships(self, new_context: str, existing_context: str) -> List[Relationship]:
        """
        Discover potential new relationships between existing nodes based on new context.
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        
        # Extract existing nodes from the context
        existing_nodes = await self.graph_ops.get_all_nodes(self.user_id)
        if not existing_nodes:
            return []
            
        # Convert NodeModel instances to Node instances for the LLM
        nodes_for_llm = [LLMNode(name=node.name, type="Unknown", discipline="", bloom_level="", confidence=0.0) for node in existing_nodes]  # Add required type field
        
        # Use the new context to find new relationships
        combined_context = f"New Information:\n{new_context}\n\nExisting Knowledge:\n{existing_context}"
        llm_relationships, _ = await get_relationships(nodes_for_llm, combined_context)  # Ignore the ID mapping
        
        # Convert LLM relationships to schema relationships
        return [Relationship(source=rel.source, target=rel.target, relation=rel.relation) for rel in llm_relationships]

    async def get_relevant_graph_context(self, user_id: str, nodes: List[Node], max_hops: int = 2) -> str:
        """
        Get relevant subgraph context for the given nodes.
        """
        if self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
        return await self.graph_context_retriever.get_relevant_graph_context(nodes=nodes, user_id=user_id, max_hops=max_hops)

    async def close(self):
        await self.__aexit__(None, None, None)



    
