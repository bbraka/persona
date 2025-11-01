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
        
        # Extract new nodes from the content (pass metadata for entity IDs)
        new_nodes = await self.extract_nodes(text, data.metadata or {})
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

        # Phase 3: Book-scoped relationships (if this data is from a book)
        if data.metadata and 'book_id' in data.metadata:
            try:
                book_id = int(data.metadata['book_id'])
                book_relationships = await self.generate_book_scoped_relationships(new_nodes, book_id)
                relationships.extend(book_relationships)
                logger.info(f"Created {len(book_relationships)} book-scoped relationships for book {book_id}")
            except (ValueError, KeyError) as e:
                logger.debug(f"Could not extract book_id for book-scoped relationships: {e}")

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

            # Convert datetime to ISO string for NodeModel
            created_at_str = None
            if hasattr(node, 'created_at') and node.created_at:
                created_at_str = node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at)

            # Convert BloomLevelUpdate objects to dicts for NodeModel
            bloom_history_dicts = []
            if hasattr(node, 'bloom_history') and node.bloom_history:
                for update in node.bloom_history:
                    bloom_history_dicts.append({
                        "level": update.level,
                        "timestamp": update.timestamp.isoformat() if hasattr(update.timestamp, 'isoformat') else str(update.timestamp),
                        "source": update.source
                    })

            nodes.append(NodeModel(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids if node.chunk_ids else [],  # Always an array
                book_id=node.book_id if node.book_id else [],
                highlight_id=node.highlight_id if node.highlight_id else [],
                writing_id=node.writing_id if node.writing_id else [],
                properties=properties,
                embedding=embedding,
                created_at=created_at_str,
                bloom_history=bloom_history_dicts
            ))
        
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

            # NOTE: Layer 2 consolidation DISABLED - it was too slow and buggy
            # Layer 1 prevention (in add_nodes) is sufficient for preventing duplicates
            # await self._consolidate_duplicates_inline()

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

    async def extract_nodes(self, text: str, metadata: Dict[str, str] = {}) -> List[Node]:
        """
        Extract nodes from the unstructured text.
        Entity IDs (book_id, highlight_id, writing_id) and chunk_ids from metadata are applied to all extracted nodes.
        """
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=[])
        llm_nodes = await get_nodes(text, graph_context)

        # Extract chunk_ids from metadata (comma-separated string to array)
        chunk_ids_from_metadata = []
        if 'chunk_ids' in metadata:
            # Can be comma-separated string: "uuid1,uuid2,uuid3"
            chunk_str = metadata['chunk_ids'].strip()
            if chunk_str:
                chunk_ids_from_metadata = [chunk.strip() for chunk in chunk_str.split(',')]

        # Extract entity IDs from metadata (convert string values to int arrays)
        book_ids = []
        if 'book_id' in metadata:
            try:
                book_ids = [int(metadata['book_id'])]
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert book_id from metadata: {e}")

        highlight_ids = []
        if 'highlight_id' in metadata:
            try:
                highlight_ids = [int(metadata['highlight_id'])]
            except (ValueError, TypeError):
                pass

        writing_ids = []
        if 'writing_id' in metadata:
            try:
                writing_ids = [int(metadata['writing_id'])]
            except (ValueError, TypeError):
                pass

        # Extract date from metadata
        from datetime import datetime
        annotation_date = None
        if 'date' in metadata:
            try:
                # Parse ISO 8601 date string or accept datetime object
                date_value = metadata['date']
                if isinstance(date_value, str):
                    annotation_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                elif isinstance(date_value, datetime):
                    annotation_date = date_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date from metadata: {e}")

        # Convert LLM nodes to schema nodes, applying IDs from metadata
        nodes = []
        for node in llm_nodes:
            llm_book_id = getattr(node, 'book_id', [])
            final_book_id = llm_book_id or book_ids

            # Initialize temporal fields
            current_time = datetime.utcnow()
            bloom_level = getattr(node, 'bloom_level', '')

            # Create initial Bloom history entry if bloom_level is present
            from persona.models.schema import BloomLevelUpdate
            bloom_history = []
            if bloom_level:
                source_info = []
                if highlight_ids:
                    source_info.append(f"highlight_id:{highlight_ids[0]}")
                elif book_ids:
                    source_info.append(f"book_id:{book_ids[0]}")
                source = ', '.join(source_info) if source_info else None

                bloom_history.append(BloomLevelUpdate(
                    level=bloom_level,
                    timestamp=annotation_date or current_time,
                    source=source
                ))

            nodes.append(Node(
                name=node.name,
                type=node.type,
                chunk_ids=getattr(node, 'chunk_ids', []) or chunk_ids_from_metadata,
                book_id=final_book_id,
                highlight_id=getattr(node, 'highlight_id', []) or highlight_ids,
                writing_id=getattr(node, 'writing_id', []) or writing_ids,
                discipline=getattr(node, 'discipline', ''),
                bloom_level=bloom_level,
                confidence=getattr(node, 'confidence', 0.0),
                created_at=annotation_date or current_time,
                bloom_history=bloom_history
            ))

        return nodes

    async def generate_relationships(self, nodes: List[Node], context_description: str = "") -> List[Relationship]:
        """
        Generate core relationships between nodes.
        Only creates relationships that are strongly justified.
        """
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=nodes)
        # Convert schema nodes to LLM nodes
        llm_nodes = [
            LLMNode(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids,
                book_id=node.book_id,
                highlight_id=node.highlight_id,
                writing_id=node.writing_id,
                discipline=node.discipline,
                bloom_level=node.bloom_level,
                confidence=node.confidence
            ) for node in nodes
        ]
        llm_relationships, _ = await get_relationships(llm_nodes, graph_context)  # Ignore the ID mapping
        # Convert LLM relationships to schema relationships
        return [Relationship(source=rel.source, target=rel.target, relation=rel.relation) for rel in llm_relationships]

    async def generate_cross_relationships(self, new_nodes: List[Node], existing_context: str) -> List[Relationship]:
        """
        Generate relationships between new and existing nodes.
        Only creates relationships when there's a strong, meaningful connection.
        """
        # Convert schema nodes to LLM nodes
        llm_nodes = [
            LLMNode(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids,
                book_id=node.book_id,
                highlight_id=node.highlight_id,
                writing_id=node.writing_id,
                discipline=node.discipline,
                bloom_level=node.bloom_level,
                confidence=node.confidence
            ) for node in new_nodes
        ]
        llm_relationships, _ = await get_relationships(llm_nodes, existing_context)  # Ignore the ID mapping
        # Convert LLM relationships to schema relationships
        return [Relationship(source=rel.source, target=rel.target, relation=rel.relation) for rel in llm_relationships]

    async def generate_book_scoped_relationships(self, new_nodes: List[Node], book_id: int) -> List[Relationship]:
        """
        Generate relationships between new nodes and existing nodes from the SAME book
        using vector similarity search filtered by book_id.

        This enables connections between reading progress data and highlights from the same source.

        Args:
            new_nodes: Newly extracted nodes
            book_id: Book ID to filter by

        Returns:
            List of relationships between new and existing same-book nodes
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        relationships = []

        # Generate embeddings for new nodes if not already present
        nodes_needing_embeddings = [node for node in new_nodes if not hasattr(node, 'embedding') or not node.embedding]
        if nodes_needing_embeddings:
            node_texts = [node.name for node in nodes_needing_embeddings]
            embeddings = generate_embeddings(node_texts)
            for node, embedding in zip(nodes_needing_embeddings, embeddings):
                node.embedding = embedding

        # For each new node, find related nodes from the same book
        for node in new_nodes:
            if not hasattr(node, 'embedding') or not node.embedding:
                continue

            try:
                # Vector search filtered by book_id
                similar_nodes = await self.graph_ops.neo4j_manager.query_text_similarity(
                    keyword_embedding=node.embedding,
                    user_id=self.user_id,
                    book_id=book_id,  # Key: filter to same book only
                    limit=10  # Higher limit since we're pre-filtering by book
                )

                if not similar_nodes:
                    continue

                # Filter to meaningful similarity (lower threshold for same book)
                # Also exclude the node itself if it was already created
                relevant_nodes = [
                    n for n in similar_nodes
                    if n.get('score', 0) >= 0.70 and n.get('nodeName') != node.name
                ]

                if not relevant_nodes:
                    continue

                # Format context for LLM to generate appropriate relationships
                context = self._format_nodes_for_context(relevant_nodes)

                # Use LLM to generate relationships with the same-book context
                node_relationships = await self.generate_cross_relationships([node], context)
                relationships.extend(node_relationships)

                logger.debug(
                    f"Found {len(node_relationships)} book-scoped relationships for node '{node.name}' "
                    f"(book_id: {book_id}, similar nodes: {len(relevant_nodes)})"
                )

            except Exception as e:
                logger.warning(f"Error finding book-scoped relationships for node '{node.name}': {e}")
                continue

        return relationships

    def _format_nodes_for_context(self, nodes: List[Dict[str, Any]]) -> str:
        """
        Format a list of similar nodes into context string for LLM relationship generation.

        Args:
            nodes: List of node dicts from vector search (with 'nodeName', 'score', etc.)

        Returns:
            Formatted context string
        """
        if not nodes:
            return ""

        context_parts = ["Existing related nodes from the same source:\n"]
        for i, node in enumerate(nodes, 1):
            node_name = node.get('nodeName', 'Unknown')
            score = node.get('score', 0.0)
            context_parts.append(f"{i}. {node_name} (similarity: {score:.2f})")

        return "\n".join(context_parts)

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
        nodes_for_llm = [LLMNode(name=node.name, type="Unknown", chunk_ids=[], discipline="", bloom_level="", confidence=0.0, book_id=[], highlight_id=[], writing_id=[]) for node in existing_nodes]  # Add required type field
        
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

    async def _consolidate_duplicates_inline(self):
        """
        Automatically consolidate duplicate nodes after ingestion.
        This runs inline as part of the ingestion process.
        """
        if self.graph_ops is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")
            
        try:
            logger.info("Running automatic duplicate consolidation...")

            report = await self.graph_ops.deduplicator.consolidate_duplicate_nodes(
                user_id=self.user_id,
                dry_run=False  # Actually consolidate
            )

            if report["duplicate_clusters"]:
                logger.info(
                    f"Consolidated {report['nodes_to_remove']} duplicate nodes "
                    f"across {len(report['duplicate_clusters'])} clusters"
                )
            else:
                logger.debug("No duplicate nodes found during consolidation")

        except Exception as e:
            # Don't fail the entire ingestion if consolidation has issues
            logger.warning(f"Duplicate consolidation failed (non-fatal): {e}")

    async def close(self):
        await self.__aexit__(None, None, None)



    
