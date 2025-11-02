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

    async def _generate_all_relationships(
        self,
        nodes: List[Node],
        text: str,
        book_ids: set[int]
    ) -> List[Relationship]:
        """
        Generate all types of relationships for the given nodes.

        Args:
            nodes: List of nodes to generate relationships for
            text: Original text for context retrieval
            book_ids: Set of book IDs for book-scoped relationships

        Returns:
            List of all relationships
        """
        relationships = []

        # Get existing graph context
        existing_context = await self.graph_context_retriever.get_rich_context(text, self.user_id)

        # Phase 1: Core relationships between new nodes
        new_node_relationships = await self.generate_relationships(nodes)
        relationships.extend(new_node_relationships)

        # Phase 2: Connect with existing nodes
        if existing_context and len(nodes) > 0:
            mixed_relationships = await self.generate_cross_relationships(nodes, existing_context)
            relationships.extend(mixed_relationships)

        # Phase 3: Book-scoped relationships
        for book_id in book_ids:
            book_relationships = await self.generate_book_scoped_relationships(nodes, book_id)
            relationships.extend(book_relationships)
            logger.info(f"Created {len(book_relationships)} book-scoped relationships for book {book_id}")

        return relationships

    def _nodes_to_node_models(self, nodes: List[Node], embeddings: List[List[float]]) -> List[NodeModel]:
        """
        Convert schema Nodes to NodeModels with embeddings.

        Args:
            nodes: List of schema Node objects
            embeddings: List of embedding vectors

        Returns:
            List of NodeModel objects ready for database
        """
        from persona.models.schema import NodeModel

        node_models = []
        for node, embedding in zip(nodes, embeddings):
            properties = {}
            if node.discipline:
                properties["discipline"] = node.discipline
            if node.bloom_level:
                properties["bloom_level"] = node.bloom_level
            if node.confidence is not None:
                properties["confidence"] = node.confidence

            # Convert datetime to ISO string
            created_at_str = None
            if hasattr(node, 'created_at') and node.created_at:
                created_at_str = node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at)

            # Convert BloomLevelUpdate objects to dicts
            bloom_history_dicts = []
            if hasattr(node, 'bloom_history') and node.bloom_history:
                for update in node.bloom_history:
                    bloom_history_dicts.append({
                        "level": update.level,
                        "timestamp": update.timestamp.isoformat() if hasattr(update.timestamp, 'isoformat') else str(update.timestamp),
                        "source": update.source
                    })

            node_models.append(NodeModel(
                name=node.name,
                type=node.type,
                chunk_ids=node.chunk_ids if node.chunk_ids else [],
                book_id=node.book_id if node.book_id else [],
                highlight_id=node.highlight_id if node.highlight_id else [],
                writing_id=node.writing_id if node.writing_id else [],
                properties=properties,
                embedding=embedding,
                created_at=created_at_str,
                bloom_history=bloom_history_dicts
            ))

        return node_models

    async def _save_graph_update(self, nodes: List[Node], relationships: List[Relationship]):
        """
        Generate embeddings and save nodes and relationships to graph.

        Args:
            nodes: List of schema Node objects
            relationships: List of Relationship objects
        """
        from persona.models.schema import RelationshipModel, NodesAndRelationshipsResponse

        # Generate embeddings
        node_texts = [node.name for node in nodes]
        embeddings = generate_embeddings(node_texts)

        # Convert to NodeModels
        node_models = self._nodes_to_node_models(nodes, embeddings)

        # Convert relationships
        relationship_models = [RelationshipModel(
            source=rel.source,
            target=rel.target,
            relation=rel.relation
        ) for rel in relationships]

        graph_update = NodesAndRelationshipsResponse(
            nodes=node_models,
            relationships=relationship_models
        )

        # Save to database
        try:
            await self.graph_ops.update_graph_with_bloom_transactional(graph_update, self.user_id)
            logger.info(f"Successfully saved {len(node_models)} nodes and {len(relationship_models)} relationships")
        except Exception as e:
            logger.error(f"Failed to save graph update (transaction rolled back): {e}")
            raise

    def _llm_node_to_schema_node(self, llm_node, metadata: Dict[str, Any]) -> Node:
        """
        Convert LLM node to schema Node with metadata applied.

        Args:
            llm_node: Node from LLM response
            metadata: Metadata dict to apply to node

        Returns:
            Schema Node object
        """
        from datetime import datetime
        from persona.models.schema import BloomLevelUpdate, Node

        # Extract entity IDs from metadata
        book_ids = [int(metadata['book_id'])] if 'book_id' in metadata else []
        highlight_ids = [int(metadata['highlight_id'])] if 'highlight_id' in metadata else []
        writing_ids = [int(metadata['writing_id'])] if 'writing_id' in metadata else []

        # Extract date from metadata
        annotation_date = None
        if 'date' in metadata:
            try:
                date_value = metadata['date']
                if isinstance(date_value, str):
                    annotation_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                elif isinstance(date_value, datetime):
                    annotation_date = date_value
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date from metadata: {e}")

        # Create Bloom history
        current_time = datetime.utcnow()
        bloom_level = getattr(llm_node, 'bloom_level', '')
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

        return Node(
            name=llm_node.name,
            type=llm_node.type,
            chunk_ids=getattr(llm_node, 'chunk_ids', []),
            book_id=book_ids,
            highlight_id=highlight_ids,
            writing_id=writing_ids,
            discipline=getattr(llm_node, 'discipline', ''),
            bloom_level=bloom_level,
            confidence=getattr(llm_node, 'confidence', 0.0),
            created_at=annotation_date or current_time,
            bloom_history=bloom_history
        )

    async def ingest_batch_unstructured_data_to_graph(self, data_items: List[UnstructuredData]):
        """
        Ingest batch of unstructured data using source indexing for accurate metadata mapping.

        Args:
            data_items: List of UnstructuredData items to ingest
        """
        if self.graph_ops is None or self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        if not data_items:
            logger.info("No data items provided for batch ingestion")
            return

        logger.info(f"Starting batch ingestion of {len(data_items)} items")

        # Format batch with source indices
        formatted_text, metadata_mapping = self.preprocess_batch_data(data_items)

        # Extract nodes (single LLM call)
        graph_context = await self.get_relevant_graph_context(user_id=self.user_id, nodes=[])
        llm_nodes = await get_nodes(formatted_text, graph_context)

        if not llm_nodes:
            logger.info("No nodes extracted from batch data")
            return

        # Map source_index to metadata and convert to schema Nodes
        schema_nodes = []
        for llm_node in llm_nodes:
            source_idx = getattr(llm_node, 'source_index', None)

            # Get metadata for this source
            if source_idx is not None and isinstance(source_idx, int) and source_idx in metadata_mapping:
                metadata = metadata_mapping[source_idx]
            elif source_idx is None:
                logger.warning(f"Node '{llm_node.name}' missing source_index, using first source")
                metadata = metadata_mapping[0] if 0 in metadata_mapping else {}
            else:
                logger.error(f"Node '{llm_node.name}' has invalid source_index {source_idx}, skipping")
                continue

            schema_nodes.append(self._llm_node_to_schema_node(llm_node, metadata))

        logger.info(f"Extracted {len(schema_nodes)} nodes from batch")

        # Collect book IDs for book-scoped relationships
        book_ids = set()
        for item in data_items:
            if item.metadata and 'book_id' in item.metadata:
                try:
                    book_ids.add(int(item.metadata['book_id']))
                except (ValueError, TypeError):
                    pass

        # Generate relationships
        relationships = await self._generate_all_relationships(schema_nodes, formatted_text, book_ids)

        # Save to database
        await self._save_graph_update(schema_nodes, relationships)

    async def ingest_unstructured_data_to_graph(self, data: UnstructuredData):
        """
        Ingest single unstructured data item into the graph.
        """
        if self.graph_ops is None or self.graph_context_retriever is None:
            raise RuntimeError("GraphConstructor must be used as an async context manager")

        text = self.preprocess_data(data)

        # Extract nodes from content
        new_nodes = await self.extract_nodes(text, data.metadata or {})
        if not new_nodes:
            logger.info("No nodes generated from unstructured data")
            return

        # Collect book IDs for book-scoped relationships
        book_ids = set()
        if data.metadata and 'book_id' in data.metadata:
            try:
                book_ids.add(int(data.metadata['book_id']))
            except (ValueError, TypeError):
                pass

        # Generate relationships
        relationships = await self._generate_all_relationships(new_nodes, text, book_ids)

        # Save to database
        await self._save_graph_update(new_nodes, relationships)

    def preprocess_data(self, data: UnstructuredData) -> str:
        """
        Preprocess the data, combine relevant fields into a single string.
        """
        preprocessed = f"{data.title}\n{data.content}\n"
        if data.metadata:
            preprocessed += "\n".join([f"{k}: {v}" for k, v in data.metadata.items()])
        return preprocessed

    def preprocess_batch_data(self, data_items: List[UnstructuredData]) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Preprocess batch data with indexed source formatting.

        Args:
            data_items: List of UnstructuredData items to process

        Returns:
            Tuple of (formatted_text, metadata_mapping)
            - formatted_text: Text with indexed sources "Source [0]:", "Source [1]:", etc.
            - metadata_mapping: Dict mapping source index to original metadata
        """
        text_parts = []
        metadata_mapping = {}

        for idx, data in enumerate(data_items):
            # Format source with index
            source_text = f"Source [{idx}]:\n"
            source_text += f"Title: {data.title}\n"
            source_text += f"Content: {data.content}\n"

            # Include metadata in the text for LLM context (optional)
            if data.metadata:
                source_text += "Metadata:\n"
                for k, v in data.metadata.items():
                    source_text += f"  {k}: {v}\n"

            text_parts.append(source_text)

            # Store metadata mapping for later
            metadata_mapping[idx] = data.metadata or {}

        formatted_text = "\n\n".join(text_parts)
        return formatted_text, metadata_mapping

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
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert highlight_id from metadata: {e}")

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

    def map_source_indices_to_metadata(
        self,
        nodes: List[Node],
        metadata_mapping: Dict[int, Dict[str, Any]]
    ) -> Tuple[List[Node], Dict[str, int]]:
        """
        Map source_index from LLM response to actual metadata from original sources.

        Args:
            nodes: List of nodes with source_index from LLM
            metadata_mapping: Dict mapping source index to original metadata

        Returns:
            Tuple of (validated_nodes, stats)
            - validated_nodes: Nodes with metadata applied from source_index mapping
            - stats: Dict with validation metrics (valid, missing_index, invalid_index)
        """
        from datetime import datetime

        validated_nodes = []
        stats = {"valid": 0, "missing_index": 0, "invalid_index": 0, "cross_source": 0}

        for node in nodes:
            source_idx = getattr(node, 'source_index', None)

            # Handle missing source_index
            if source_idx is None:
                logger.warning(f"Node '{node.name}' missing source_index, assigning to first source")
                stats["missing_index"] += 1
                source_idx = 0 if 0 in metadata_mapping else None

                if source_idx is None:
                    logger.error(f"Node '{node.name}' missing source_index and no sources available, skipping")
                    continue

            # Handle single source_index
            if isinstance(source_idx, int):
                if source_idx not in metadata_mapping:
                    logger.error(f"Node '{node.name}' has invalid source_index {source_idx}, skipping")
                    stats["invalid_index"] += 1
                    continue

                # Apply metadata from source
                metadata = metadata_mapping[source_idx]
                node.book_id = [int(metadata['book_id'])] if 'book_id' in metadata else []
                node.highlight_id = [int(metadata['highlight_id'])] if 'highlight_id' in metadata else []
                node.writing_id = [int(metadata['writing_id'])] if 'writing_id' in metadata else []

                # Update bloom history source if needed
                if hasattr(node, 'bloom_history') and node.bloom_history:
                    for bloom_update in node.bloom_history:
                        if bloom_update.source is None:
                            source_info = []
                            if node.highlight_id:
                                source_info.append(f"highlight_id:{node.highlight_id[0]}")
                            elif node.book_id:
                                source_info.append(f"book_id:{node.book_id[0]}")
                            bloom_update.source = ', '.join(source_info) if source_info else None

                stats["valid"] += 1

            # Handle multi-source cross-concept (source_index is array)
            elif isinstance(source_idx, list):
                book_ids = []
                highlight_ids = []
                writing_ids = []

                for idx in source_idx:
                    if idx not in metadata_mapping:
                        logger.warning(f"Node '{node.name}' has invalid source_index {idx} in list, skipping this index")
                        continue

                    metadata = metadata_mapping[idx]
                    if 'book_id' in metadata:
                        book_id = int(metadata['book_id'])
                        if book_id not in book_ids:
                            book_ids.append(book_id)
                    if 'highlight_id' in metadata:
                        highlight_id = int(metadata['highlight_id'])
                        if highlight_id not in highlight_ids:
                            highlight_ids.append(highlight_id)
                    if 'writing_id' in metadata:
                        writing_id = int(metadata['writing_id'])
                        if writing_id not in writing_ids:
                            writing_ids.append(writing_id)

                if not (book_ids or highlight_ids or writing_ids):
                    logger.error(f"Node '{node.name}' has all invalid source_indices {source_idx}, skipping")
                    stats["invalid_index"] += 1
                    continue

                node.book_id = book_ids
                node.highlight_id = highlight_ids
                node.writing_id = writing_ids
                stats["valid"] += 1
                stats["cross_source"] += 1

            validated_nodes.append(node)

        # Log accuracy metrics
        total = sum([stats["valid"], stats["missing_index"], stats["invalid_index"]])
        accuracy = stats["valid"] / total if total > 0 else 0
        logger.info(
            f"Source index mapping: valid={stats['valid']}, missing={stats['missing_index']}, "
            f"invalid={stats['invalid_index']}, cross_source={stats['cross_source']}, "
            f"accuracy={accuracy:.1%}"
        )

        return validated_nodes, stats

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

        # Generate embeddings for all new nodes
        node_texts = [node.name for node in new_nodes]
        embeddings = generate_embeddings(node_texts)

        # For each new node, find related nodes from the same book
        for idx, (node, embedding) in enumerate(zip(new_nodes, embeddings)):
            if not embedding:
                continue

            try:
                # Vector search filtered by book_id
                similar_nodes = await self.graph_ops.neo4j_manager.query_text_similarity(
                    keyword_embedding=embedding,
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



    
