from persona.core.graph_ops import GraphOps
from persona.models.schema import CustomGraphUpdate, CustomNodeData, CustomRelationshipData, NodesAndRelationshipsResponse
from persona.models.schema import NodeModel, RelationshipModel
from typing import Dict, Any
from server.logging_config import get_logger

logger = get_logger(__name__)

class CustomDataService:
    def __init__(self, graph_ops: GraphOps):
        self.graph_ops = graph_ops
    
    async def update_custom_data(self, user_id: str, update: CustomGraphUpdate) -> Dict[str, Any]:
        """
        Add or update custom structured data using existing GraphOps
        """
        try:
            # Convert CustomNodeData to NodeModel
            nodes = []
            for node in update.nodes:
                # Merge perspective into properties if it exists
                node_properties = node.properties.copy() if node.properties else {}
                if node.perspective:
                    node_properties['perspective'] = node.perspective

                # Extract entity IDs from properties if present
                book_id = node_properties.pop('book_id', [])
                highlight_id = node_properties.pop('highlight_id', [])
                writing_id = node_properties.pop('writing_id', [])

                nodes.append(NodeModel(
                    name=node.name,
                    type=node.type,
                    chunk_ids=node.chunk_ids if node.chunk_ids else [],  # Always use array
                    book_id=book_id if isinstance(book_id, list) else [],
                    highlight_id=highlight_id if isinstance(highlight_id, list) else [],
                    writing_id=writing_id if isinstance(writing_id, list) else [],
                    properties=node_properties
                ))

            # Convert CustomRelationshipData to RelationshipModel
            relationships = [
                RelationshipModel(
                    source=rel.source,
                    target=rel.target,
                    relation=rel.relation_type,  # Map relation_type to relation
                    properties=rel.data  # Map data to properties
                ) for rel in update.relationships
            ]

            # Use existing GraphOps to update the graph
            await self.graph_ops.update_graph(
                NodesAndRelationshipsResponse(
                    nodes=nodes,
                    relationships=relationships
                ),
                user_id
            )
            return {
                "status": "success",
                "message": f"Updated {len(nodes)} nodes and {len(relationships)} relationships"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }