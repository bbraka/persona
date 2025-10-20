from persona.core.graph_ops import GraphOps
from persona.models.schema import CustomGraphUpdate, CustomNodeData, CustomRelationshipData, NodesAndRelationshipsResponse
from persona.models.schema import NodeModel, RelationshipModel
from typing import Dict, Any
    
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
                # Handle chunk_id (legacy) or chunk_ids (new) from CustomNodeData
                chunk_ids_list = []
                if hasattr(node, 'chunk_ids') and node.chunk_ids:
                    chunk_ids_list = node.chunk_ids
                elif hasattr(node, 'chunk_id') and node.chunk_id:
                    chunk_ids_list = [node.chunk_id]

                nodes.append(NodeModel(
                    name=node.name,
                    type=node.type,
                    chunk_ids=chunk_ids_list,  # Always use array
                    properties=node.properties
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