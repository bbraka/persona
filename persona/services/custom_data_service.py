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
            nodes = [
                NodeModel(
                    name=node.name,
                    type=node.properties.get("type") if node.properties else None,  # Extract type from properties for Neo4j label
                    perspective=node.perspective,
                    properties=node.properties
                ) for node in update.nodes
            ]
            
            # Convert CustomRelationshipData to RelationshipModel
            relationships = [
                RelationshipModel(
                    source=rel.source,
                    target=rel.target,
                    relation=rel.relation_type  # Map relation_type to relation
                ) for rel in update.relationships
            ]
            
            # Use existing GraphOps to update the graph with custom properties enabled
            await self.graph_ops.update_graph(
                NodesAndRelationshipsResponse(
                    nodes=nodes,
                    relationships=relationships
                ),
                user_id,
                store_custom_properties=True  # Enable storing all custom properties dynamically
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