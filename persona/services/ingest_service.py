from persona.core.constructor import GraphConstructor
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.models.schema import UnstructuredData
from typing import List, Union

class IngestService:
    @staticmethod
    async def ingest_data(user_id: str, data: Union[UnstructuredData, List[UnstructuredData]], graph_ops: GraphOps):
        """
        Ingest unstructured data into the graph.
        Accepts either a single UnstructuredData or a list of UnstructuredData items.

        Args:
            user_id: User ID
            data: Single UnstructuredData or List[UnstructuredData]
            graph_ops: GraphOps instance

        Returns:
            Success message with count of items processed
        """
        constructor = GraphConstructor(user_id)
        constructor.graph_ops = graph_ops
        constructor.graph_context_retriever = GraphContextRetriever(graph_ops)

        # Handle both single item and array
        if isinstance(data, list):
            await constructor.ingest_batch_unstructured_data_to_graph(data)
            return {"message": f"Successfully ingested {len(data)} data items"}
        else:
            await constructor.ingest_unstructured_data_to_graph(data)
            return {"message": "Data ingested successfully"}