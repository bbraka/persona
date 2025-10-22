from persona.core.rag_interface import RAGInterface
from persona.models.schema import AskRequest, AskResponse
from persona.core.graph_ops import GraphOps, GraphContextRetriever
from persona.llm.llm_graph import generate_structured_insights
from typing import Dict, Any

class AskService:
    @staticmethod
    async def ask_insights(user_id: str, ask_request: AskRequest, graph_ops: GraphOps) -> AskResponse:
        """
        Generate structured insights based on the requested schema, optionally filtered by entity IDs
        """
        rag = RAGInterface(user_id)
        rag.graph_ops = graph_ops
        rag.graph_context_retriever = GraphContextRetriever(graph_ops)

        # Get context using existing RAG functionality with entity ID filters
        context = await rag.get_context(
            ask_request.query,
            book_id=ask_request.book_id,
            highlight_id=ask_request.highlight_id,
            writing_id=ask_request.writing_id
        )

        structured_response = await generate_structured_insights(ask_request, context)

        # Return the structured response
        return AskResponse(result=structured_response)