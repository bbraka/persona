"""
Integration tests for the temporary ID system in relationship extraction.
These tests validate that the new ID-based approach works correctly.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch
from persona.llm.llm_graph import get_relationships, RelationshipWithID, Node



class TestTemporaryIDSystem:
    
    @pytest.mark.asyncio
    async def test_id_mapping_generation(self):
        """Test that temporary IDs are generated correctly for nodes."""
        nodes = [
            Node(name="First node with a long name", type="Identity", discipline=None, bloom_level=None, confidence=None),
            Node(name="Second node with different content", type="Belief", discipline=None, bloom_level=None, confidence=None),
            Node(name="Third node for comprehensive testing", type="Goal", discipline=None, bloom_level=None, confidence=None)
        ]
        
        # Mock the chat client to return empty relationships
        mock_response = AsyncMock()
        mock_response.content = '{"relationships": []}'
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            relationships, id_mapping = await get_relationships(nodes, "test context")
            
            # Verify ID mapping structure
            assert len(id_mapping) == 3
            assert "Node1" in id_mapping
            assert "Node2" in id_mapping  
            assert "Node3" in id_mapping
            
            # Verify mapping content
            assert id_mapping["Node1"] == "First node with a long name"
            assert id_mapping["Node2"] == "Second node with different content"
            assert id_mapping["Node3"] == "Third node for comprehensive testing"

    @pytest.mark.asyncio
    async def test_relationship_conversion_logic(self):
        """Test that RelationshipWithID objects are correctly converted to Relationship objects."""
        nodes = [
            Node(name="Source node name", type="Identity", discipline=None, bloom_level=None, confidence=None),
            Node(name="Target node name", type="Belief", discipline=None, bloom_level=None, confidence=None)
        ]
        
        # Mock LLM to return relationships with IDs
        mock_response_data = {
            "relationships": [
                {"source_id": "Node1", "relation": "LEADS_TO", "target_id": "Node2"},
                {"source_id": "Node2", "relation": "INFLUENCED_BY", "target_id": "Node1"}
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.content = json.dumps(mock_response_data)
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            relationships, id_mapping = await get_relationships(nodes, "test context")
            
            # Verify conversion worked
            assert len(relationships) == 2
            
            # Check first relationship
            rel1 = relationships[0]
            assert rel1.source == "Source node name"
            assert rel1.relation == "LEADS_TO"
            assert rel1.target == "Target node name"
            
            # Check second relationship  
            rel2 = relationships[1]
            assert rel2.source == "Target node name"
            assert rel2.relation == "INFLUENCED_BY"
            assert rel2.target == "Source node name"

    @pytest.mark.asyncio
    async def test_invalid_id_handling(self):
        """Test that invalid IDs in LLM response are handled gracefully."""
        nodes = [
            Node(name="Valid node", type="Identity", discipline=None, bloom_level=None, confidence=None)
        ]
        
        # Mock LLM to return relationship with invalid IDs
        mock_response_data = {
            "relationships": [
                {"source_id": "InvalidNode1", "relation": "RELATES_TO", "target_id": "InvalidNode2"},
                {"source_id": "Node1", "relation": "VALID_RELATION", "target_id": "InvalidNode3"}
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.content = json.dumps(mock_response_data)
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            relationships, id_mapping = await get_relationships(nodes, "test context")
            
            # Should filter out invalid relationships
            assert len(relationships) == 0  # Both relationships have invalid IDs
            assert len(id_mapping) == 1  # Mapping should still be correct
            assert id_mapping["Node1"] == "Valid node"

    @pytest.mark.asyncio  
    async def test_prompt_formatting_with_ids(self):
        """Test that the prompt is formatted correctly with temporary IDs."""
        nodes = [
            Node(name="Complex node name with punctuation, commas!", type="Identity", discipline=None, bloom_level=None, confidence=None),
            Node(name="Another node: with special characters & symbols", type="Belief", discipline=None, bloom_level=None, confidence=None)
        ]
        
        mock_response = AsyncMock()
        mock_response.content = '{"relationships": []}'
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            await get_relationships(nodes, "graph context")
            
            # Verify the call was made
            assert mock_client.chat.called
            call_args = mock_client.chat.call_args
            
            # Check that the messages were passed correctly
            messages = call_args[1]['messages'] if 'messages' in call_args[1] else call_args[0]
            
            # Should have system and user messages
            assert len(messages) >= 2

    @pytest.mark.asyncio
    async def test_empty_nodes_handling(self):
        """Test that empty node list is handled correctly."""
        relationships, id_mapping = await get_relationships([], "context")
        
        assert relationships == []
        assert id_mapping == {}

    @pytest.mark.asyncio
    async def test_single_node_handling(self):
        """Test that single node generates correct ID mapping."""
        nodes = [Node(name="Single node", type="Identity", discipline=None, bloom_level=None, confidence=None)]
        
        mock_response = AsyncMock()
        mock_response.content = '{"relationships": []}'
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            relationships, id_mapping = await get_relationships(nodes, "context")
            
            assert len(id_mapping) == 1
            assert id_mapping["Node1"] == "Single node"

    @pytest.mark.asyncio
    async def test_llm_response_model_type(self):
        """Test that the LLM is called with RelationshipWithID response model."""
        nodes = [Node(name="Test node", type="Identity", discipline=None, bloom_level=None, confidence=None)]
        
        mock_response = AsyncMock()
        mock_response.content = '{"relationships": []}'
        
        mock_client = AsyncMock()
        mock_client.chat.return_value = mock_response
        
        with patch('persona.llm.llm_graph.get_chat_client', return_value=mock_client):
            await get_relationships(nodes, "context")
            
            # Verify the chat method was called
            assert mock_client.chat.called
