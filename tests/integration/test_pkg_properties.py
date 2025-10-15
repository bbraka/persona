"""
Integration tests for PKG (Personal Knowledge Graph) properties handling.
Tests that discipline and confidence are preserved while bloom_level is calculated from topology.
"""
import pytest
from typing import Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock
from persona.core.graph_ops import GraphOps
from persona.models.schema import NodeModel, NodesAndRelationshipsResponse, RelationshipModel


@pytest.mark.asyncio
async def test_pkg_properties_preserved_for_new_nodes():
    """
    Test that discipline and confidence from LLM are preserved,
    while bloom_level is calculated from graph topology.
    """
    # Setup: Create nodes with PKG properties
    nodes_with_pkg = [
        NodeModel(
            name="Edmond Dantès",
            type="Character",
            properties={
                "discipline": "Literature",
                "confidence": 0.92,
                "bloom_level": "Remember"  # LLM-extracted, will be overwritten
            },
            embedding=[0.1, 0.2, 0.3]
        ),
        NodeModel(
            name="Count of Monte Cristo",
            type="Literary Work",
            properties={
                "discipline": "Literature",
                "confidence": 0.95,
                "bloom_level": "Understand"  # LLM-extracted, will be overwritten
            },
            embedding=[0.4, 0.5, 0.6]
        )
    ]
    
    relationships = [
        RelationshipModel(
            source="Edmond Dantès",
            target="Count of Monte Cristo",
            relation="PROTAGONIST_OF"
        )
    ]
    
    graph_update = NodesAndRelationshipsResponse(
        nodes=nodes_with_pkg,
        relationships=relationships
    )
    
    # Mock Neo4j manager
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        
        # Mock user exists
        mock_manager.user_exists = AsyncMock(return_value=True)
        
        # Mock get_node_relationships to return empty (new nodes have no prior relationships)
        mock_manager.get_node_relationships = AsyncMock(return_value=[])
        
        # Mock driver for calculate_bloom_level
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{
            'degree': 1,  # 1 relationship
            'contexts': ['Literary Work']  # 1 context
        }])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_manager.driver = MagicMock()
        mock_manager.driver.session = MagicMock(return_value=mock_session)
        mock_manager.driver.__aenter__ = AsyncMock(return_value=mock_manager.driver)
        mock_manager.driver.__aexit__ = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        # Capture what gets passed to update_graph_transactional
        captured_bloom_updates: list[dict[str, Any]] | None = None
        
        async def capture_transaction_call(nodes, relationships, embeddings_data, bloom_updates, user_id):
            nonlocal captured_bloom_updates
            captured_bloom_updates = bloom_updates
        
        mock_manager.update_graph_transactional = AsyncMock(side_effect=capture_transaction_call)
        
        # Execute
        async with GraphOps(mock_manager) as graph_ops:
            await graph_ops.update_graph_with_bloom_transactional(graph_update, "test_user")
        
        # Verify
        assert isinstance(captured_bloom_updates, list), "bloom_updates should be a list"
        assert len(captured_bloom_updates) == 2, "Should have bloom updates for both nodes"

        # Check first node (Edmond Dantès)
        edmond_update = next(u for u in captured_bloom_updates if u["node_name"] == "Edmond Dantès")
        assert edmond_update["properties"]["discipline"] == "Literature", "Discipline should be preserved"
        assert edmond_update["properties"]["confidence"] == 0.92, "Confidence should be preserved"
        assert edmond_update["properties"]["bloom_level"] == "Understand", "Bloom level should be calculated (degree=1 → Understand)"

        # Check second node (Count of Monte Cristo)
        book_update = next(u for u in captured_bloom_updates if u["node_name"] == "Count of Monte Cristo")
        assert book_update["properties"]["discipline"] == "Literature", "Discipline should be preserved"
        assert book_update["properties"]["confidence"] == 0.95, "Confidence should be preserved"
        assert book_update["properties"]["bloom_level"] == "Understand", "Bloom level should be calculated"


@pytest.mark.asyncio
async def test_existing_node_properties_preserved_when_neighbor_added():
    """
    Test that when a new node is added, its neighboring existing nodes
    have their properties preserved during bloom recalculation.
    """
    # Setup: New node to be added
    new_node = NodeModel(
        name="Abbé Faria",
        type="Character",
        properties={
            "discipline": "Literature",
            "confidence": 0.88,
            "bloom_level": "Remember"
        },
        embedding=[0.7, 0.8, 0.9]
    )
    
    new_relationship = RelationshipModel(
        source="Abbé Faria",
        target="Edmond Dantès",
        relation="MENTORS"
    )
    
    graph_update = NodesAndRelationshipsResponse(
        nodes=[new_node],
        relationships=[new_relationship]
    )
    
    # Mock Neo4j manager
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        
        mock_manager.user_exists = AsyncMock(return_value=True)
        
        # Mock get_node_relationships to return the new relationship
        async def mock_get_relationships(node_name, user_id):
            if node_name == "Abbé Faria":
                return [{"source": "Abbé Faria", "target": "Edmond Dantès", "relation": "MENTORS"}]
            return []
        
        mock_manager.get_node_relationships = AsyncMock(side_effect=mock_get_relationships)
        
        # Mock get_node_data to return existing node with properties
        async def mock_get_node_data(node_name, user_id):
            if node_name == "Edmond Dantès":
                return {
                    "name": "Edmond Dantès",
                    "type": "Character",
                    "properties": {
                        "discipline": "Literature",  # Existing property
                        "confidence": 0.92,  # Existing property
                        "bloom_level": "Understand"  # Will be recalculated
                    }
                }
            return None
        
        mock_manager.get_node_data = AsyncMock(side_effect=mock_get_node_data)
        
        # Mock driver for calculate_bloom_level
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        
        # Return different bloom data based on node
        async def mock_run(query, **params):
            result = AsyncMock()
            if params.get('node_name') == "Abbé Faria":
                result.data = AsyncMock(return_value=[{'degree': 1, 'contexts': ['Character']}])
            else:  # Edmond Dantès now has more connections
                result.data = AsyncMock(return_value=[{'degree': 2, 'contexts': ['Character']}])
            return result
        
        mock_session.run = AsyncMock(side_effect=mock_run)
        mock_manager.driver = MagicMock()
        mock_manager.driver.session = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        # Capture bloom updates
        captured_bloom_updates: list[dict[str, Any]] | None = None
        
        async def capture_transaction_call(nodes, relationships, embeddings_data, bloom_updates, user_id):
            nonlocal captured_bloom_updates
            captured_bloom_updates = bloom_updates
        
        mock_manager.update_graph_transactional = AsyncMock(side_effect=capture_transaction_call)
        
        # Execute
        graph_ops = GraphOps(mock_manager)
        await graph_ops.initialize()
        await graph_ops.update_graph_with_bloom_transactional(graph_update, "test_user")
        
        # Verify
        assert isinstance(captured_bloom_updates, list), "bloom_updates should be a list"

        # Find updates for both nodes
        faria_update = next((u for u in captured_bloom_updates if u["node_name"] == "Abbé Faria"), None)
        edmond_update = next((u for u in captured_bloom_updates if u["node_name"] == "Edmond Dantès"), None)
        
        # New node should have its properties preserved
        assert faria_update is not None, "New node should have bloom update"
        assert faria_update["properties"]["discipline"] == "Literature"
        assert faria_update["properties"]["confidence"] == 0.88
        assert faria_update["properties"]["bloom_level"] == "Understand"  # Calculated from degree=1
        
        # Existing node should have its properties preserved from DB
        assert edmond_update is not None, "Existing node should have bloom update"
        assert edmond_update["properties"]["discipline"] == "Literature", "Existing discipline should be preserved"
        assert edmond_update["properties"]["confidence"] == 0.92, "Existing confidence should be preserved"
        assert edmond_update["properties"]["bloom_level"] == "Understand", "Bloom level recalculated (degree=2)"


@pytest.mark.asyncio
async def test_bloom_level_calculation_topology_based():
    """
    Test that bloom_level is calculated based on graph topology (degree and contexts),
    not from LLM extraction.
    """
    # Mock Neo4j manager
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        
        mock_manager.user_exists = AsyncMock(return_value=True)
        
        # Mock driver
        mock_session = AsyncMock()
        mock_manager.driver = MagicMock()
        mock_manager.driver.session = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        
        graph_ops = GraphOps(mock_manager)
        await graph_ops.initialize()
        
        # Test case 1: Low connectivity → Remember
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{'degree': 0, 'contexts': []}])
        mock_session.run = AsyncMock(return_value=mock_result)
        
        bloom = await graph_ops.calculate_bloom_level("isolated_node", "test_user")
        assert bloom == "Remember", "Isolated node should be 'Remember'"
        
        # Test case 2: Medium connectivity → Understand
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{'degree': 2, 'contexts': ['Type1']}])
        mock_session.run = AsyncMock(return_value=mock_result)
        
        bloom = await graph_ops.calculate_bloom_level("connected_node", "test_user")
        assert bloom == "Understand", "Node with degree=2 should be 'Understand'"
        
        # Test case 3: High connectivity + contexts → Apply
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{'degree': 4, 'contexts': ['Type1', 'Type2']}])
        mock_session.run = AsyncMock(return_value=mock_result)
        
        bloom = await graph_ops.calculate_bloom_level("well_connected_node", "test_user")
        assert bloom == "Apply", "Node with degree=4 and 2 contexts should be 'Apply'"
        
        # Test case 4: Very high connectivity + many contexts → Analyze
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{'degree': 8, 'contexts': ['Type1', 'Type2', 'Type3', 'Type4']}])
        mock_session.run = AsyncMock(return_value=mock_result)
        
        bloom = await graph_ops.calculate_bloom_level("hub_node", "test_user")
        assert bloom == "Analyze", "Hub node with degree=8 and 4 contexts should be 'Analyze'"


@pytest.mark.asyncio
async def test_empty_properties_handling():
    """
    Test that nodes without PKG properties are handled gracefully.
    """
    node_without_props = NodeModel(
        name="Generic Node",
        type="Concept",
        properties={},  # No PKG properties
        embedding=[0.1, 0.2]
    )
    
    graph_update = NodesAndRelationshipsResponse(
        nodes=[node_without_props],
        relationships=[]
    )
    
    with patch('persona.core.neo4j_database.Neo4jConnectionManager') as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        
        mock_manager.user_exists = AsyncMock(return_value=True)
        mock_manager.get_node_relationships = AsyncMock(return_value=[])
        
        # Mock driver
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{'degree': 0, 'contexts': []}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_manager.driver = MagicMock()
        mock_manager.driver.session = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        captured_bloom_updates: list[dict[str, Any]] | None = None

        async def capture_transaction_call(nodes, relationships, embeddings_data, bloom_updates, user_id):
            nonlocal captured_bloom_updates
            captured_bloom_updates = bloom_updates
        
        mock_manager.update_graph_transactional = AsyncMock(side_effect=capture_transaction_call)
        
        async with GraphOps(mock_manager) as graph_ops:
            await graph_ops.update_graph_with_bloom_transactional(graph_update, "test_user")
        
        # Verify node without properties gets bloom_level added
        assert isinstance(captured_bloom_updates, list), "bloom_updates should be a list"
        assert len(captured_bloom_updates) == 1

        update = captured_bloom_updates[0]
        assert update["node_name"] == "Generic Node"
        assert "bloom_level" in update["properties"]
        assert update["properties"]["bloom_level"] == "Remember"
