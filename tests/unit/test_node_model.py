"""
Unit tests for NodeModel in schema.py

Tests that NodeModel properly handles properties with various data types,
especially for PKG fields (discipline, bloom_level, confidence).
"""

import pytest
from persona.models.schema import NodeModel


class TestNodeModelProperties:
    """Test NodeModel properties dict with various data types."""

    def test_node_model_with_string_properties(self):
        """Test NodeModel with traditional string properties."""
        node = NodeModel(
            name="Test node",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Psychology",
                "bloom_level": "Understand"
            }
        )

        assert node.properties is not None
        assert node.properties["discipline"] == "Psychology"
        assert node.properties["bloom_level"] == "Understand"

    def test_node_model_with_float_confidence(self):
        """Test NodeModel accepts float confidence in properties."""
        node = NodeModel(
            name="Test node with float confidence",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Computer Science",
                "bloom_level": "Apply",
                "confidence": 0.95  # Float value
            }
        )

        assert node.properties is not None
        assert node.properties["confidence"] == 0.95

    def test_node_model_with_mixed_types(self):
        """Test NodeModel with mixed data types in properties."""
        node = NodeModel(
            name="Mixed types node",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Physics",  # string
                "bloom_level": "Analyze",  # string
                "confidence": 0.88,  # float
                "count": 42,  # int
                "active": True  # bool
            }
        )

        assert node.properties is not None
        assert isinstance(node.properties["discipline"], str)
        assert isinstance(node.properties["bloom_level"], str)
        assert isinstance(node.properties["confidence"], float)
        assert isinstance(node.properties["count"], int)
        assert isinstance(node.properties["active"], bool)

    def test_node_model_with_integer_confidence(self):
        """Test NodeModel accepts integer values."""
        node = NodeModel(
            name="Node with int",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "confidence": 1,  # int that represents 1.0
                "priority": 5
            }
        )

        assert node.properties is not None
        assert node.properties["confidence"] == 1
        assert node.properties["priority"] == 5

    def test_node_model_empty_properties(self):
        """Test NodeModel with empty properties dict."""
        node = NodeModel(
            name="Empty properties node",
            type="Test",
            embedding=[0.1, 0.2, 0.3]
        )

        assert node.properties == {}

    def test_node_model_none_properties(self):
        """Test NodeModel with None properties stays None when explicitly set."""
        node = NodeModel(
            name="None properties node",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties=None
        )

        # When explicitly set to None, it stays None
        assert node.properties is None

    def test_node_model_all_pkg_fields(self):
        """Test NodeModel with all PKG fields as they come from Node."""
        node = NodeModel(
            name="Complete PKG node",
            type="Concept",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Psychology",
                "bloom_level": "Evaluate",
                "confidence": 0.92
            }
        )

        assert node.properties is not None
        assert len(node.properties) == 3
        assert node.properties["discipline"] == "Psychology"
        assert node.properties["bloom_level"] == "Evaluate"
        assert node.properties["confidence"] == 0.92

    def test_node_model_with_custom_metadata(self):
        """Test NodeModel with custom metadata alongside PKG fields."""
        node = NodeModel(
            name="Node with custom metadata",
            type="Annotation",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Literature",
                "bloom_level": "Create",
                "confidence": 0.87,
                "source": "Book: 1984",
                "page": 42,
                "chapter": "Part 1",
                "highlight_color": "yellow",
                "is_favorite": True
            }
        )

        assert node.properties is not None
        # PKG fields preserved
        assert node.properties["discipline"] == "Literature"
        assert node.properties["bloom_level"] == "Create"
        assert node.properties["confidence"] == 0.87

        # Custom metadata preserved
        assert node.properties["source"] == "Book: 1984"
        assert node.properties["page"] == 42
        assert node.properties["chapter"] == "Part 1"
        assert node.properties["highlight_color"] == "yellow"
        assert node.properties["is_favorite"] is True

    def test_node_model_optional_type(self):
        """Test that type is optional."""
        node = NodeModel(
            name="Node without type",
            type=None,
            embedding=[0.1, 0.2, 0.3]
        )
        assert node.type is None

    def test_node_model_dict_export(self):
        """Test NodeModel exports to dict correctly."""
        node = NodeModel(
            name="Export test",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Testing",
                "confidence": 0.95
            }
        )

        node_dict = node.model_dump()
        assert node_dict["name"] == "Export test"
        assert node_dict["properties"] is not None
        assert node_dict["properties"]["confidence"] == 0.95

    def test_node_model_json_serialization(self):
        """Test NodeModel can be serialized to JSON."""
        node = NodeModel(
            name="JSON test",
            type="Test",
            embedding=[0.1, 0.2, 0.3],
            properties={
                "discipline": "Computer Science",
                "bloom_level": "Understand",
                "confidence": 0.88
            }
        )

        json_str = node.model_dump_json()
        assert isinstance(json_str, str)
        assert "JSON test" in json_str
        assert "Computer Science" in json_str
