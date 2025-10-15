"""
Comprehensive unit tests for LLM Node class (persona.llm.llm_graph.Node).

This test file consolidates tests for:
- Basic Node properties handling
- PKG field extraction and packing (discipline, bloom_level, confidence)
- Node to NodeModel flow
- Various use cases and scenarios

Tests that Node properly:
1. Accepts PKG fields from LLM responses
2. Packs them into properties dict automatically
3. Preserves custom properties
4. Can be transferred to NodeModel correctly
"""

import pytest
from persona.llm.llm_graph import Node
from persona.models.schema import NodeModel


class TestBasicNodeCreation:
    """Test basic Node model creation and properties."""

    def test_llm_node_basic_creation(self):
        """Test basic LLM node creation without PKG fields."""
        node = Node(name="Test Node", type="Concept")
        assert node.name == "Test Node"
        assert node.type == "Concept"
        assert node.properties == {}

    def test_node_without_pkg_fields(self):
        """Test that Node works without PKG fields (backward compatibility)."""
        node = Node(
            name="Simple node without metadata",
            type="Identity"
        )

        assert node.name == "Simple node without metadata"
        assert node.type == "Identity"
        assert node.discipline is None
        assert node.bloom_level is None
        assert node.confidence is None
        assert node.properties == {}


class TestPKGFieldPacking:
    """Test that PKG fields are automatically packed into properties dict."""

    def test_node_with_all_pkg_fields(self):
        """Test that Node accepts all PKG fields from LLM response."""
        node_data = {
            "name": "System 1 thinking is fast and automatic",
            "type": "Concept",
            "discipline": "Psychology",
            "bloom_level": "Understand",
            "confidence": 0.95
        }

        node = Node(**node_data)

        # Verify fields are set
        assert node.name == "System 1 thinking is fast and automatic"
        assert node.type == "Concept"
        assert node.discipline == "Psychology"
        assert node.bloom_level == "Understand"
        assert node.confidence == 0.95

        # Verify fields are packed into properties dict
        assert node.properties["discipline"] == "Psychology"
        assert node.properties["bloom_level"] == "Understand"
        assert node.properties["confidence"] == 0.95

    def test_node_packs_discipline_into_properties(self):
        """Test that discipline field is automatically packed into properties."""
        node = Node(
            name="Cognitive biases affect decision-making",
            type="Insight",
            discipline="Behavioral Economics",
            bloom_level="Analyze",
            confidence=0.9
        )

        assert "discipline" in node.properties
        assert node.properties["discipline"] == "Behavioral Economics"

    def test_node_packs_bloom_level_into_properties(self):
        """Test that bloom_level field is automatically packed into properties."""
        node = Node(
            name="Memorized the definition of metacognition",
            type="Memory",
            discipline="Education",
            bloom_level="Remember",
            confidence=1.0
        )

        assert "bloom_level" in node.properties
        assert node.properties["bloom_level"] == "Remember"

    def test_node_packs_confidence_into_properties(self):
        """Test that confidence field is automatically packed into properties."""
        node = Node(
            name="Considering starting a meditation practice",
            type="Goal",
            discipline="Health",
            bloom_level="Apply",
            confidence=0.7
        )

        assert "confidence" in node.properties
        assert node.properties["confidence"] == 0.7

    def test_node_with_partial_pkg_fields(self):
        """Test that Node works with only some PKG fields."""
        node = Node(
            name="Reading about quantum computing",
            type="Interest",
            discipline="Computer Science"
            # No bloom_level or confidence
        )

        assert node.discipline == "Computer Science"
        assert node.bloom_level is None
        assert node.confidence is None
        assert "discipline" in node.properties
        assert "bloom_level" not in node.properties
        assert "confidence" not in node.properties


class TestCustomPropertiesPreservation:
    """Test that custom properties work alongside PKG fields."""

    def test_node_with_existing_properties_dict(self):
        """Test that PKG fields are added to existing properties."""
        node = Node(
            name="Test node",
            type="Test",
            discipline="Testing",
            bloom_level="Create",
            confidence=0.8,
            properties={"custom_field": "custom_value"}
        )

        # Custom property preserved
        assert node.properties["custom_field"] == "custom_value"

        # PKG fields added
        assert node.properties["discipline"] == "Testing"
        assert node.properties["bloom_level"] == "Create"
        assert node.properties["confidence"] == 0.8

    def test_node_with_custom_properties(self):
        """Test that custom properties in LLM Node are preserved."""
        node = Node(
            name="Custom Node",
            type="Entity",
            discipline="Science",
            confidence=0.88,
            properties={"custom_field": "custom_value", "count": 42}
        )

        # PKG fields should be added to existing properties
        assert node.properties["discipline"] == "Science"
        assert node.properties["confidence"] == 0.88
        assert node.properties["custom_field"] == "custom_value"
        assert node.properties["count"] == 42


class TestBloomLevels:
    """Test all Bloom's taxonomy levels."""

    def test_all_bloom_levels_accepted(self):
        """Test that all Bloom's taxonomy levels are accepted."""
        bloom_levels = [
            "Remember",
            "Understand",
            "Apply",
            "Analyze",
            "Evaluate",
            "Create"
        ]

        for level in bloom_levels:
            node = Node(
                name=f"Node at {level} level",
                type="Concept",
                discipline="Education",
                bloom_level=level,
                confidence=0.9
            )

            assert node.bloom_level == level
            assert node.properties["bloom_level"] == level

    def test_all_bloom_levels_in_properties(self):
        """Test all six Bloom taxonomy levels can be stored in properties."""
        bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

        for level in bloom_levels:
            node = Node(
                name=f"Test {level}",
                type="Knowledge",
                bloom_level=level
            )
            assert node.properties["bloom_level"] == level


class TestDisciplineDomains:
    """Test various discipline domains."""

    def test_discipline_domains(self):
        """Test various discipline domains."""
        disciplines = [
            "Psychology",
            "Computer Science",
            "Philosophy",
            "Biology",
            "History",
            "Career",
            "Health",
            "Finance",
            "Relationships",
            "Personal Development"
        ]

        for discipline in disciplines:
            node = Node(
                name=f"Concept in {discipline}",
                type="Concept",
                discipline=discipline,
                bloom_level="Understand",
                confidence=0.85
            )

            assert node.discipline == discipline
            assert node.properties["discipline"] == discipline

    def test_common_disciplines_in_properties(self):
        """Test common academic disciplines can be stored."""
        disciplines = ["Mathematics", "Physics", "Computer Science", "Biology", "Psychology"]

        for discipline in disciplines:
            node = Node(
                name="Test Concept",
                type="Concept",
                discipline=discipline
            )
            assert node.properties["discipline"] == discipline


class TestConfidenceScores:
    """Test confidence score handling."""

    def test_confidence_range(self):
        """Test confidence scores at different levels."""
        confidence_scores = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

        for score in confidence_scores:
            node = Node(
                name=f"Node with confidence {score}",
                type="Test",
                discipline="Testing",
                bloom_level="Evaluate",
                confidence=score
            )

            assert node.confidence == score
            assert node.properties["confidence"] == score

    def test_confidence_scores_in_properties(self):
        """Test various confidence scores can be stored."""
        confidence_scores = [0.0, 0.5, 0.75, 0.9, 0.95, 1.0]

        for score in confidence_scores:
            node = Node(
                name="Test",
                type="Concept",
                confidence=score
            )
            assert node.properties["confidence"] == score


class TestPropertiesSerialization:
    """Test that properties are properly structured, not double-serialized."""

    def test_properties_dict_not_double_serialized(self):
        """Test that properties dict is NOT serialized as JSON string."""
        node = Node(
            name="Test node",
            type="Test",
            discipline="Testing",
            bloom_level="Apply",
            confidence=0.88
        )

        # Properties should be a dict, not a string
        assert isinstance(node.properties, dict)
        assert not isinstance(node.properties, str)

        # Values should be their actual types, not stringified
        assert isinstance(node.properties["discipline"], str)
        assert isinstance(node.properties["bloom_level"], str)
        assert isinstance(node.properties["confidence"], float)

    def test_node_model_validator_execution(self):
        """Test that the model_validator is executed and packs properties."""
        # Create node without properties dict
        node = Node(
            name="Validator test",
            type="Test",
            discipline="Unit Testing",
            bloom_level="Create",
            confidence=0.92
        )

        # Validator should have created properties dict and packed fields
        assert isinstance(node.properties, dict)
        assert len(node.properties) == 3
        assert all(key in node.properties for key in ["discipline", "bloom_level", "confidence"])


class TestNodeToNodeModelFlow:
    """Test the flow from LLM Node to NodeModel (schema model)."""

    def test_node_packs_properties(self):
        """Test that LLM Node auto-packs PKG fields into properties dict."""
        node = Node(
            name="The Count of Monte Cristo",
            type="Literary Work",
            discipline="Literature",
            bloom_level="Remember",
            confidence=0.95
        )

        # Verify properties dict was populated by @model_validator
        assert "discipline" in node.properties
        assert node.properties["discipline"] == "Literature"
        assert node.properties["bloom_level"] == "Remember"
        assert node.properties["confidence"] == 0.95

    def test_node_properties_can_be_passed_to_node_model(self):
        """Test that LLM Node.properties can be passed to NodeModel."""
        # Create LLM Node with PKG fields
        llm_node = Node(
            name="Edmond Dantès",
            type="Character",
            discipline="Literature",
            bloom_level="Understand",
            confidence=0.92
        )

        # Transfer to NodeModel
        node_model = NodeModel(
            name=llm_node.name,
            type=llm_node.type,
            properties=llm_node.properties,
            embedding=[0.1, 0.2, 0.3]  # dummy embedding
        )

        # Verify properties were transferred
        assert node_model.properties is not None
        assert node_model.properties["discipline"] == "Literature"
        assert node_model.properties["bloom_level"] == "Understand"
        assert node_model.properties["confidence"] == 0.92
        assert node_model.name == "Edmond Dantès"
        assert node_model.type == "Character"

    def test_node_without_pkg_fields_still_works(self):
        """Test that nodes without PKG fields still work (backwards compatibility)."""
        node = Node(
            name="Generic Node",
            type="Concept"
        )

        node_model = NodeModel(
            name=node.name,
            type=node.type,
            properties=node.properties,
            embedding=[0.1, 0.2]
        )

        # Properties dict should exist but not have PKG fields
        assert node_model.properties is not None
        assert "discipline" not in node_model.properties
        assert node_model.name == "Generic Node"


class TestRealWorldScenarios:
    """Integration scenarios with realistic data."""

    def test_psychology_book_annotation(self):
        """Test Node with data from psychology book annotation."""
        node = Node(
            name="System 2 thinking requires significant mental effort and glucose",
            type="Insight",
            discipline="Psychology",
            bloom_level="Understand",
            confidence=0.95
        )

        assert node.properties["discipline"] == "Psychology"
        assert node.properties["bloom_level"] == "Understand"
        assert node.properties["confidence"] == 0.95

    def test_self_help_book_application(self):
        """Test Node with application from self-help book."""
        node = Node(
            name="Applying atomic habits principle to morning meditation routine",
            type="Goal",
            discipline="Personal Development",
            bloom_level="Apply",
            confidence=0.88
        )

        assert node.properties["discipline"] == "Personal Development"
        assert node.properties["bloom_level"] == "Apply"
        assert node.properties["confidence"] == 0.88

    def test_technical_concept_node(self):
        """Test Node with technical concept from computer science."""
        node = Node(
            name="Understanding asynchronous programming with async/await patterns",
            type="Concept",
            discipline="Computer Science",
            bloom_level="Analyze",
            confidence=0.82
        )

        assert node.properties["discipline"] == "Computer Science"
        assert node.properties["bloom_level"] == "Analyze"
        assert node.properties["confidence"] == 0.82

    def test_reading_annotation_node_with_properties(self):
        """Test node representing a reading annotation with PKG properties."""
        node = Node(
            name="Active recall is more effective than passive re-reading",
            type="Finding",
            discipline="Cognitive Psychology",
            bloom_level="Understand",
            confidence=0.95,
            properties={
                "source": "Make It Stick, Chapter 2",
                "page": "28"
            }
        )

        assert "Active recall" in node.name
        assert node.type == "Finding"
        assert node.properties["discipline"] == "Cognitive Psychology"
        assert node.properties["bloom_level"] == "Understand"
        assert node.properties["confidence"] == 0.95
        assert node.properties["source"] == "Make It Stick, Chapter 2"
        assert node.properties["page"] == "28"

    def test_concept_node_with_calculated_bloom_level(self):
        """Test concept node with graph-calculated bloom level."""
        node = Node(
            name="Backpropagation Algorithm",
            type="Concept",
            discipline="Deep Learning",
            bloom_level="Analyze",
            properties={
                "degree": 6,
                "contexts": ["Neural Networks", "Optimization"]
            }
        )

        assert node.properties["bloom_level"] == "Analyze"
        assert node.properties["degree"] == 6
        assert node.properties["contexts"] == ["Neural Networks", "Optimization"]
