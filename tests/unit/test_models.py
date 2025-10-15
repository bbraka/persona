"""
Unit tests for Node properties including bloom_level, discipline, and confidence.
"""

import pytest
from persona.models.schema import NodeModel, Relationship, RelationshipModel
from persona.llm.llm_graph import Node as LLMNode


class TestLLMNodeWithProperties:
    """Tests for LLM Node model with properties dict for PKG metadata"""
    
    def test_llm_node_basic_creation(self):
        """Test basic LLM node creation"""
        node = LLMNode(name="Test Node", type="Concept")
        assert node.name == "Test Node"
        assert node.type == "Concept"
        assert node.properties == {}
    
    def test_llm_node_with_bloom_level_property(self):
        """Test LLM node with bloom_level in properties"""
        node = LLMNode(
            name="Important Concept",
            type="Concept",
            properties={"bloom_level": "Understand"}
        )
        assert node.properties["bloom_level"] == "Understand"
    
    def test_llm_node_with_discipline_property(self):
        """Test LLM node with discipline in properties"""
        node = LLMNode(
            name="Machine Learning",
            type="Concept",
            properties={"discipline": "Computer Science"}
        )
        assert node.properties["discipline"] == "Computer Science"
    
    def test_llm_node_with_confidence_property(self):
        """Test LLM node with confidence in properties"""
        node = LLMNode(
            name="Extracted Fact",
            type="Fact",
            properties={"confidence": 0.95}
        )
        assert node.properties["confidence"] == 0.95
    
    def test_llm_node_with_all_pkg_properties(self):
        """Test LLM node with all PKG properties"""
        node = LLMNode(
            name="Neural Networks",
            type="Concept",
            properties={
                "bloom_level": "Analyze",
                "discipline": "Machine Learning",
                "confidence": 0.88
            }
        )
        assert node.properties["bloom_level"] == "Analyze"
        assert node.properties["discipline"] == "Machine Learning"
        assert node.properties["confidence"] == 0.88


class TestBloomLevelProperty:
    """Specific tests for Bloom taxonomy levels in properties"""
    
    def test_all_bloom_levels_in_properties(self):
        """Test all six Bloom taxonomy levels can be stored"""
        bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        
        for level in bloom_levels:
            node = LLMNode(
                name=f"Test {level}",
                type="Knowledge",
                properties={"bloom_level": level}
            )
            assert node.properties["bloom_level"] == level


class TestDisciplineProperty:
    """Specific tests for discipline property"""
    
    def test_common_disciplines_in_properties(self):
        """Test common academic disciplines can be stored"""
        disciplines = ["Mathematics", "Physics", "Computer Science", "Biology", "Psychology"]
        
        for discipline in disciplines:
            node = LLMNode(
                name="Test Concept",
                type="Concept",
                properties={"discipline": discipline}
            )
            assert node.properties["discipline"] == discipline


class TestConfidenceProperty:
    """Specific tests for confidence score property"""
    
    def test_confidence_scores_in_properties(self):
        """Test various confidence scores can be stored"""
        confidence_scores = [0.0, 0.5, 0.75, 0.9, 0.95, 1.0]
        
        for score in confidence_scores:
            node = LLMNode(
                name="Test",
                type="Concept",
                properties={"confidence": score}
            )
            assert node.properties["confidence"] == score


class TestIntegrationScenarios:
    """Integration scenarios with all properties"""
    
    def test_reading_annotation_node_with_properties(self):
        """Test node representing a reading annotation with PKG properties"""
        node = LLMNode(
            name="Active recall is more effective than passive re-reading",
            type="Finding",
            properties={
                "discipline": "Cognitive Psychology",
                "bloom_level": "Understand",
                "confidence": 0.95,
                "source": "Make It Stick, Chapter 2",
                "page": "28"
            }
        )
        
        assert "Active recall" in node.name
        assert node.type == "Finding"
        assert node.properties["discipline"] == "Cognitive Psychology"
        assert node.properties["bloom_level"] == "Understand"
        assert node.properties["confidence"] == 0.95
    
    def test_concept_node_with_calculated_bloom_level(self):
        """Test concept node with graph-calculated bloom level"""
        node = LLMNode(
            name="Backpropagation Algorithm",
            type="Concept",
            properties={
                "discipline": "Deep Learning",
                "bloom_level": "Analyze",
                "degree": 6,
                "contexts": ["Neural Networks", "Optimization"]
            }
        )
        
        assert node.properties["bloom_level"] == "Analyze"
        assert node.properties["degree"] == 6


class TestRelationshipModel:
    """Tests for Relationship models"""
    
    def test_relationship_basic(self):
        """Test basic relationship creation"""
        rel = Relationship(source="Node A", target="Node B", relation="RELATED_TO")
        assert rel.source == "Node A"
        assert rel.target == "Node B"
        assert rel.relation == "RELATED_TO"
