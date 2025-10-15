# Personal Knowledge Graph (PKG) - Feature Specification
**Version:** 1.0  
**Date:** October 15, 2025  
**Purpose:** Complete specification for building a Personal Knowledge Graph system from scratch

---

## 1. System Overview

### 1.1 Vision
Build a system that transforms unstructured reading notes and annotations into an interconnected knowledge graph that:
- Extracts atomic, self-contained concepts
- Discovers semantic relationships automatically
- Calculates cognitive understanding levels
- Enables contextual knowledge retrieval

### 1.2 Core Components
```
┌─────────────────────────────────────────────────────────────┐
│                     PKG SYSTEM ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT LAYER                                                  │
│  ├─ Reading Notes/Annotations                                │
│  ├─ Book Excerpts                                            │
│  └─ Learning Materials                                       │
│                           │                                   │
│                           ▼                                   │
│  EXTRACTION SERVICE (LLM-powered)                            │
│  ├─ Atomic Concept Extraction                                │
│  ├─ Type Classification                                      │
│  ├─ Discipline Tagging                                       │
│  └─ Confidence Scoring                                       │
│                           │                                   │
│                           ▼                                   │
│  RELATIONSHIP DISCOVERY                                       │
│  ├─ Vector Similarity Search                                 │
│  ├─ Semantic Connection Detection                            │
│  └─ Relationship Type Classification                         │
│                           │                                   │
│                           ▼                                   │
│  GRAPH ENRICHMENT                                             │
│  ├─ Bloom's Taxonomy Calculation                             │
│  ├─ Context Analysis                                         │
│  └─ Property Updates                                         │
│                           │                                   │
│                           ▼                                   │
│  STORAGE LAYER                                                │
│  ├─ Neo4j Graph Database (nodes + relationships)             │
│  └─ Vector Index (embeddings for similarity search)          │
│                           │                                   │
│                           ▼                                   │
│  RETRIEVAL & QUERY                                            │
│  ├─ RAG-based Contextual Queries                             │
│  ├─ Similarity Search                                        │
│  └─ Graph Traversal                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Feature Requirements

### 2.1 FEATURE: Atomic Concept Extraction

#### 2.1.1 Purpose
Extract 1-3 atomic, self-contained concepts from unstructured text that represent distinct knowledge units.

#### 2.1.2 Input
```json
{
  "title": "Book Title - Chapter X",
  "content": "Raw annotation text from reading...",
  "metadata": {
    "book": "Book Name",
    "author": "Author Name",
    "chapter": "3",
    "page": "45",
    "date": "2025-10-15"
  }
}
```

#### 2.1.3 LLM Prompt Requirements
Create a prompt that instructs the LLM to:
- Extract atomic concepts (self-contained, 5-20 words each)
- Classify each concept by type (Identity, Memory, Belief, Preference, Goal, Event, etc.)
- Tag with academic discipline (Psychology, Computer Science, Philosophy, etc.) when applicable
- Provide confidence score (0.0-1.0) for extraction quality
- Return valid JSON format

#### 2.1.4 Output Schema
```json
{
  "nodes": [
    {
      "name": "System 1 thinking operates automatically and quickly",
      "type": "Concept",
      "discipline": "Cognitive Psychology",
      "confidence": 0.95
    },
    {
      "name": "System 2 requires effortful mental activities",
      "type": "Concept",
      "discipline": "Cognitive Psychology",
      "confidence": 0.92
    }
  ]
}
```

#### 2.1.5 Technical Implementation
1. **LLM Integration**
   - Use structured output mode (JSON)
   - Temperature: 0.3-0.5 (balanced creativity/consistency)
   - Implement retry logic for JSON parsing failures
   - Support multiple LLM providers (OpenAI, Azure, Anthropic)

2. **Validation**
   - Ensure nodes are self-contained (can stand alone)
   - Verify confidence scores are in range [0.0, 1.0]
   - Check discipline tags against known categories
   - Validate node names are descriptive (min 5 words)

3. **Error Handling**
   - Graceful degradation if LLM fails
   - Log extraction failures with context
   - Provide meaningful error messages to users

---

### 2.2 FEATURE: Relationship Discovery

#### 2.2.1 Purpose
Automatically discover semantic relationships between new concepts and existing knowledge graph.

#### 2.2.2 Relationship Types to Support

**Semantic Relationships:**
- `SIMILAR_TO`: Concepts share similar meanings
- `CONTRASTS_WITH`: Concepts are opposites
- `RELATED_TO`: General semantic connection
- `EXTENDS`: Builds upon another concept
- `SPECIALIZES`: More specific instance

**Hierarchical:**
- `PARENT_OF` / `CHILD_OF`: Categorical hierarchy
- `PART_OF` / `CONTAINS`: Composition
- `SUBTOPIC_OF`: Knowledge organization

**Argumentative:**
- `SUPPORTS`: Provides evidence for
- `OPPOSES`: Contradicts or argues against
- `EVIDENCES`: Provides proof
- `REFUTES`: Disproves

**Causal:**
- `LEADS_TO` / `CAUSES`: Direct causation
- `RESULTS_IN`: Consequence
- `ENABLES` / `PREVENTS`: Possibility control

**Temporal:**
- `PRECEDES` / `FOLLOWS`: Time sequence
- `HAPPENS_BEFORE` / `HAPPENS_AFTER`: Event ordering

**Influence:**
- `SHAPES` / `INFLUENCES`: Effect on another
- `INSPIRES`: Creative motivation
- `MOTIVATES`: Drives action
- `ENHANCES` / `WEAKENS`: Strength modification

**Cognitive:**
- `RESONATES_WITH`: Emotional/intellectual alignment
- `CONFLICTS_WITH`: Internal contradiction
- `EVOLVES_INTO` / `TRANSFORMS_TO`: Personal growth
- `APPLIES_TO`: Practical application

#### 2.2.3 Discovery Process

**Step 1: Vector Similarity Search**
```python
# For each new node:
1. Generate embedding (1536-dim vector using OpenAI ada-002)
2. Query existing nodes using cosine similarity
3. Filter results by threshold (e.g., similarity > 0.7)
4. Return top-K candidates (K=5)
```

**Step 2: LLM Relationship Classification**
```python
# For each similar node pair:
1. Pass both node names to LLM
2. Ask LLM to determine relationship type and direction
3. Only create relationship if confidence > threshold
4. Validate relationship makes semantic sense
```

**Step 3: Storage**
```python
# Store in Neo4j:
CREATE (source:NodeName {name: $source_name, UserId: $user_id})
CREATE (target:NodeName {name: $target_name, UserId: $user_id})
CREATE (source)-[r:RELATIONSHIP_TYPE {type: $relation}]->(target)
```

#### 2.2.4 Technical Implementation

1. **Vector Search**
   - Use Neo4j vector index for similarity queries
   - Embedding dimension: 1536 (OpenAI ada-002)
   - Distance metric: Cosine similarity
   - Index type: HNSW (Hierarchical Navigable Small World)

2. **Relationship Generation**
   - Use LLM with structured output
   - Include relationship type taxonomy in prompt
   - Validate directionality (source → target)
   - Avoid duplicate relationships

3. **Performance Requirements**
   - Similarity search: < 100ms
   - Relationship generation: < 2 seconds
   - Batch processing for multiple nodes

---

### 2.3 FEATURE: Bloom's Taxonomy Calculation

#### 2.3.1 Purpose
Automatically assess cognitive understanding level of each concept based on graph structure.

#### 2.3.2 Bloom's Taxonomy Levels
1. **Remember**: Isolated fact (minimal connections)
2. **Understand**: Connected to similar concepts (2+ connections)
3. **Apply**: Connected across multiple contexts (3+ connections, 2+ context types)
4. **Analyze**: Highly connected across diverse contexts (6+ connections, 3+ context types)
5. **Evaluate**: Critical analysis with supporting/opposing connections
6. **Create**: Synthesis of multiple concepts into new insights

#### 2.3.3 Calculation Algorithm
```python
async def calculate_bloom_level(node_name: str, user_id: str) -> str:
    """
    Calculate Bloom level based on graph evidence:
    - degree: Number of relationships (edges) connected to node
    - contexts: Number of unique node types in neighborhood
    - relationship_types: Types of relationships (causal, argumentative, etc.)
    """
    
    # Query Neo4j for graph metrics
    degree = count_relationships(node_name)
    contexts = count_unique_neighbor_types(node_name)
    has_argumentative = has_relationship_types(node_name, ['SUPPORTS', 'OPPOSES'])
    
    # Classification logic
    if degree >= 6 and contexts >= 3:
        if has_argumentative:
            return "Evaluate"  # Critical thinking evident
        return "Analyze"  # Deep cross-contextual understanding
    
    elif degree >= 3 and contexts >= 2:
        return "Apply"  # Cross-contextual application
    
    elif degree >= 2:
        return "Understand"  # Basic connections
    
    else:
        return "Remember"  # New or isolated concept
```

#### 2.3.4 Update Triggers
Recalculate Bloom level when:
- New node is added
- New relationship is created
- Node's neighborhood changes

Update affected nodes:
- The new node itself
- All directly connected neighbors
- Potentially 2-hop neighbors (for major changes)

#### 2.3.5 Storage
Store as node property:
```cypher
MATCH (n:NodeName {name: $node_name, UserId: $user_id})
SET n.bloom_level = $bloom_level
SET n.last_updated = datetime()
```

---

### 2.4 FEATURE: Graph Storage

#### 2.4.1 Database: Neo4j

**Node Schema:**
```cypher
(:NodeName {
  name: STRING,              // Primary identifier
  type: STRING,              // Identity, Belief, Concept, etc.
  UserId: STRING,            // Multi-tenancy isolation
  embedding: LIST[FLOAT],    // 1536-dim vector
  discipline: STRING?,       // Optional: academic field
  bloom_level: STRING,       // Remember, Understand, Apply, Analyze, Evaluate, Create
  confidence: FLOAT?,        // Optional: 0.0-1.0
  created_at: DATETIME,
  last_updated: DATETIME
})
```

**Relationship Schema:**
```cypher
(source)-[:RELATIONSHIP_TYPE {
  type: STRING,           // SIMILAR_TO, SUPPORTS, etc.
  created_at: DATETIME,
  confidence: FLOAT?      // Optional
}]->(target)
```

**Indexes:**
```cypher
// Text search
CREATE INDEX node_name_index FOR (n:NodeName) ON (n.name);

// User isolation
CREATE INDEX user_id_index FOR (n:NodeName) ON (n.UserId);

// Vector search
CREATE VECTOR INDEX node_embedding_index
FOR (n:NodeName) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

#### 2.4.2 Multi-Tenancy
- Every node/relationship includes `UserId` property
- All queries filter by `UserId`
- Complete data isolation between users

#### 2.4.3 Transactions
Use transactions for atomic operations:
```python
async with neo4j_driver.session() as session:
    await session.execute_write(
        # Create nodes
        # Create relationships
        # Update embeddings
        # Calculate Bloom levels
    )
```

---

### 2.5 FEATURE: Contextual Retrieval (RAG)

#### 2.5.1 Purpose
Answer user queries by retrieving relevant context from knowledge graph.

#### 2.5.2 Retrieval Strategy

**Hybrid Approach:**
1. **Vector Similarity** (70% weight)
   - Embed user query
   - Find top-K similar nodes
   - Include similarity scores

2. **Graph Traversal** (30% weight)
   - Start from similar nodes
   - Traverse 1-2 hops
   - Collect connected context
   - Include relationship types

3. **Context Assembly**
   ```
   Context Format:
   "You have learned about:
   - [Node 1] (Type: Concept, Discipline: Psychology, Bloom: Analyze)
     Connected to: [Node 2] via SUPPORTS
   - [Node 2] (Type: Theory, Discipline: Psychology, Bloom: Understand)
   - [Node 3] (Type: Application, Discipline: Education, Bloom: Apply)
     Connected to: [Node 1] via APPLIES_TO
   "
   ```

#### 2.5.3 Query Processing
```python
async def rag_query(user_id: str, query: str):
    # 1. Vector search
    similar_nodes = await vector_search(query, user_id, limit=5)
    
    # 2. Graph traversal
    context_nodes = []
    for node in similar_nodes:
        neighbors = await get_neighbors(node, depth=1)
        context_nodes.extend(neighbors)
    
    # 3. Build context string
    context = build_context_string(similar_nodes, context_nodes)
    
    # 4. LLM generation
    response = await llm.chat(
        system="You are a knowledge assistant. Use the provided context.",
        user=f"Context: {context}\n\nQuestion: {query}"
    )
    
    return response
```

---

## 3. API Endpoints

### 3.1 User Management
```
POST   /api/v1/users/{user_id}          Create user
DELETE /api/v1/users/{user_id}          Delete user and all data
GET    /api/v1/users/{user_id}          Get user status
```

### 3.2 Data Ingestion
```
POST   /api/v1/users/{user_id}/ingest   Ingest unstructured data
```
**Request Body:**
```json
{
  "title": "string",
  "content": "string",
  "metadata": {
    "key": "value"
  }
}
```

### 3.3 Knowledge Retrieval
```
POST   /api/v1/users/{user_id}/rag/query         RAG-based query
POST   /api/v1/users/{user_id}/rag/query-vector  Vector similarity search
POST   /api/v1/users/{user_id}/ask               Structured query with schema
```

### 3.4 Graph Operations
```
GET    /api/v1/users/{user_id}/custom-data       Get all nodes and relationships
POST   /api/v1/users/{user_id}/graph/nodes       Add nodes manually
POST   /api/v1/users/{user_id}/graph/relationships  Add relationships manually
```

---

## 4. Technology Stack

### 4.1 Backend
- **Language**: Python 3.12+
- **Framework**: FastAPI
- **Async**: asyncio, aiohttp

### 4.2 Database
- **Graph DB**: Neo4j 5.x
- **Vector Index**: Neo4j native vector search
- **Driver**: neo4j-python-driver (async)

### 4.3 LLM Integration
- **Primary**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-ada-002
- **Alternatives**: Azure OpenAI, Anthropic Claude
- **Pattern**: Provider abstraction layer

### 4.4 Infrastructure
- **Containerization**: Docker + Docker Compose
- **Dependency Management**: Poetry
- **Testing**: pytest + pytest-asyncio
- **Logging**: structlog or Python logging

---

## 5. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up FastAPI application
- [ ] Configure Neo4j database
- [ ] Implement user management
- [ ] Create data models (Pydantic schemas)

### Phase 2: Extraction Service (Week 2-3)
- [ ] Design LLM prompts for node extraction
- [ ] Implement LLM client abstraction
- [ ] Build extraction pipeline
- [ ] Add confidence scoring
- [ ] Add discipline tagging

### Phase 3: Relationship Discovery (Week 3-4)
- [ ] Implement vector embedding generation
- [ ] Build similarity search
- [ ] Create relationship classification prompt
- [ ] Implement relationship generation
- [ ] Add relationship type validation

### Phase 4: Graph Enrichment (Week 4-5)
- [ ] Implement Bloom's taxonomy calculator
- [ ] Build property update pipeline
- [ ] Add automatic enrichment triggers
- [ ] Implement batch updates

### Phase 5: Retrieval & Query (Week 5-6)
- [ ] Build RAG pipeline
- [ ] Implement hybrid retrieval
- [ ] Add context assembly
- [ ] Create query endpoints

### Phase 6: Testing & Optimization (Week 6-8)
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Error handling refinement
- [ ] Documentation

---

## 6. Success Metrics

### 6.1 Functional Metrics
- Extraction accuracy: >85% concepts are atomic and meaningful
- Relationship precision: >80% relationships are semantically valid
- Bloom calculation: <5% error rate vs manual labeling
- Query relevance: >90% queries return useful context

### 6.2 Performance Metrics
- Ingestion time: <3 seconds per annotation
- Similarity search: <100ms
- RAG query response: <2 seconds
- Graph update: <500ms

### 6.3 Quality Metrics
- Confidence score calibration: ±10% of human judgment
- Discipline tagging accuracy: >85%
- Relationship type accuracy: >80%

---

## 7. Testing Strategy

### 7.1 Unit Tests
- LLM prompt parsing
- Node extraction logic
- Relationship generation
- Bloom calculation algorithm
- Vector similarity functions

### 7.2 Integration Tests
- End-to-end ingestion flow
- Graph operations with Neo4j
- Multi-user isolation
- RAG retrieval pipeline

### 7.3 Test Data
Create fixtures:
- Sample annotations from different domains
- Pre-built small knowledge graphs
- Edge cases (empty content, malformed JSON, etc.)

---

## 8. Deployment

### 8.1 Development
```bash
docker-compose up -d          # Neo4j + app
poetry install                # Dependencies
poetry run pytest             # Tests
poetry run uvicorn server.main:app --reload
```

### 8.2 Production
- Environment variables for secrets
- Separate Neo4j instance
- Load balancing for API
- Monitoring and logging
- Backup strategy for graph data

---

## 9. Future Enhancements

### 9.1 UI/UX
- Interactive graph visualization (D3.js, Cytoscape.js)
- Visual query builder
- Node editing interface
- Relationship explorer

### 9.2 Advanced Features
- Temporal analysis (how knowledge evolves)
- Community detection (topic clustering)
- Knowledge gaps identification
- Learning path recommendations
- Export to Obsidian/Roam format

### 9.3 Intelligence
- Active learning (ask user for clarification)
- Contradiction detection
- Concept consolidation (merge similar nodes)
- Automatic summarization

---

## 10. Documentation Requirements

### 10.1 User Documentation
- Quick start guide
- API reference
- Example workflows
- Best practices for annotations

### 10.2 Developer Documentation
- Architecture overview
- Code structure
- Prompt engineering guide
- Database schema reference
- Testing guide

---

## Appendix A: Example Prompts

### A.1 Node Extraction Prompt
```
You are a knowledge extraction expert. Extract 1-3 atomic, self-contained concepts from the text.

INCLUDE these fields per node:
- name: Self-contained concept (5-20 words)
- type: Category (Concept, Theory, Method, Finding, etc.)
- discipline: Academic field (Psychology, Computer Science, etc.) or empty
- confidence: Extraction quality (0.0-1.0)

Example:
{
  "nodes": [
    {
      "name": "Working memory has limited capacity of 7±2 items",
      "type": "Finding",
      "discipline": "Cognitive Psychology",
      "confidence": 0.95
    }
  ]
}
```

### A.2 Relationship Discovery Prompt
```
You are an expert in semantic relationships. Determine the relationship between these concepts:

Node1: "Working memory has limited capacity"
Node2: "Chunking improves memory retention"

Relationship types: LEADS_TO, SUPPORTS, OPPOSES, SIMILAR_TO, APPLIES_TO, etc.

Output:
{
  "source_id": "Node1",
  "relation": "LEADS_TO",
  "target_id": "Node2",
  "confidence": 0.85
}
```

---

## Appendix B: Configuration

### B.1 Environment Variables
```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM
OPENAI_API_KEY=sk-...
LLM_SERVICE=openai
LLM_MODEL=gpt-4

# Embedding
EMBEDDING_MODEL=text-embedding-ada-002

# App
APP_ENV=development
LOG_LEVEL=INFO
```

### B.2 Feature Flags
```python
ENABLE_DISCIPLINE_TAGGING = True
ENABLE_CONFIDENCE_SCORING = True
ENABLE_BLOOM_CALCULATION = True
SIMILARITY_THRESHOLD = 0.7
MAX_RELATIONSHIPS_PER_NODE = 10
```

---

**End of Specification**
