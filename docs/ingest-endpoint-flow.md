# Ingest Endpoint Flow

## System Overview

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client     │─────▶│  API Server  │─────▶│  LLM Service │      │   Neo4j      │
│  (Frontend)  │      │   (FastAPI)  │      │   (OpenAI)   │      │  (Database)  │
└──────────────┘      └──────┬───────┘      └──────────────┘      └──────────────┘
                             │                       ▲                      ▲
                             │                       │                      │
                             └───────────────────────┴──────────────────────┘
                                   GraphConstructor coordinates
```

## What It Receives

**Endpoint:** `POST /api/v1/users/{user_id}/ingest`

**Input:** Book highlights with title, content, and metadata (book_id, highlight_id, date)

## How It's Processed

```
┌─────────────────────────────────────────────────────────────┐
│ 1. API SERVER validates & preprocesses                     │
│    ▸ Checks user exists, content not empty                 │
│    ▸ Combines title + content + metadata into text         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GRAPH CONSTRUCTOR extracts knowledge                    │
│    ▸ Retrieves similar nodes from existing graph (context) │
│    ▸ Sends to LLM for extraction                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LLM SERVICE processes (5 calls)                         │
│    Call 1: Extract entities (Concept, Theme, Person, etc.) │
│    Call 2: Generate relationships between new entities     │
│    Call 3: Connect new entities to existing graph          │
│    Call 4: Find similar concepts in same book              │
│    Call 5: Detect contrasting/opposing concepts            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. GRAPH CONSTRUCTOR enriches                              │
│    ▸ Adds cognitive level tracking                         │
│    ▸ Links highlights to user notes                        │
│    ▸ Generates embeddings for similarity search            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DATABASE LAYER stores                                   │
│    ▸ Checks for duplicate concepts (deduplication)         │
│    ▸ Saves nodes & relationships to Neo4j                  │
│    ▸ Removes infrequent themes                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. API SERVER responds                                     │
│    ▸ Returns success + pruning stats                       │
└─────────────────────────────────────────────────────────────┘
```

## LLM Calls Explained

**Total: ~8 calls per single item**

| Call | Purpose | Why | When |
|------|---------|-----|------|
| **1. Node Extraction** | Turn text into structured entities | Extract concepts, people, themes from highlights | Always |
| **2. Core Relationships** | Link new entities together | Create "synthesizes", "relates to", "supports" connections | Always |
| **3. Cross-Relationships** | Connect to existing knowledge | Build on user's existing graph | If graph exists |
| **4. Book-Scoped** | Group book content | Find related concepts in same book | If book_id provided |
| **5. Contrast Detection** | Find opposing ideas | Detect "individualism" vs "collectivism" debates | If book_id provided |

**Batch Optimization:** Batch endpoint processes 3 items with 8 calls instead of 24 (67% reduction)

## Where Data Goes

**Neo4j Knowledge Graph** stores:
- **Nodes:** Concepts, CognitiveLevel (Bloom's taxonomy), Themes, People, Terms, Characters, etc.
- **Relationships:** RELATED_TO, HAS_UNDERSTANDING_LEVEL, CONTRASTS_WITH, SYNTHESIZED_INTO, etc.

Each concept is deduplicated, embedded for similarity search, and connected to existing knowledge.

## Response

**Success:** Message + theme pruning statistics (how many themes removed)

**Errors:** 400 (empty content), 422 (invalid user), 500 (processing failed)
