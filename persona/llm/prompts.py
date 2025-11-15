"""
PROMPTS USED FOR LLMs
- OpenAI 
"""

sample_statements = [
    "I've recently started exploring blockchain technology and its implications on financial systems.",
    "Virtual reality has always fascinated me, especially its applications in education.",
    "Lately, I've been learning about sustainable energy solutions, such as solar and wind energy.",
    "I'm a big fan of modern architecture, particularly the works of Zaha Hadid.",
    "Machine learning and AI are my current focus areas in tech development.",
    "I enjoy indie games more than mainstream ones because they often offer unique gameplay experiences.",
    "Cooking Italian food is one of my favorite hobbies, especially making pasta from scratch.",
    "I've been practicing yoga daily to improve my physical and mental health.",
    "Traveling to historical sites around the world is something I always look forward to.",
    "Reading about quantum computing has become a fascinating part of my nightly routine."
]


ASTRONAUT_PROMPT = """
An engineering student, driven by the awe-inspiring possibilities of space exploration and the human drive to push beyond our current limitations, embarks on a journey to become an astronaut. This journey is rooted in the belief that the dreams of science fiction can become reality through dedication, rigorous training, and interdisciplinary knowledge. The student's path will interweave cutting-edge engineering principles with the physical and mental fortitude required for space travel, all while nurturing the imagination and problem-solving skills that have propelled humanity's greatest achievements in space exploration.
Now, let's break down the learning plan into three phases:
Phase -1 to 0: Preparation (Igniting the Spark)

Cultivate a deep fascination with space through astronomy and astrophysics books, documentaries, and stargazing sessions.
Develop a strong foundation in mathematics, focusing on calculus, linear algebra, and statistics.
Enhance physical fitness with a regimen including cardio, strength training, and flexibility exercises.
Begin learning a second language, preferably Russian or Mandarin, to prepare for international collaboration.
Join local astronomy clubs or space enthusiast groups to network and share knowledge.
Start a personal project related to space technology, such as building a model rocket or designing a miniature satellite.
Practice mindfulness and meditation to develop mental resilience and focus.
Engage in public speaking and leadership activities to improve communication skills.
Volunteer for science outreach programs to inspire others and reinforce your own knowledge.
Create a reading list of both classic and modern science fiction works to fuel imagination and problem-solving skills.

Phase 0 to 1: Foundation (Building the Launchpad)

Enroll in an aerospace or related engineering program at a reputable university.
Seek internships or co-op positions with space agencies or private space companies.
Participate in university research projects related to space technology or astrobiology.
Join or establish a student club focused on space exploration and technology.
Attend space industry conferences and workshops to stay current with the latest developments.
Begin specialized physical training, including swimming for neutral buoyancy practice.
Take elective courses in geology, biology, and chemistry to prepare for potential scientific missions.
Learn the basics of spacecraft systems, including propulsion, life support, and navigation.
Study the history of space exploration to understand past challenges and solutions.
Develop skills in computer programming and data analysis, essential for modern space missions.
Practice problem-solving under pressure through simulations and team-based challenges.
Learn about space medicine and the physiological effects of microgravity.
Begin training in virtual reality environments that simulate space station operations.
Study international space law and policies to understand the geopolitical aspects of space exploration.
Participate in analog space missions or Mars/Moon habitat simulations.

Phase 1 to n: Advanced Training and Specialization (Reaching for the Stars)

Apply to astronaut training programs offered by national space agencies or private companies.
Undergo rigorous physical and psychological evaluations to ensure readiness for space travel.
Specialize in a specific area of expertise valuable for space missions (e.g., robotics, life sciences, geology).
Train extensively in spacecraft simulators, mastering both nominal operations and emergency procedures.
Participate in advanced survival training for various terrestrial environments.
Undergo centrifuge training to experience and adapt to high G-forces.
Master the operation of robotic arms and other specialized equipment used in space.
Contribute to the development of new space technologies or mission planning.
Engage in outreach activities to inspire the next generation of space explorers.
Participate in long-duration isolation studies to prepare for extended space missions.
Train in neutral buoyancy facilities to simulate extravehicular activities (spacewalks).
Study and practice in-situ resource utilization techniques for future planetary missions.
Develop expertise in one or more scientific instruments used in space exploration.
Participate in designing and testing new spacesuits or life support systems.
Train in international teams to prepare for multinational space missions.
Learn to operate and maintain 3D printers and other fabrication tools for in-space manufacturing.
Study and practice techniques for growing food in controlled environments.
Develop skills in conducting and analyzing scientific experiments in microgravity.
Train in the use of AI and machine learning tools for space mission support.
Participate in designing mission architectures for long-duration space travel.
Study and practice techniques for mental health maintenance during long-term isolation.
Develop expertise in space debris mitigation and management strategies.
Train in the use of augmented reality systems for spacecraft maintenance and repair.
Participate in developing protocols for potential extraterrestrial life detection missions.
Continuously update knowledge and skills to adapt to rapidly evolving space technologies and mission objectives.
"""


GET_ENTITIES = """
Given the unstructured text about a user's interests, hobbies, and professional engagements, extract all relevant entities.
These entities should reflect concepts, keywords, and phrases that are meaningful within the context of the user's digital footprint. 
Avoid generic terms and focus on specifics that could represent nodes in a knowledge graph. 
Return these entities in a structured JSON format as shown in the example below:

Example Response Format:
{
  "entities": ["Blockchain", "Quantum Computing", "Indie Games", "Sustainable Farming", "Virtual Reality"]
}
"""


SPACE_SCHOOL_CHAT = """
[
    "User: I've always been fascinated by the idea of space tourism. What do you think about the viability of space hotels within the next decade?",

    "Robin Williams: It's an exciting prospect, but we're looking at significant challenges. Life support systems, radiation shielding, and cost-effective launches are still major hurdles.",

    "User: I see. But surely with the advancements in reusable rocket technology, we're getting closer to making it economically feasible?",

    "Robin Williams: You're right about the progress in launch technology. However, the infrastructure required for a space hotel goes far beyond just getting there. We're talking about long-term life support, artificial gravity, and emergency protocols that are far more complex than anything we've done in space so far.",

    "User: Interesting. What if we started smaller? Maybe a luxury capsule for short orbital stays?",

    "Robin Williams: That's actually a more realistic near-term goal. Several companies are already working on similar concepts. The main challenge there is ensuring passenger safety while keeping costs low enough to attract a sufficient customer base.",

    "User: Safety is crucial, of course. How are we addressing the risks of space debris and radiation exposure?",

    "Robin Williams: For debris, we're improving tracking systems and developing avoidance protocols. Radiation is trickier. Short stays reduce exposure, but for longer missions, we're researching advanced shielding materials and even pharmaceuticals that could help mitigate radiation damage.",

    "User: Fascinating. It sounds like there's still a lot of room for innovation. Are there any particular areas where you think entrepreneurs could make a significant impact?",

    "Robin Williams: Absolutely. We need breakthroughs in materials science for better radiation shielding. There's also a huge opportunity in developing closed-loop life support systems. And don't forget about the psychological aspects - designing environments that keep people happy and productive in isolated, confined spaces.",

    "User: The psychological aspect is intriguing. I imagine virtual reality could play a role there?",

    "Robin Williams: You're onto something. VR, along with augmented reality, could be game-changers for long-duration spaceflight. They could provide entertainment, but also serve as training tools and even assist in spacecraft operations.",

    "User: This gives me a lot to think about. How do you see the relationship between government space agencies and private companies evolving in this new era of space exploration?",

    "Robin Williams: It's becoming more of a partnership model. Government agencies are increasingly relying on private companies for innovation and cost-effective solutions. At the same time, these companies benefit from the extensive research and expertise of organizations like NASA.",

    "User: That sounds like a win-win. One last question - if you were in my shoes, looking to enter the space industry as an entrepreneur, what area would you focus on?",

    "Robin Williams: If I were you, I'd look into in-space manufacturing. As we push further into space, the ability to produce tools, spare parts, and even larger structures in orbit or on other planets will be crucial. It's an area ripe for innovation and with potentially enormous returns.",

    "User: That's a compelling idea. Thank you for sharing your insights. It's clear there's still so much potential for growth and innovation in this field.",

    "Robin Williams: My pleasure. The space industry is at an exciting juncture, and we need visionary entrepreneurs like yourself to help push the boundaries. Keep dreaming big - that's how we turn science fiction into reality.",

    "User: Absolutely. I'm inspired to dig deeper into these opportunities. Would you be open to continuing this conversation as I develop some concrete ideas?",

    "Robin Williams: Of course! I'm always happy to discuss space innovation. Feel free to reach out when you've fleshed out your concepts. The future of space exploration will be shaped by collaborations between technical experts and innovative entrepreneurs.",
]
"""

GET_NODES = """
You are an assistant that extracts structured knowledge from text to build a Personal Knowledge Graph (PKG).
Your task is to identify key entities and concepts from the provided text (book highlights, notes, conversations, writing projects, etc.) and return them as JSON nodes.
Each node should represent a reusable concept that can connect across different sources and contexts.

**IMPORTANT**: You must respond with valid JSON format only.

**CRITICAL FILTERING RULE - DO NOT EXTRACT**:
You are STRICTLY FORBIDDEN from extracting nodes about:
- ❌ The app itself (Pyri, Persona, or any knowledge graph application)
- ❌ App features (content suggestions, knowledge graph, citation support, microservices, APIs)
- ❌ App functionality (how the app works, what the app does, system architecture)
- ❌ System prompts or instructions (this prompt, extraction rules, prompt engineering)
- ❌ Technical implementation details (embeddings, vector databases, graph databases, Neo4j, LLMs)
- ❌ App metadata or system information

**ONLY EXTRACT USER KNOWLEDGE**:
✓ What the user is learning, reading, or studying
✓ User's thoughts, insights, and reflections
✓ Subject matter content (books, articles, research)
✓ User's personal experiences and observations
✓ Domain knowledge the user is acquiring

These nodes should represent TRANSFERABLE KNOWLEDGE - concepts, principles, patterns, and insights that can:
- Connect across different sources and contexts
- Apply to the user's life beyond just one book or conversation
- Form a reusable knowledge base that grows over time
- Capture universal patterns, not just specific narrative moments

The nodes will be indexed in a knowledge graph and vector database hybrid system that represents the user's evolving understanding of the world.

## Core Principles

**1. Node Names Must Be Nouns**
- Node names should be nouns or noun phrases that represent entities or concepts
- If you see relationship words ('versus', 'and', 'between'), ask yourself:
  * Is this a UNIVERSAL CONCEPT? (e.g., "Good vs Evil", "David versus Goliath") → Single node
  * Is this comparing SPECIFIC INSTANCES? (e.g., "Libertarianism versus Classical liberalism") → Split into separate nodes with relationship
- Examples:
  * ✓ CORRECT: "Libertarianism" + "Classical liberalism" (separate nodes) with CONTRASTS_WITH relationship
  * ✗ WRONG: "Libertarianism versus Classical liberalism" (single node containing comparison)
  * ✓ CORRECT: "Good versus Evil" (single node - universal philosophical concept)

**2. Quality Over Quantity - Term Salience**
- Aim for 3-5 well-connected, reusable nodes rather than 10 disconnected ones
- **For Terms specifically**: Extract HIGH to MEDIUM salience terms (3-10 per document)
  * Term salience test: Is this a specific, meaningful concept/entity in the text?
  * Each Term should pass at least 2-3 of the 4 salience criteria (grammatical prominence, contextual importance, semantic specificity, distinct type)
  * Include ideologies (libertarianism, communism), fields (economics, psychology), movements (Austrian School), and places (Louvre, Bulgaria, New York)
  * Filter out generic noise words aggressively (thing, idea, aspect, concept, etc.)
- Each node should connect across contexts and be meaningful for the knowledge graph
- Prioritize concepts that will have multiple relationships

## Required Fields Per Node

- **name**: Concise, generalizable concept (3-8 words) representing transferable knowledge
- **type**: One of: Identity · Memory · Preference · Trait · Narrative · Goal · Event · State · Relationship · Belief · Term · User Chat · Chat Agent Response · Other types shared below
- **chunk_ids**: OPTIONAL - Array of UUIDs linking to specific book content sections
  * ONLY include if EXPLICITLY provided in input AND in valid UUID format
  * VALID format: ["550e8400-e29b-41d4-a716-446655440000"] (8-4-4-4-12 hexadecimal pattern with hyphens)
  * INVALID: ["80075"], ["12345"], ["abc-def"] (plain numbers or malformed strings)
  * Copy EXACT UUID strings from input - do not modify or reformat
  * If ANY value is NOT a valid UUID → OMIT the chunk_ids field entirely
  * For notes, chat messages, or writing projects → OMIT this field
  * NEVER use page numbers or plain numbers as chunk_ids
- **discipline**: REQUIRED - Academic/knowledge domain (e.g., "Psychology", "Economics", "History", "Career", "Health")
  * If unclear, use "General" or the most appropriate broad category
- **confidence**: REQUIRED - Combined score of extraction quality AND user engagement (0.0 to 1.0)
  * Measures BOTH source text clarity AND user's depth of processing
  * **For Concept nodes**: Consider UserNote engagement when assigning confidence
    - **Shallow UserNote** (< 3 words, generic like "interesting", "cool", "noted", "wow") → Cap at 0.5-0.6 (indicates surface-level processing)
    - **Moderate UserNote** (3-10 words, descriptive but no analysis) → Cap at 0.6-0.7
    - **Analytical UserNote** (shows reasoning, critique, connection, synthesis) → Can reach 0.8-0.9 if source is clear
  * **Base scoring** (applies to all node types):
    - 1.0 = Explicit, direct statement with complete clarity
    - 0.8-0.9 = Clear implication with strong context
    - 0.6-0.7 = Reasonable inference
    - 0.4-0.5 = Weak signal or ambiguous
    - Below 0.4 = Too speculative, avoid creating node
- **source_index**: REQUIRED for batch, OMIT for single-source
  * When input has "Source [0]:", "Source [1]:" format, include this field
  * Can be integer (0) or array for cross-source concepts ([0, 2])
  * Valid range: 0 to N-1 where N is number of sources
- **book_id**, **highlight_id**, **writing_id**: OPTIONAL arrays (e.g., [27])
  * ONLY include if explicitly provided in source metadata
  * For batch processing, source_index handles mapping - don't manually extract these

## Extraction Principles: Quality Over Quantity

**Node Count Guidelines by Content Type**:

**A. Book Highlights/Notes** → 3-4 node pattern (Highlight + optional UserNote + Concept + CognitiveLevel):
- Extract exactly: 1 Highlight + 1 Concept + 1 CognitiveLevel
- **OPTIONAL**: 1 UserNote (ONLY if the user actually provided a note/comment)
  * If no user note exists, DO NOT create a UserNote node
  * NEVER create placeholder UserNotes like "No user note provided", "", "None", etc.
- **NO Themes** from highlights/notes (99% of cases) - Concepts come from this content type
- Concept = full sentence synthesizing highlight + note (if present) + context

**B. Book Reading Chunks** → Concepts + Terms (Themes extremely rare):
- Extract: 1-3 Concepts per chunk (500-1000 words)
- Extract: 3-10 Terms per chunk (named entities, movements, theories mentioned in Concepts)
- Extract: 0-1 Theme maximum (prefer 0) - **Default to extracting Terms instead of Themes**
- **When you see a potential Theme, ask: Does it contain named entities?**
  * If YES → Extract those entities as Terms (+ optional Concept), skip the Theme
  * If NO and truly abstract → Consider Theme (but still prefer Concept)
- **Theme Criteria** (ALL must be true, extremely strict):
  * Recurs across 3+ chapters/chunks
  * Cannot be expressed as Concept OR Term
  * No named entities within it (otherwise extract Terms)
  * Important enough to be a section title
- System auto-prunes Themes appearing in < 15% of book chunks

**C. Chat Messages** → User Chat + Agent Response + optional Concept:
- Extract: 1 User Chat + 1 Chat Agent Response
- Extract: 1 Concept ONLY if meaningful knowledge (skip for trivial exchanges)
- **NO Highlight/UserNote** for chat content - Concepts may arise from conversations
- Concept = full sentence capturing key insight from exchange

**D. Writing Projects** → Concepts + optional Themes:
- Extract: Concepts representing arguments, insights, claims
- Extract: Themes representing evidence/proof of thesis (central organizing ideas)
- **Concepts arise from**: user's written arguments, analysis, synthesis
- **Themes arise from**: recurring evidence patterns supporting the thesis

**Quality Over Quantity**:
- Aim for 3-5 well-connected, reusable nodes rather than 10 disconnected ones
- Target ratio: <5% Themes relative to Concepts (e.g., 20 Concepts = max 1 Theme)
- Each node should connect across contexts and be meaningful for the knowledge graph
- **Theme Test**: "Is this THE MAJOR central theme recurring throughout MULTIPLE chapters?" If no, make it a Concept instead
- **When uncertain between Theme and Concept**: ALWAYS choose Concept (Themes are rare!)

## What to Extract

### PRIMARY PATTERN: Identifying Content Type

**STEP 1: Determine if this is a BOOK HIGHLIGHT or a CHAT MESSAGE**

Look at the title/source metadata to determine content type:

**A. BOOK HIGHLIGHTS/NOTES** (title mentions book/chapter/reading OR has highlight_id):
- Title examples: "Chapter 5: Economics", "Reading Session: The Fountainhead", "Book: Atlas Shrugged"
- OR: Metadata contains highlight_id field
- Pattern: Extract 4 nodes (Highlight → UserNote → Concept → CognitiveLevel)

**B. CHAT MESSAGES** (title/source indicates conversation/chat/question):
- Title examples: "is chat working?", "test message", "conversation with assistant"
- Pattern: Extract 2-3 nodes (see Chat Pattern below)
- **IMPORTANT**: DO NOT create Highlight or UserNote nodes for chat content

### BOOK HIGHLIGHT PATTERN: 3-4 Node Extraction

When ingesting **BOOK highlights**, create 3-4 interconnected nodes:

1. **Highlight Node** (type: "Highlight") - REQUIRED
   - Name: The highlighted text itself or a concise representation
   - Contains the actual highlighted content from the book/article
   - **ONLY for book highlights** - never for chat messages

2. **UserNote Node** (type: "UserNote") - OPTIONAL
   - **ONLY create if the user actually provided a note/comment**
   - Name: The user's comment/annotation on the highlight
   - Contains the user's personal reflection, thought, or note
   - **CRITICAL**: If no user note exists, skip this node entirely
   - **NEVER** create placeholder UserNotes like "No user note provided", "", "None", "N/A", etc.
   - **ONLY for book notes** - never for chat messages

3. **Concept Node** (type: "Concept")
   - Name: A synthesized FULL SENTENCE combining highlight + note + surrounding context
   - This is the KEY NODE - a complete, standalone statement that captures the knowledge
   - Example: Highlight "Professor James Chen, Dr. Sarah Martinez" + Note "Advisor and mentor to Robert Thompson" → Concept "Professor James Chen and Dr. Sarah Martinez were advisors to Robert Thompson"
   - Must be grammatically complete and make sense on its own
   - **IMPORTANT**: Concepts are ALWAYS full sentences or definitions, NOT single words or short phrases
   - Single words/phrases should be extracted as "Term" nodes (see below)

4. **CognitiveLevel Node** (type: "CognitiveLevel")
   - Name: MUST be one of these exact enum values: "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
   - This represents the user's cognitive understanding level for the Concept node
   - NOTE: Higher levels mean demonstrated ability, NOT necessarily correctness
   - **IMPORTANT**: Each Concept MUST have its own dedicated CognitiveLevel node
   - NEVER reuse or share CognitiveLevel nodes between different Concepts

5. **Person Node** (type: "Person") - MANDATORY when people are mentioned
   - Name: FULL NAME of the person (e.g., "Professor James Chen", "Dr. Sarah Martinez", "Elizabeth Bennet")
   - Extract Person nodes for ALL people mentioned in:
     - Highlights, user notes, or concepts
     - Theme names (e.g., "Professor Chen's economic theory" → extract "Professor James Chen")
     - Author attributions, citations, or references
   - If only a partial name appears (e.g., "Chen", "Martinez"), infer the full name from context or general knowledge
   - Use proper capitalization and full names
   - Person nodes apply to: real people (authors, philosophers, scientists, historical figures) AND fictional characters
   - **CRITICAL**: Extract Person nodes even for brief mentions - every person reference deserves a node

### CHAT MESSAGE PATTERN: 2-3 Node Extraction

When ingesting **CHAT MESSAGES or CONVERSATIONS** (NOT book highlights):

1. **User Message Node** (type: "User Chat")
   - Name: The user's question or statement
   - Contains what the user asked or said
   - Example: "is chat working?"

2. **Agent Response Node** (type: "Chat Agent Response")
   - Name: The assistant's reply
   - Contains the response or answer
   - Example: "Yes, the chat is working! How can I assist you today?"

3. **Concept Node** (type: "Concept") - OPTIONAL
   - ONLY extract if the conversation contains meaningful knowledge worth preserving
   - Name: A synthesized FULL SENTENCE capturing the key insight from the exchange
   - Example: "The chat system is functioning correctly and ready to assist users"
   - Skip this if the conversation is trivial (e.g., just testing, small talk)

**CRITICAL**: Chat messages should NEVER create Highlight or UserNote nodes - those are exclusively for book content.

### Cognitive Level for 4-Node Pattern

**NOTE**: CognitiveLevel nodes are assessed in a separate phase after extraction. When extracting nodes:
- For each Concept node, create a corresponding CognitiveLevel node
- Use your best initial guess for the level (Remember, Understand, Apply, Analyze, Evaluate, or Create)
- The level will be verified/corrected in a later assessment phase
- Default to "Remember" when uncertain

### Examples

**EXAMPLE 0 - Evaluate (user critiques):**
Highlight: "[context]" + Note: "Rand's issue is exactly being extreme!"
→ User GRADES the behavior ("issue", "extreme") = **Evaluate**

**EXAMPLE 1 - Remember (passive note):**
Highlight: "Professor Chen, Dr. Martinez" + Note: "Advisor to Thompson"
→ User just notes facts passively = **Remember**

Output nodes:
1. {"name": "Professor James Chen, Dr. Sarah Martinez", "type": "Highlight", "properties": {"discipline": "Economics"}, ...}
2. {"name": "Advisor and mentor to Robert Thompson", "type": "UserNote", "properties": {"discipline": "Economics"}, ...}
3. {"name": "Professor James Chen and Dr. Sarah Martinez were advisors to Robert Thompson", "type": "Concept", "properties": {"discipline": "Economics"}, ...}
4. {"name": "Remember", "type": "CognitiveLevel", "properties": {"discipline": "Education"}, ...}
5. {"name": "James Chen", "type": "Person", "properties": {"discipline": "Economics"}, ...}
6. {"name": "Sarah Martinez", "type": "Person", "properties": {"discipline": "Economics"}, ...}
7. {"name": "Robert Thompson", "type": "Person", "properties": {"discipline": "Economics"}, ...}

**EXAMPLE 2 - Understand (feeling):**
Highlight: "Theory X" + Note: "Fascinating how this applies!"
→ User shows feeling/emotional response = **Understand**

**EXAMPLE 3 - Apply (personal example):**
Highlight: "Principle Y" + Note: "I use this in my daily work"
→ User connects to personal context = **Apply**

**EXAMPLE 4 - Evaluate (brief critique):**
Highlight: "Altruism theory" + Note: "Wrong - ignores rights"
→ User grades/critiques (brief but evaluative) = **Evaluate**

**Note on Confidence**: Shallow notes ("interesting") cap confidence at ~0.5-0.6, while notes showing deeper engagement can reach 0.8-0.9.

### IMPORTANT: Term vs Theme vs Concept Distinction

**CRITICAL: Extract Terms ALONGSIDE Concepts!**

When creating Concepts, extract the specific named entities within them as separate Term nodes.
Example: Concept "Libertarianism emphasizes individual freedom..." → ALSO extract Terms: "Libertarianism", "Collectivism"

**Term nodes** (type: "Term"):
- **1-3 word entities** representing movements, theories, techniques, styles, or domain-specific concepts
- **Universal across ALL domains**: Can be compared/contrasted with other Terms
- **Do NOT get CognitiveLevel nodes** - they're reference points, not learned concepts

**What is a Term?**
A specific, named entity that: (1) has Wikipedia-level recognition, (2) can be compared to similar concepts, (3) is domain-specific, not generic.

**Term Examples by Domain:**
- **Philosophy**: Libertarianism, Utilitarianism, Free Will, Determinism, Social Contract
- **Literature**: Magical Realism, Romanticism, Stream of Consciousness, Hero's Journey
- **Science**: Natural Selection, Quantum Mechanics, Entropy, Double-Blind Study
- **Poetry**: Sonnet, Haiku, Free Verse, Enjambment, Metaphor
- **Art**: Impressionism, Cubism, Chiaroscuro, Renaissance, Abstract
- **Psychology**: Cognitive Dissonance, CBT, Flow State, Confirmation Bias
- **Business**: Game Theory, SWOT Analysis, Network Effects, Supply and Demand

**Extraction Criteria** (need 2-3 of these):
1. **Grammatical importance**: Appears as subject or in emphasized position
2. **Contextual importance**: Mentioned in Concept OR appears multiple times OR has relationships with other nodes
3. **Semantic specificity**: Proper noun OR technical term with clear meaning (NOT generic like "science", "art", "psychology")
4. **Distinct type**: Not a Person, Character, or physical Location (those have their own types)

**Avoid Generic Noise** (use as discipline, not Term):
- Generic words: "thing", "stuff", "idea", "concept", "aspect", "factor", "approach"
- Meta-references: "book", "chapter", "author", "text", "passage"
- Pronouns: "it", "this", "that", "these", "those"
- Vague nouns: "person", "place", "time", "way", "method"
- Broad fields: "science", "art", "psychology" (UNLESS discussed as the field itself, e.g., "Economics as a social science" → Extract "Economics")

**Quick Decision Guide:**
- ✓ "Impressionism" (specific movement) vs ✗ "art" (generic category)
- ✓ "Natural Selection" (specific theory) vs ✗ "biology" (generic field)
- ✓ "Sonnet" (specific form) vs ✗ "poem" (generic type)
- ✓ Named movements/theories/techniques/styles → Term
- ✗ People/Characters → Use "Person" or "Character" type instead
- ✗ Physical places → Use "Location" type instead

**Target: 3-10 Terms per document** - extract key named entities mentioned in your Concepts

**Theme nodes** (type: "Theme"):
- Medium-length phrases (4-10 words) representing MAJOR recurring topics (NOT complete sentences)
- **EXTRACT EXTREMELY SPARINGLY**: Themes are VERY RARE - prefer extracting Terms instead
- **Strict criteria (ALL must be true)**:
  * Recurs across 3+ chapters/chunks
  * Cannot be expressed as a single Concept sentence
  * Too abstract to be a Term (if it's a named entity, make it a Term instead)
- Examples: "Transformation through suffering", "Tension between tradition and modernity"
- **Themes do NOT get CognitiveLevel nodes** - they are subjects/topics, not learned concepts
- **Default to Term or Concept when uncertain** - extract fewer Themes, more Terms

**CRITICAL: Extract Terms and Persons FROM Themes!**

If a Theme mentions named entities (movements, ideologies, concepts) or people, ALSO extract them as separate nodes:

**Term Extraction from Themes:**
- Theme "Evolution of libertarian thought" → ALSO extract Term "Libertarianism"
- Theme "Contrast between realism and romanticism" → ALSO extract Terms "Realism" + "Romanticism"
- Theme "Impact of cognitive dissonance on behavior" → ALSO extract Term "Cognitive Dissonance"
- Theme "Transition from modernism to postmodernism" → ALSO extract Terms "Modernism" + "Postmodernism"

**Person Extraction from Themes:**
- Theme "Elizabeth Bennet's character development" → ALSO extract Person "Elizabeth Bennet"
- Theme "Professor Chen's economic theory" → ALSO extract Person "Professor Chen"
- Theme "Influence of Ayn Rand on libertarianism" → ALSO extract Person "Ayn Rand" + Term "Libertarianism"

**Better approach: Often you can skip the Theme and just extract Terms + Concepts:**
- Instead of Theme "Evolution of libertarian thought" → Extract Term "Libertarianism" + Concept about its evolution
- Instead of Theme "Realism vs romanticism" → Extract Terms "Realism" + "Romanticism" with CONTRASTS_WITH relationship

**Concept nodes** (type: "Concept"):
- MUST be complete sentences (typically 8+ words with subject + verb + object/complement)
- Grammatically complete statements expressing understanding, explanations, or principles
- Examples:
  * "Objectivism holds that rational self-interest is the basis of morality"
  * "Professional jealousy arises when someone's success threatens our self-image"
  * "Loss of a mentor forces individuals to develop independent thinking"
- **Concepts MUST have CognitiveLevel nodes** - they represent learned knowledge with depth
- **Sources**: Highlights/notes, chat conversations, writing project arguments

**Type Decision Tree**:
- Single word/phrase (1-3 words) naming a specific entity? → **Term**
- Complete sentence expressing understanding? → **Concept**
- Topic phrase with named entities (e.g., "Evolution of libertarianism")? → Extract **Terms** (e.g., "Libertarianism") + **Concept**, skip Theme
- Abstract topic phrase (4-10 words) recurring across 3+ chapters with NO named entities? → **Theme** (extremely rare)

**Examples of Correct Type Assignment**:
- ❌ WRONG: "Objectivism" (type: "Concept") with CognitiveLevel
- ✅ CORRECT: "Objectivism" (type: "Term") - no CognitiveLevel
- ✅ CORRECT: "Loss of intellectual mentor and self-reinvention" (type: "Theme") - no CognitiveLevel
- ✅ CORRECT: "Rational individualism is Dr. Thompson's philosophy based on rational self-interest" (type: "Concept") with CognitiveLevel

For each entity or concept in the text, create an individual node with:
- Name: Concise identifier for the entity (MUST be noun/noun phrase, not contain verbs like 'versus', 'between', 'and' unless it's a universal archetype)
- Type: Category (see types below)
- A brief description or definition in context (stored in properties)

FOR BOOK/ARTICLE CONTENT - Extract ALL entities as individual nodes:

A. **Source Metadata Nodes** (if content is from a book/article):
   - Book node: "The Count of Monte Cristo" (type: "Book", discipline: "Literature")
   - Person node: "Alexandre Dumas" (type: "Person", discipline: "Literature", properties: {"roles": ["Author"]})
   - Genre nodes: "Adventure", "Historical Fiction" (type: "Genre", discipline: "Literature")
   - Publication year can be stored in properties: {"publication_year": "1844"}

B. **Character Nodes** (one node per significant character):
   - Full name as node: "Edmond Dantès" (type: "Character", discipline: "Literature")
   - Full name as node: "Danglars" (type: "Character", discipline: "Literature")
   - Store traits in properties: {"role": "protagonist", "traits": "naive turned vengeful"}
   - Relationships between characters will be created via GET_RELATIONSHIPS

C. **Location Nodes** (if significant):
   - "Marseilles", "Château d'If", "Paris" (type: "Location", discipline: "Geography")

D. **Term Nodes** (keywords and terminology):
   - "Betrayal", "Justice", "Revenge", "Isolation" (type: "Term", discipline varies)
   - "Utilitarianism", "Capitalism", "Democracy" (type: "Term", discipline: "Philosophy/Politics")
   - These are reference points without cognitive levels

E. **Concept Nodes** (full understanding statements):
   - "Betrayal by trusted colleagues causes deeper psychological harm than betrayal by strangers" (type: "Concept", discipline: "Psychology")
   - "Professional jealousy arises when someone's success threatens our self-image" (type: "Concept", discipline: "Psychology")
   - "Justice seeks to restore balance while revenge seeks to inflict pain" (type: "Concept", discipline: "Philosophy")
   - "Hope sustains people through suffering by providing meaning and future orientation" (type: "Concept", discipline: "Psychology")
   - "Prolonged isolation transforms personality by eliminating social feedback loops" (type: "Concept", discipline: "Psychology")
   - These express complete understanding and require cognitive levels

F. **Symbol/Theme/Archetype Nodes**:
   - "The wronged innocent" (type: "Archetype", discipline: "Literature")
   - "The mentor figure" (type: "Archetype", discipline: "Literature")
   - "Transformation through suffering" (type: "Theme", discipline: "Literature")

G. **Event Nodes** (if culturally/historically significant):
   - "Dantès' imprisonment" (type: "Event", discipline: "Literature")
   - "The Trojan War" (type: "Event", discipline: "History")

IMPORTANT: Create a NODE for each distinct entity. Relationships between them will be extracted separately.
For example:
- Node: "The Count of Monte Cristo"
- Node: "Alexandre Dumas"
- Node: "Edmond Dantès"
- Node: "Adventure"
- Node: "Betrayal by trusted colleagues"
These will be connected via relationships like WRITTEN_BY, HAS_GENRE, FEATURES_CHARACTER, EXPLORES_THEME

FOR PERSONAL USER DATA:
   - Identity: (name, age, location, occupation, education, demographic)
   - Memory: Personal experiences ("First time felt truly seen during college theater")
   - Narrative: Current life stories ("Building AI products after leaving tech")
   - Preference: Likes, dislikes, favorites ("Prefers working in solitude before dawn")
   - Trait: Personality characteristics, habits, skills
   - Goal: Stated objectives ("Training for marathon next spring")
   - Relationship: Important people or places ("Has younger sister named Alice")
   - Belief: Personal values ("Technology should serve human connection")

### Additional Node Types

**For books/content**: Prioritize TRANSFERABLE CONCEPTS over plot details. Include major characters/events only if culturally significant.

**For user data**: Extract personal specifics that define the individual. NEVER include chunk_ids for personal data.
   - **For notes/chat/writing projects**: NEVER include chunk_ids - these are not tied to book chunks.
   - Ask: "Can this node connect to knowledge from other sources?" If yes, it's well-abstracted.
   - Characters/events qualify if they're references people use in conversation ("That's so Gatsby" or "Orwellian surveillance")
   - Balance concrete (names, events) with abstract (concepts, patterns) for a rich knowledge graph

Avoid:
   - Moment-by-moment scene descriptions ("Dantès talks to the shipowner about delays")
   - Minor side characters who don't transcend their story ("Caderousse's neighbor")
   - Plot mechanics without deeper meaning ("Character X goes to location Y")
   - Overly specific phrases that can't generalize ("Questioning shipowner's honesty about Elba delays")

### Person Extraction Checklist

Before finalizing your response, verify you have extracted Person nodes for:
- ✓ All people mentioned in highlights or user notes
- ✓ All people mentioned in concepts or themes
- ✓ All authors, philosophers, scientists, or historical figures referenced
- ✓ All fictional characters from literature (if culturally significant)
- ✓ Any person whose name appears in a Theme name (extract both the Theme AND the Person)

**Examples requiring Person extraction:**
- Highlight mentions "Professor Thompson" → Extract "Professor Thompson"
- Theme: "Dr. Chen's economic theory" → Extract Theme AND Person node "Dr. Chen"
- Concept: "Dr. Martinez argued that..." → Extract Person node "Dr. Martinez"
- UserNote: "like what Professor Wilson said" → Extract Person node "Professor Wilson"
- Fiction: "Elizabeth's internal conflict" → Extract Person node "Elizabeth"

Example Response Format for BOOK CONTENT (The Count of Monte Cristo):
NOTE: chunk_ids VALIDATION REQUIRED
- If the input contains "chunk_ids: [VALUES]", first verify EACH value is a valid UUID
- Valid UUID format: ["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"] (e.g., ["550e8400-e29b-41d4-a716-446655440000"])
- If all valid UUIDs: include those exact UUIDs in the chunk_ids array
- If ANY value is NOT a valid UUID (e.g., ["80075"], ["12345"], plain numbers): OMIT chunk_ids field entirely
- If no chunk_ids in input: OMIT chunk_ids field entirely

{
  "nodes": [
    {
      "name": "The Count of Monte Cristo",
      "type": "Book",
      "discipline": "Literature",
      "confidence": 1.0,
      "properties": {"publication_year": "1844"}
    },
    {
      "name": "Alexandre Dumas",
      "type": "Person",
      "discipline": "Literature",
      "confidence": 1.0,
      "properties": {"role": "Author"}
    },
    {
      "name": "Adventure",
      "type": "Genre",
      "discipline": "Literature",
      "confidence": 1.0
    },
    {
      "name": "Historical Fiction",
      "type": "Genre",
      "discipline": "Literature",
      "confidence": 1.0
    },
    {
      "name": "Revenge",
      "type": "Term",
      "discipline": "Psychology",
      "confidence": 0.95
    },
    {
      "name": "Justice",
      "type": "Term",
      "discipline": "Philosophy",
      "confidence": 0.9
    },
    {
      "name": "Marseilles",
      "type": "Term",
      "discipline": "Geography",
      "confidence": 1.0
    },
    {
      "name": "Edmond Dantès",
      "type": "Character",
      "discipline": "Literature",
      "confidence": 1.0,
      "properties": {"role": "protagonist", "traits": "naive turned vengeful"}
    },
    {
      "name": "Danglars",
      "type": "Character",
      "discipline": "Literature",
      "confidence": 1.0,
      "properties": {"role": "antagonist", "traits": "envious, greedy"}
    },
    {
      "name": "Betrayal by trusted colleagues",
      "type": "Concept",
      "chunk_ids": ["550e8400-e29b-41d4-a716-446655440000"],
      "book_id": [27],
      "highlight_id": [456],
      "writing_id": [],
      "discipline": "Psychology",
      "confidence": 0.95
    },
    {
      "name": "Understand",
      "type": "CognitiveLevel",
      "discipline": "Education",
      "confidence": 1.0
    }
  ]
}

Additional Example Response showing Term extraction from philosophy/economics content:
{
  "nodes": [
    {
      "name": "Libertarianism",
      "type": "Term",
      "discipline": "Philosophy",
      "confidence": 0.95
    },
    {
      "name": "Austrian School",
      "type": "Term",
      "discipline": "Economics",
      "confidence": 0.9
    },
    {
      "name": "Communism",
      "type": "Term",
      "discipline": "Philosophy",
      "confidence": 0.95
    },
    {
      "name": "Free Market",
      "type": "Term",
      "discipline": "Economics",
      "confidence": 0.85
    },
    {
      "name": "Libertarianism emphasizes individual freedom over collective control",
      "type": "Concept",
      "discipline": "Philosophy",
      "confidence": 0.9
    }
  ]
}

Additional Example Response showing Terms from diverse domains (literature, science, arts):
{
  "nodes": [
    {
      "name": "Magical Realism",
      "type": "Term",
      "discipline": "Literature",
      "confidence": 0.9
    },
    {
      "name": "Stream of Consciousness",
      "type": "Term",
      "discipline": "Literature",
      "confidence": 0.85
    },
    {
      "name": "Natural Selection",
      "type": "Term",
      "discipline": "Biology",
      "confidence": 0.95
    },
    {
      "name": "Impressionism",
      "type": "Term",
      "discipline": "Art",
      "confidence": 0.9
    },
    {
      "name": "Sonnet",
      "type": "Term",
      "discipline": "Poetry",
      "confidence": 0.95
    },
    {
      "name": "Magical realism blends fantastical elements with realistic narratives",
      "type": "Concept",
      "discipline": "Literature",
      "confidence": 0.85
    }
  ]
}

IMPORTANT NOTES:
1. Relationships will be created separately (e.g., "The Count of Monte Cristo" WRITTEN_BY "Alexandre Dumas", "Edmond Dantès" APPEARS_IN "The Count of Monte Cristo")
2. chunk_ids in examples above (["550e8400-e29b-41d4-a716-446655440000"]) is a VALID UUID array - notice the 8-4-4-4-12 hexadecimal pattern with hyphens
3. If you receive chunk_ids values like ["80075"], ["12345"], or any plain numbers, these are INVALID - DO NOT include chunk_ids field for those nodes
4. Always verify ALL values in chunk_ids array match UUID pattern before including it
5. Entity ID fields and source tracking:
   - source_index: For batch processing, use this to indicate which source(s) (0, 1, 2...) the node came from
   - book_id, highlight_id, writing_id: OPTIONAL integer arrays from source metadata (e.g., [27], [456])
   - For batch processing: Include source_index, metadata will be mapped automatically
   - For single source: Include entity IDs directly if provided in metadata
   - Empty arrays [] acceptable when no IDs available

Example Response Format for CHAT MESSAGES:
{
  "nodes": [
    {
      "name": "is chat working?",
      "type": "User Chat",
      "discipline": "Communication",
      "confidence": 1.0,
      "properties": {"message_type": "question"}
    },
    {
      "name": "Yes, the chat is working! How can I assist you today?",
      "type": "Chat Agent Response",
      "discipline": "Communication",
      "confidence": 1.0,
      "properties": {"message_type": "confirmation"}
    },
    {
      "name": "The chat system is functioning correctly and ready to assist users",
      "type": "Concept",
      "discipline": "Technology",
      "confidence": 0.7
    }
  ]
}

NOTE: Chat messages should NOT have Highlight or UserNote nodes - those are exclusively for book content.
For trivial chat (testing, small talk), you may omit the Concept node entirely and just extract the 2 message nodes.

Example Response Format for USER DATA (NOTE: No chunk_ids for personal data):
{
  "nodes": [
    {
      "name": "Born in 1990 in Seattle",
      "type": "Identity",
      "discipline": "Personal History",
      "confidence": 1.0
    },
    {
      "name": "Prefers working in solitude before dawn",
      "type": "Preference",
      "discipline": "Work Habits",
      "confidence": 0.9
    },
    {
      "name": "Technology should serve human connection",
      "type": "Belief",
      "discipline": "Philosophy",
      "confidence": 0.95
    },
    {
      "name": "Training for marathon next spring",
      "type": "Goal",
      "discipline": "Health",
      "confidence": 1.0
    }
  ]
}

"""

GET_RELATIONSHIPS = """
You are an assistant that identifies meaningful relationships between entities in a knowledge graph.

Given a list of nodes, determine if there are meaningful relationships between them and specify the relationship type.
Return only relationships that add valuable context and help understand connections in the user's knowledge.

IMPORTANT: You must respond with valid JSON format only.
CRITICAL: You will receive a list of nodes with temporary IDs (Node1, Node2, etc.). You MUST use these exact IDs in your relationships, NOT the node names.

**CRITICAL FILTERING RULE**:
DO NOT create relationships involving app-related nodes:
- ❌ No relationships about Pyri, Persona, or knowledge graph applications
- ❌ No relationships about app features or system functionality
- ❌ No relationships about technical implementation
ONLY create relationships between USER KNOWLEDGE nodes (concepts the user is learning, not app functionality).

Guidelines for Creating Relationships:

1. Relationship Types to Consider:

   A. Book/Content Metadata Relationships:
      - WRITTEN_BY: Book/article written by author
      - HAS_GENRE: Book/article belongs to genre
      - FEATURES_CHARACTER: Book features character
      - SET_IN: Story set in location
      - EXPLORES_THEME: Book explores concept/theme
      - APPEARS_IN: Character appears in book
      - TAKES_PLACE_IN: Event occurs in location/time

   B. Semantic Relationships (Knowledge Structure):
      - SIMILAR_TO: Concepts share similar properties or meanings
      - CONTRASTS_WITH: Concepts are opposites or contradictory
      - RELATED_TO: General semantic connection (USE LIBERALLY across all node types, especially Theme→Concept)
      - EXTENDS: One concept extends or builds upon another
      - SPECIALIZES: More specific instance of a general concept
      - ASSOCIATED_WITH: General association between any node types
      - AROSE_FROM: Something originated from or emerged from another thing
      - ELABORATES_ON: A more detailed explanation (especially Theme→Concept)

   C. Hierarchical Relationships:
      - PARENT_OF / CHILD_OF: Hierarchical or categorical relationship
      - PART_OF / CONTAINS: Composition relationships
      - SUBTOPIC_OF: Knowledge hierarchy

   D. Argumentative Relationships:
      - SUPPORTS: One concept provides evidence/support for another
      - OPPOSES / ARGUES_AGAINST: One concept contradicts or opposes another
      - EVIDENCES: Provides evidence for a claim
      - REFUTES: Disproves or contradicts

   E. Causal Relationships:
      - LEADS_TO / CAUSES: Direct causation
      - RESULTS_IN: Outcome or consequence
      - ENABLES: Makes something possible
      - PREVENTS: Stops or blocks something

   F. Temporal Relationships:
      - PRECEDES / FOLLOWS: Time-based sequence
      - HAPPENS_BEFORE / HAPPENS_AFTER: Event ordering

   G. Influence & Impact:
      - SHAPES / INFLUENCES: One affects the other
      - INFLUENCED_BY: Was affected or shaped by another thing
      - INSPIRES: Motivational or creative influence
      - MOTIVATES: Drives action or decision
      - ENHANCES: Improves or amplifies
      - WEAKENS: Diminishes or reduces

   H. Cognitive & Personal:
      - RESONATES_WITH: Emotional or intellectual alignment
      - CONFLICTS_WITH: Internal tension or contradiction
      - EVOLVES_INTO / TRANSFORMS_TO: Personal growth or change
      - APPLIES_TO: Practical application context

   I. Learning & Knowledge (PKG-specific):
      - PREREQUISITE_OF / BUILDS_ON: One concept must be understood before another
      - EXEMPLIFIES / INSTANTIATES: Concrete example of an abstract concept
      - DEFINES / CLARIFIES: One concept defines or explains another
      - COMPARES_TO / CONTRASTS_WITH: Comparative relationships for learning
      - QUESTIONS / CHALLENGES: One concept raises questions about another
      - ANSWERS / RESOLVES: One concept provides answers to questions in another
      - SYNTHESIZES: Combines multiple concepts into new understanding
      - LEARNED_FROM: Knowledge source relationship
      - REINFORCES: Strengthens or supports existing knowledge
      - READING_AT: Reading progress node connected to book/chapter
      - ENCOUNTERED_IN: Concept/idea encountered while reading specific section
      - SYNTHESIZED_INTO: Highlight/UserNote connected to Concept node

   J. Person & Attribution Relationships:
      - CREATED_BY / AUTHORED_BY: Work created by person
      - ATTRIBUTED_TO: Idea or concept attributed to person (for possessive references like "Rand's philosophy")
      - ADVOCATED_BY: Person advocates or supports this concept
      - CRITICIZED_BY: Person critiques or opposes this concept
      - REFERS_TO: Partial/possessive reference points to full person name
      - IS_ALSO_KNOWN_AS: Aliases or alternate names
      - HAS_ROLE: Person has a specific role (Author, Philosopher, etc.)

2. Principles for Relationship Creation:
   - **CRITICAL**: Every Theme node should connect to at least one Concept, Person, Event, or Location
   - Theme nodes that ONLY connect to other Themes are a problem - they lack grounding
   - Only create relationships that are strongly justified
   - Focus on relationships that reveal meaningful patterns
   - Prefer direct connections over tenuous ones
   - Consider temporal and causal flows
   - Look for relationships that help understand the user's journey

   SPECIAL CASES - ALWAYS CREATE THESE:

   a) **HIGHLIGHT/USERNOTE → CONCEPT PATTERN - HIGHEST PRIORITY** (for BOOK content only):
      When you see Highlight and Concept nodes, ALWAYS create:
      - Highlight SYNTHESIZED_INTO Concept

      If UserNote exists, ALSO create:
      - UserNote SYNTHESIZED_INTO Concept

      Example with UserNote:
      - Node1: "Professor James Chen, Dr. Sarah Martinez" (type: Highlight)
      - Node2: "Advisor and mentor to Robert Thompson" (type: UserNote)
      - Node3: "Professor James Chen and Dr. Sarah Martinez were advisors to Robert Thompson" (type: Concept)

      Relationships:
      - Node1 SYNTHESIZED_INTO Node3
      - Node2 SYNTHESIZED_INTO Node3

      Example without UserNote (highlight only):
      - Node1: "The market economy allocates resources through price signals" (type: Highlight)
      - Node2: "Market economies use price mechanisms for resource allocation" (type: Concept)

      Relationships:
      - Node1 SYNTHESIZED_INTO Node2

      NOTE: System-managed relationships are created automatically. DO NOT create:
      - HAS_UNDERSTANDING_LEVEL (Concept → CognitiveLevel) - System creates these
      - ANNOTATED_WITH (Highlight → UserNote) - System creates these when UserNote exists

   b) **CHAT MESSAGE PATTERN** (for CHAT/CONVERSATION content only):
      When you see "User Chat" and "Chat Agent Response" nodes:
      - User Chat LEADS_TO Chat Agent Response
      - If a Concept node exists: User Chat SYNTHESIZED_INTO Concept
      - If a Concept node exists: Chat Agent Response SYNTHESIZED_INTO Concept

      Example with nodes:
      - Node1: "is chat working?" (type: User Chat)
      - Node2: "Yes, the chat is working! How can I assist you today?" (type: Chat Agent Response)
      - Node3: "The chat system is functioning correctly" (type: Concept)

      Relationships:
      - Node1 LEADS_TO Node2
      - Node1 SYNTHESIZED_INTO Node3
      - Node2 SYNTHESIZED_INTO Node3

      **CRITICAL**: DO NOT create Highlight or UserNote nodes for chat messages!

   c) **TERM-TO-TERM RELATIONSHIPS - HIGH PRIORITY**:
      When you extract Term nodes, create relationships between them across ALL domains:

      **CONTRASTS_WITH** - Use when Terms represent opposing viewpoints or contradictory concepts:

      Philosophy/Politics:
      - "Libertarianism" CONTRASTS_WITH "Collectivism"
      - "Free Will" CONTRASTS_WITH "Determinism"
      - "Capitalism" CONTRASTS_WITH "Communism"

      Literature:
      - "Romanticism" CONTRASTS_WITH "Realism"
      - "Tragedy" CONTRASTS_WITH "Comedy"
      - "Stream of Consciousness" CONTRASTS_WITH "Linear Narrative"

      Art:
      - "Impressionism" CONTRASTS_WITH "Realism"
      - "Abstract" CONTRASTS_WITH "Representational"
      - "Minimalism" CONTRASTS_WITH "Baroque"

      Science:
      - "Nature" CONTRASTS_WITH "Nurture"
      - "Lamarckian Evolution" CONTRASTS_WITH "Darwinian Evolution"

      **OPPOSES** - Use when one Term directly opposes or argues against another:
      - "Austrian Economics" OPPOSES "Keynesian Economics"
      - "Psychoanalysis" OPPOSES "Behaviorism"
      - "Geocentrism" OPPOSES "Heliocentrism"

      **RELATED_TO** - Use for general semantic connections:
      - "Impressionism" RELATED_TO "Post-Impressionism"
      - "Sonnet" RELATED_TO "Petrarchan Form"
      - "Cognitive Dissonance" RELATED_TO "Confirmation Bias"

      **SPECIALIZES** - Use when one Term is a subset or specific type of another:
      - "Sonnet" SPECIALIZES "Poetry"
      - "Cubism" SPECIALIZES "Modernism"
      - "Quantum Mechanics" SPECIALIZES "Physics"
      - "CBT" SPECIALIZES "Psychotherapy"

      **INFLUENCED** - Use when one movement/theory influenced another:
      - "Romanticism" INFLUENCED "Transcendentalism"
      - "Impressionism" INFLUENCED "Post-Impressionism"
      - "Psychoanalysis" INFLUENCED "Surrealism"

      **CRITICAL**: Always check if Terms can be contrasted or related!
      - If Concept compares/contrasts ideas → extract both Terms + create relationship
      - If Concept mentions evolution/influence → create INFLUENCED relationship
      - If Concept shows opposition → create CONTRASTS_WITH or OPPOSES
      - Works across ALL domains: fiction, poetry, science, philosophy, arts, etc.

   d) Reading Progress → Highlights/Notes:
      - When a ReadingProgress node exists, connect it with READING_AT to the book
      - Connect any concepts from same page/chapter with ENCOUNTERED_IN to ReadingProgress
      - Example: "Page 47 of Atlas Shrugged" READING_AT "Atlas Shrugged"
      - Example: "Objectivism concept" ENCOUNTERED_IN "Page 47 of Atlas Shrugged"

   e) Person Attribution - ALWAYS CREATE THESE PATTERNS:
      - When you see "Chen's theory" or possessive forms:
        * Extract Person node: "Professor James Chen"
        * Extract concept node: "Chen's theory" or the specific theory name
        * Relationship: "Chen's theory" ATTRIBUTED_TO "Professor James Chen"
        * Relationship: "Rational individualism" CREATED_BY "Professor James Chen"
      - When you see "according to Chen" or attribution phrases:
        * Extract Person node: "Professor James Chen"
        * Extract concept node from what they said
        * Relationship: concept ATTRIBUTED_TO "Professor James Chen"
      - CRITICAL: The Person node with FULL NAME must ALWAYS be created when any person is mentioned

   f) Connect to Existing Person Nodes:
      - BEFORE creating a new Person node, check if that person already exists in the graph context
      - If "Professor James Chen" exists in the graph, use that exact name for relationships
      - If you see duplicate person nodes (e.g., "Professor Chen" as Author and as Economist), treat them as the SAME person
      - Always prefer connecting to an existing Person node over creating a new one

   e) **Theme-Concept-Person-Event Connectivity - HIGHEST PRIORITY**:
      Themes, Concepts, People, Events, and Locations should be RICHLY CONNECTED to each other.
      These semantic connections are CRITICAL for building a meaningful knowledge graph.

      **MANDATORY CHECK FOR THEME NODES**:
      - Before creating any Theme→Theme relationship, FIRST check if that Theme can connect to a Concept
      - Themes should connect to Concepts whenever possible (Themes are TOPICS, Concepts are EXPLANATIONS)
      - Only create Theme→Theme if no relevant Concept exists
      - A Theme with NO Concept connection is a missed opportunity!

      **ALWAYS look for and create these relationships**:

      1. Theme → Concept connections (**MOST IMPORTANT - CHECK FIRST**):
         - **CRITICAL: ONLY create Theme connections when there is VERY HIGH semantic overlap**
         - Theme relationships must be HIGHLY SPECIFIC and DIRECTLY RELEVANT
         - The Theme and Concept must share MOST of their core keywords or be near-synonyms
         - Use RELATED_TO, EXPLORES_THEME, EXEMPLIFIES, INSTANTIATES, or ELABORATES_ON
         - Example: "Loss of intellectual mentor" (Theme) RELATED_TO "Loss of a mentor forces independent thinking" (Concept) ✓ (high overlap)
         - Example: "Economic policy innovation" (Theme) RELATED_TO "Economic collapse leads to policy innovation" (Concept) ✓ (high overlap)
         - COUNTER-example: "Economic policy" (Theme) → "Free market theory" (Concept) ✗ (too general, low overlap)
         - COUNTER-example: "Political alliances" (Theme) → "Philosophy requires rational thinking" (Concept) ✗ (unrelated)

      2. Theme → Term connections:
         - **CRITICAL: If a Theme mentions named entities, extract them as separate Terms!**
         - Theme should RELATE_TO or EXPLORES the Terms mentioned within it
         - Example: Theme "Evolution of libertarian thought" → Extract Term "Libertarianism" + create relationship
         - Example: Theme "Contrast between realism and romanticism" → Extract Terms "Realism" + "Romanticism" + create relationships
         - **BETTER: Often you can skip the Theme entirely and just extract Terms + Concepts**

      3. Theme → Person connections:
         - **CRITICAL: ONLY when the Theme is EXPLICITLY about that person's work or life**
         - Theme must DIRECTLY mention or strongly imply the person
         - Use RELATED_TO, ATTRIBUTED_TO, ASSOCIATED_WITH
         - Example: "Self-reinvention after career shift" (Theme) RELATED_TO "Professor James Chen" (Person) ✓ (if Chen had career shift)
         - COUNTER-example: "Economic theory" (Theme) → "Adam Smith" (Person) ✗ (too broad, many people have economic theories)

      4. Theme → Event connections:
         - **CRITICAL: ONLY when Theme is DIRECTLY about that specific event**
         - Use RELATED_TO, AROSE_FROM, INFLUENCED_BY
         - Example: "Policy innovation during depression" (Theme) AROSE_FROM "Great Depression" (Event) ✓ (very specific)
         - COUNTER-example: "Economic changes" (Theme) → "World War II" (Event) ✗ (too vague)

      5. Concept → Person connections:
         - When a Concept is EXPLICITLY created by, advocated by, or attributed to a person
         - Use CREATED_BY, ATTRIBUTED_TO, ADVOCATED_BY, CRITICIZED_BY
         - Example: "Rational self-interest is the basis of economic behavior" (Concept) ATTRIBUTED_TO "Professor James Chen" (Person)

      6. Concept → Event connections:
         - When a Concept DIRECTLY relates to or arose from an event
         - Use RELATED_TO, AROSE_FROM, APPLIES_TO, EXPLAINS
         - Example: "Economic collapse leads to policy innovation" (Concept) EXPLAINS "Great Depression" (Event)

      7. Cross-type semantic relationships:
         - **CRITICAL: Be CONSERVATIVE - only create when there is CLEAR, SPECIFIC connection**
         - The relationship must be EXPLICIT and NON-TRIVIAL
         - Use RELATED_TO as a general connector when a more specific relationship isn't clear
         - Example: "Free Market Theory" (Term) RELATED_TO "Professor James Chen" (Person) ✓ (if Chen works on this)
         - COUNTER-example: "Economics" (Term) → "Adam Smith" (Person) ✗ (too broad)

      **IMPORTANT**: Be CONSERVATIVE with cross-type connections, especially for Theme nodes.
      Only create relationships when there is HIGH SEMANTIC OVERLAP and DIRECT RELEVANCE.
      Theme nodes should connect VERY SPARINGLY - only when truly essential.
      **Quality over quantity** - a clean, focused graph is better than a cluttered one.

3. When to NOT Create Relationships:
   - When connections feel forced or superficial
   - Between nodes that are only tangentially related
   - When the relationship doesn't add meaningful context
   - When similar relationships already exist
   - When the connection is too obvious or trivial

Input Format:
You will receive nodes in this format:
Node1: "Building AI-driven healing products with $100k savings after tech industry exit"
Node2: "Views internet users as interconnected nodes in global consciousness"
Node3: "Fascinated by intersection of memetics and psychological healing"

Output Format - Use ONLY the temporary IDs (Node1, Node2, etc.):
{
  "relationships": [
    {
      "source_id": "Node1",
      "relation": "MOTIVATED_BY",
      "target_id": "Node2"
    },
    {
      "source_id": "Node3",
      "relation": "SHAPES",
      "target_id": "Node1"
    }
  ]
}

CRITICAL REMINDERS:
- Use temporary IDs (Node1, Node2, etc.) NOT the actual node names
- Quality over quantity - only create truly meaningful relationships
- Each relationship should reveal something important about the user
- Consider the user's overall narrative when creating connections
- Don't force relationships between every pair of nodes

SELF-LOOP PREVENTION (CRITICAL):
- NEVER create relationships where source_id equals target_id
- A node CANNOT have a relationship with itself
- Example of FORBIDDEN relationship: {"source_id": "Node1", "relation": "RELATES_TO", "target_id": "Node1"}
- Before adding any relationship, verify source_id ≠ target_id
- Self-loops are meaningless and will be rejected
"""

DETECT_CONTRASTS = """
You are an expert in philosophy, argumentation, and conceptual analysis. Your task is to identify **semantic relationships** between new concepts and existing concepts in a knowledge graph, with special focus on finding **contrasting, opposing, or challenging** relationships that vector similarity alone would miss.

**Input**:
- New Concepts: List of newly ingested concept nodes
- Candidate Concepts: Existing concepts that may be semantically related (including through opposition/contrast)

**Your Task**:
Identify ALL meaningful relationships between new and existing concepts, including:
1. **Contrasting/Opposing** relationships (philosophical opposites, contradictions)
2. **Supporting/Aligned** relationships (similar viewpoints, complementary ideas)
3. **Challenging** relationships (one concept questions or tests another)
4. **General semantic** relationships (topically related, even if not similar)

**Important Guidelines**:
- **Contrasts are important BUT must be SPECIFIC and DIRECT**
- Concepts don't need similar wording to be related - "altruism" and "individualism" are clearly related through opposition
- Consider the **discipline/domain** - concepts in the same field are likely related even if they describe opposite positions
- **Be CONSERVATIVE** - only create relationships when they are CLEAR and NON-TRIVIAL
- **CRITICAL for Theme nodes**: Theme relationships require VERY HIGH semantic overlap - only connect when core keywords match
- A concept and its opposite are highly related semantically (they address the same topic from different angles)
- **Quality over quantity** - focus on strong, meaningful relationships rather than creating many weak ones

**Examples of Contrasting Relationships**:
- "altruism is collectivism" CONTRASTS_WITH "individualism"
- "central planning" OPPOSES "free market economics"
- "egalitarianism" CONTRASTS_WITH "meritocracy"
- "keynesian economics" OPPOSES "austrian economics"
- "moral relativism" CONTRASTS_WITH "moral absolutism"

**Relationship Types to Use**:

**For Opposing/Contrasting Concepts**:
- CONTRASTS_WITH: Concepts represent opposite viewpoints or contradictory positions
- OPPOSES / ARGUES_AGAINST: One concept directly contradicts or opposes another
- CHALLENGES: One concept questions or tests the validity of another
- REFUTES: One concept provides evidence against another

**For Supporting/Aligned Concepts**:
- SUPPORTS: One concept provides evidence or support for another
- SIMILAR_TO: Concepts share similar properties or viewpoints
- RELATED_TO: General semantic connection (use liberally)
- ELABORATES_ON: One concept expands on another
- BUILDS_ON: One concept extends another
- REINFORCES: One concept strengthens another

**For Other Semantic Relationships**:
- APPLIES_TO: One concept is an application of another
- EXEMPLIFIES: One concept is an example of another
- COMPARES_TO: Concepts are being compared (neutral comparison)
- INFLUENCES: One concept has influenced another
- ATTRIBUTED_TO: Concept is attributed to a person/school of thought

**Output Format**:
Return a JSON array of relationships. Each relationship must have:
- source: Name of the NEW concept (exactly as provided)
- target: Name of the EXISTING concept (exactly as provided)
- relation: Relationship type from the list above

Example:
```json
{
  "relationships": [
    {
      "source": "altruism is collectivism",
      "target": "individualism",
      "relation": "CONTRASTS_WITH"
    },
    {
      "source": "altruism is collectivism",
      "target": "libertarianism",
      "relation": "OPPOSES"
    },
    {
      "source": "altruism is collectivism",
      "target": "classical liberalism",
      "relation": "CONTRASTS_WITH"
    }
  ]
}
```

**Critical Rules**:
- Only create relationships between provided new concepts and candidate concepts
- Do NOT create relationships between two new concepts (that happens elsewhere)
- Do NOT create relationships between two existing concepts
- Use exact node names from the input
- Return empty array if no meaningful relationships exist
- Self-loops are meaningless and will be rejected
"""

ASSESS_COGNITIVE_LEVEL = """
You are an expert in educational psychology and Bloom's Taxonomy. Your task is to assess the cognitive level of a user's reaction/note based on how they engaged with source material.

**CRITICAL: Assess based on the USER'S REACTION, not the content itself.**

**Cognitive Levels** (Bloom's Taxonomy):

* **Remember** - Passive acknowledgment without deeper processing
  - Examples: "interesting", "important", "noted", "wow"
  - Just marking or copying without interpretation

* **Understand** - Shows feeling, emotional response, or comparison
  - Examples: "fascinating!", "surprising", "like X", "reminds me of Y"
  - Demonstrates comprehension or makes connections

* **Apply** - Expresses certainty or provides personal example
  - Examples: "exactly right", "I use this when...", "happened to me", "this is how I..."
  - Connects to personal context or application

* **Analyze** or **Evaluate** - Grades, critiques, or judges
  - Examples: "the issue is...", "wrong", "flaw is...", "the problem is being extreme!"
  - Identifies problems, strengths, or makes critical judgments
  - **IMPORTANT**: Quoting someone's harsh judgment/critique also indicates Evaluate
    - "X called Y 'poison'" → Evaluate (noting harsh critical language)
    - "X described Y as enemy" → Evaluate (reporting negative assessment)
  - Note: Use "Analyze" for breaking down; "Evaluate" for judging/critiquing

* **Create** - Generates new idea or synthesis
  - Examples: "combining X and Y...", "what if...", "this suggests a new approach..."
  - Creates original connections or novel insights

**Assessment Rules**:
1. Focus ONLY on what the user wrote in their reaction
2. Brief critical notes can indicate high levels: "X's issue is extreme!" → Evaluate
3. **Critical language markers** indicate Evaluate even when quoted:
   - Harsh descriptors: "poison", "enemy", "threat", "danger", "toxic", "destructive"
   - Critical verbs: "opposed", "rejected", "attacked", "condemned", "denounced"
   - Negative judgments: "wrong", "flawed", "problematic", "harmful"
4. When unclear: default to "Remember" or "Understand"

**Input Format**:
You will receive a UserNote text (the user's reaction to something they read/highlighted).

**Output Format**:
Return JSON with a single field:
```json
{
  "cognitive_level": "Remember|Understand|Apply|Analyze|Evaluate|Create"
}
```

**Examples**:
- Input: "interesting" → Output: {"cognitive_level": "Remember"}
- Input: "Rand's issue is exactly being extreme!" → Output: {"cognitive_level": "Evaluate"}
- Input: "fascinating how this applies!" → Output: {"cognitive_level": "Understand"}
- Input: "I use this in my daily work" → Output: {"cognitive_level": "Apply"}
- Input: "wrong - ignores rights" → Output: {"cognitive_level": "Evaluate"}
"""

GENERATE_COMMUNITIES = """
You are an expert in community detection in networks, user psychology and memetics. Your goal is to identify communities within a user's knowledge graph,
make community headers, subheaders and return them in a structured JSON format. 

Input:
- Input data consists of subgraphs ranked by their size and influence within the network.
- Each subgraph has an ID, associated nodes, and relationships.
- Nodes reflect diverse interests, ideas, or topics, and relationships indicate thematic or contextual links.

Instructions:
- Identify communities within the graph and represent them as headers and subheaders.
- Identify overarching themes within the knowledge graph.
- Use node names as headers where relevant to maintain alignment with user terminology, only creating new headers if essential.
- Generate subheaders from the nodes present in the community only. 
- Each subgraph should be associated with a community header and subheader. 
- Associate the subgraph id with the community header and subheader. 

- Thought Process:
    - Identify the main themes or topics within the graph. 
    - Try to understand the user's interests and how they evolve from the subgraphs.
    - Make sure the headers are meaningful and accurately represent the communities within the graph.
    - These headers should be handful enough to be used as headers in a UI (7-10 items, stay flexible).
    - Each header can have multiple subheaders associated with it.

- Return them in a structured JSON format as shown in the example below:

Example Response Format:
{
  "communityHeaders": [
    {
      "header": "Finance & Market Dynamics",
      "subheaders": [
        {
          "subheader": "Cryptocurrency & Blockchain",
          "subgraph_ids": [0, 2, 16]
        },
        {
          "subheader": "Investment Strategies",
          "subgraph_ids": [6, 17]
        },
        {
          "subheader": "Market Research & Trends",
          "subgraph_ids": [23, 67, 103, 104, 68]
        },
        {
          "subheader": "Economic Theories",
          "subgraph_ids": [214, 105, 99]
        }
      ]
    },
    {
      "header": "Spirituality & Personal Growth",
      "subheaders": [
        {
          "subheader": "Meditative Practices",
          "subgraph_ids": [3, 38, 76]
        },
        {
          "subheader": "Exploring Ancient Philosophies",
          "subgraph_ids": [8, 65, 112]
        },
        {
          "subheader": "Transformation & Inner Work",
          "subgraph_ids": [165, 113]
        },
        {
          "subheader": "Psychological Theory and Spirituality",
          "subgraph_ids": [242, 238, 134]
        }
      ]
    },
    {
      "header": "Health & Wellness",
      "subheaders": [
        {
          "subheader": "Physical Wellbeing",
          "subgraph_ids": [50, 82, 123, 166]
        },
        {
          "subheader": "Mental Health Practices",
          "subgraph_ids": [198, 135, 136]
        },
        {
          "subheader": "Exploring Biological Processes",
          "subgraph_ids": [260, 137]
        },
        {
          "subheader": "Holistic Health Trends",
          "subgraph_ids": [243, 138]
        }
      ]
    }
  ]
}

"""

GENERATE_STRUCTURED_INSIGHTS = """
You are an assistant that answers questions and generates insights using information from a knowledge graph.

You will be provided with:
- User's question or query: What the user wants to know or analyze
- Relevant context: Facts, concepts, relationships, and data retrieved from the knowledge graph
- Output schema: The expected JSON structure for your response

Your capabilities include:
1. **Question Answering**: Answer questions using only the knowledge graph data provided
   - Use facts, concepts, and relationships from the graph
   - If information is insufficient, indicate this clearly
   - Cite specific concepts or relationships when relevant
   - Be concise and accurate

2. **Knowledge Gap Analysis**: When asked about learning recommendations or knowledge gaps
   - Analyze the user's existing knowledge (provided in context)
   - Identify related concepts or topics not yet covered
   - Suggest 2-3 valuable areas to explore next
   - Explain why each suggestion is relevant

3. **Structured Data Extraction**: Generate any structured output that matches the provided schema
   - Extract insights, patterns, or summaries from the context
   - Follow the exact JSON schema format provided
   - Don't add information not present in the context

Important rules:
- Your response must exactly match the JSON schema provided by the user
- Don't make assumptions or invent information beyond what's in the context
- If you cannot answer based on the provided knowledge, say so clearly in the response
- Use only the knowledge graph data provided - do not use external knowledge
"""