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
IMPORTANT: You must respond with valid JSON format only.

⚠️ CRITICAL: chunk_ids VALIDATION RULE ⚠️
Before including ANY chunk_ids field, you MUST verify each UUID is in valid format:
- VALID: ["67a60841-5373-48c6-83f8-83b6ec784372"] (array of UUIDs in 8-4-4-4-12 hexadecimal with hyphens)
- INVALID: ["80075"], ["80078"], ["12345"], ["abc"] (plain numbers or text)
If the provided chunk_ids contain NON-UUID values → OMIT the chunk_ids field entirely from that node.
Only valid UUIDs with the pattern "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" are accepted.

NOTE: Output chunk_ids as an array of UUID strings. Multiple sources can be tracked per node.
If a node is extracted from a specific book section, include the chunk UUID in the array.

These nodes should represent TRANSFERABLE KNOWLEDGE - concepts, principles, patterns, and insights that can:
- Connect across different sources and contexts
- Apply to the user's life beyond just one book or conversation
- Form a reusable knowledge base that grows over time
- Capture universal patterns, not just specific narrative moments

The nodes will be indexed in a knowledge graph and vector database hybrid system that represents the user's evolving understanding of the world.

Principles for Node Extraction:

⚠️ CRITICAL: Node Names Must Be Nouns
- Node names should be nouns or noun phrases that represent entities or concepts
- If you see relationship words ('versus', 'and', 'between', 'from', 'with'), ask yourself:
  * Is this a UNIVERSAL CONCEPT, or archetype, or title, or phrase? (e.g., "Good vs Evil", "Mother and Child", "Hero's Journey", "Kramer vs Kramer", "Johnson vs USA", "Polly wants a cracker") → Single node
  * Is this comparing SPECIFIC INSTANCES? (e.g., "Libertarianism versus Classical liberalism", "Apple versus Samsung", "Harry likes apples") → Split into separate nodes with relationship
- Examples:
  * ✓ CORRECT: "Libertarianism" (node), "Classical liberalism" (node), relationship: CONTRASTS_WITH
  * ✗ WRONG: "Libertarianism versus Classical liberalism" (single node with verb)
  * ✓ CORRECT: "Good versus Evil" (single node - universal philosophical concept)
  * ✓ CORRECT: "David versus Goliath" (single node - archetypal narrative pattern)

⚠️ CRITICAL: Conservative Node Creation
- Aim for 3-5 well-connected, reusable nodes rather than 10 disconnected ones
- Create nodes that can connect across different contexts and sources
- Prefer fewer rich nodes with multiple relationships over many isolated nodes
- Quality over quantity: Each node should be meaningful and reusable

INCLUDE exactly these fields per node:
- name: Concise, generalizable concept (3-8 words) that represents transferable knowledge, not narrative specifics.
- type: One of: Identity · Memory · Preference · Trait · Narrative · Goal · Event · State · Relationship · Belief · Other types shared below.
- chunk_ids: OPTIONAL - ONLY include if EXPLICITLY provided in the input content in VALID UUID format array.
  * CRITICAL: chunk_ids MUST be an array of valid UUIDs in the format ["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"] where x is a hexadecimal digit (0-9, a-f)
  * VALID UUID example: ["550e8400-e29b-41d4-a716-446655440000"] (array with 8-4-4-4-12 hexadecimal characters separated by hyphens)
  * INVALID examples: ["80075"] (plain number), ["12345"] (not UUID format), ["abc-def"] (not proper UUID structure)
  * CRITICAL: Copy the EXACT UUID strings from the input - do not modify, truncate, or reformat them
  * CRITICAL: If you see "chunk_ids: [VALUES]" in the input, verify each value matches UUID format (8-4-4-4-12 pattern) before including it
  * If ANY provided value is NOT a valid UUID format, OMIT the chunk_ids field entirely
  * DO NOT infer, generate, or assume chunk_ids values
  * Used to link concepts back to specific book content sections in an external system
  * For notes, chat messages, or writing projects: OMIT this field entirely
  * NEVER use page numbers, chapter numbers, or plain numbers as chunk_ids
- discipline: REQUIRED field indicating the academic/knowledge domain this node belongs to. Examples:
  * Academic domains: "Psychology", "Computer Science", "History", "Biology", "Philosophy", "Economics", "Physics", "Literature"
  * Life domains: "Career", "Health", "Relationships", "Hobbies", "Finance", "Education", "Personal Development"
  * If unclear or general, use "General" or the most appropriate broad category
- bloom_level: REQUIRED cognitive level based on Bloom's taxonomy - SET CONSERVATIVELY using your knowledge to VALIDATE accuracy:

  ASSESSMENT RULES (Use your knowledge to verify correctness):

  * "Remember" - DEFAULT. Use when:
    - User is passively reading/highlighting
    - User mentions concept but shows no understanding
    - User's explanation is INCORRECT or CONFUSED (even if they wrote something)
    - No evidence of cognitive work beyond recognition

  * "Understand" - Use ONLY if:
    - User paraphrases or explains in their own words, AND
    - The explanation is FACTUALLY CORRECT (validate against your knowledge)
    - User compares/contrasts accurately with other concepts
    - Shows comprehension, not just repetition
    - If explanation has significant ERRORS → downgrade to "Remember"

  * "Apply" - Use ONLY if:
    - User describes using knowledge to solve a problem, AND
    - The application is CORRECT and appropriate
    - User demonstrates proper use in a new context
    - If misapplied or incorrect usage → max "Understand" or "Remember"

  * "Analyze" - Use ONLY if:
    - User breaks down concept into components, AND
    - The analysis is LOGICALLY SOUND and accurate
    - User identifies patterns or distinguishes parts correctly
    - If analysis is flawed or confused → downgrade accordingly

  * "Evaluate" - Use ONLY if:
    - User makes informed judgments with VALID criteria
    - Critique is reasoned and demonstrates deep understanding
    - Not just opinions - must show evaluative thinking
    - If judgment is unfounded or illogical → downgrade

  * "Create" - Use ONLY if:
    - User synthesizes to produce genuinely NEW insights
    - Creation is coherent and demonstrates mastery
    - Not just recombination - must show innovation

  CRITICAL VALIDATION STEP:
  Before assigning Understand or higher, ask yourself:
  1. "Is what the user wrote/said CORRECT according to my knowledge?"
  2. "Does this demonstrate actual cognitive work, or just exposure?"
  3. "If this were on an exam, would it receive credit?"

  If the answer to #1 is NO → assign "Remember" (encountered but misunderstood)
  If the answer to #2 is "just exposure" → assign "Remember"
  If the answer to #3 is NO → assign maximum "Remember" or "Understand" (partial credit)

  IMPORTANT: Most nodes from passive reading should be "Remember". Higher levels require DEMONSTRATED and CORRECT cognitive work.
- confidence: REQUIRED extraction quality score (0.0 to 1.0):
  * 1.0 = Explicit, direct statement with complete clarity
  * 0.8-0.9 = Clear implication with strong supporting context
  * 0.6-0.7 = Reasonable inference from available information
  * 0.4-0.5 = Weak signal or ambiguous data
  * Below 0.4 = Too speculative, avoid creating node
- book_id: OPTIONAL - ONLY include if the node is extracted from a specific book. Use the book's unique id.  
- highlight_id: OPTIONAL - ONLY include if the node is linked to a specific user highlight. Use the highlight's unique id.  
- writing_id: OPTIONAL - ONLY include if the node is linked to a specific user writing project. Use the writing project's unique id.

What to Extract - CREATE SEPARATE NODES FOR EACH ENTITY:

⚠️ CONSERVATIVE EXTRACTION: Focus on quality over quantity
- Extract 3-5 core, reusable concepts rather than exhaustively listing every detail
- Each node should be meaningful enough to connect to other sources and contexts
- Prefer creating nodes that will have multiple relationships over isolated facts

For each entity or concept in the text, create an individual node with:
- Name: Concise identifier for the entity (MUST be noun/noun phrase, not contain verbs like 'versus', 'between', 'and' unless it's a universal archetype)
- Type: Category (see types below)
- A brief description or definition in context (stored in properties)

FOR BOOK/ARTICLE CONTENT - Extract ALL entities as individual nodes:

A. **Source Metadata Nodes** (if content is from a book/article):
   - Book node: "The Count of Monte Cristo" (type: "Book", discipline: "Literature")
   - Author node: "Alexandre Dumas" (type: "Author", discipline: "Literature")
   - Genre nodes: "Adventure", "Historical Fiction" (type: "Genre", discipline: "Literature")
   - Publication year can be stored in properties: {"publication_year": "1844"}

B. **Character Nodes** (one node per significant character):
   - Full name as node: "Edmond Dantès" (type: "Character", discipline: "Literature")
   - Full name as node: "Danglars" (type: "Character", discipline: "Literature")
   - Store traits in properties: {"role": "protagonist", "traits": "naive turned vengeful"}
   - Relationships between characters will be created via GET_RELATIONSHIPS

C. **Location Nodes** (if significant):
   - "Marseilles", "Château d'If", "Paris" (type: "Location", discipline: "Geography")

D. **Transferable Concept Nodes**:
   - "Betrayal by trusted colleagues" (type: "Concept", discipline: "Psychology")
   - "Professional jealousy" (type: "Concept", discipline: "Psychology")
   - "Justice versus revenge" (type: "Concept", discipline: "Philosophy")
   - "Hope sustains through suffering" (type: "Insight", discipline: "Philosophy")
   - "Isolation transforms personality" (type: "Pattern", discipline: "Psychology")
   - "Power corrupts" (type: "Principle", discipline: "Philosophy")

E. **Symbol/Theme/Archetype Nodes**:
   - "The wronged innocent" (type: "Archetype", discipline: "Literature")
   - "The mentor figure" (type: "Archetype", discipline: "Literature")
   - "Transformation through suffering" (type: "Theme", discipline: "Literature")

F. **Event Nodes** (if culturally/historically significant):
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

Guidelines for Node Creation:
   - **For books/content**: Prioritize TRANSFERABLE CONCEPTS over plot details. Include major characters/events only if culturally significant.
   - **Avoid verb-based node names**: If comparing/contrasting concepts, create separate nodes with relationships UNLESS it's a universal archetype (e.g., "David versus Goliath" = archetype, but "Hayek versus Keynes" = two economist nodes with CONTRASTS_WITH relationship).
   - **chunk_ids usage - STRICT VALIDATION REQUIRED**:
     * ONLY include chunk_ids if you see VALID UUIDs in the input (format: ["xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"])
     * Before including chunk_ids, verify EACH value has exactly 5 groups of hexadecimal characters: 8-4-4-4-12 digits separated by hyphens
     * Valid: ["67a60841-5373-48c6-83f8-83b6ec784372"], ["550e8400-e29b-41d4-a716-446655440000"]
     * Invalid: ["80075"], ["80078"], ["12345"], ["abc"], ["chunk_123"] - REJECT these and OMIT chunk_ids field
     * If the input provides ANY non-UUID value (like a plain number), DO NOT include chunk_ids at all
     * Copy the exact UUID strings character-by-character from the input without any modifications
   - **For user data**: Extract personal specifics that define the individual. NEVER include chunk_ids for personal data.
   - **For notes/chat/writing projects**: NEVER include chunk_ids - these are not tied to book chunks.
   - Ask: "Can this node connect to knowledge from other sources?" If yes, it's well-abstracted.
   - Characters/events qualify if they're references people use in conversation ("That's so Gatsby" or "Orwellian surveillance")
   - Balance concrete (names, events) with abstract (concepts, patterns) for a rich knowledge graph

Avoid:
   - Moment-by-moment scene descriptions ("Dantès talks to the shipowner about delays")
   - Minor side characters who don't transcend their story ("Caderousse's neighbor")
   - Plot mechanics without deeper meaning ("Character X goes to location Y")
   - Overly specific phrases that can't generalize ("Questioning shipowner's honesty about Elba delays")

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
      "bloom_level": "Remember",
      "confidence": 1.0,
      "properties": {"publication_year": "1844"}
    },
    {
      "name": "Alexandre Dumas",
      "type": "Author",
      "discipline": "Literature",
      "bloom_level": "Remember",
      "confidence": 1.0
    },
    {
      "name": "Adventure",
      "type": "Genre",
      "discipline": "Literature",
      "bloom_level": "Remember",
      "confidence": 1.0
    },
    {
      "name": "Historical Fiction",
      "type": "Genre",
      "discipline": "Literature",
      "bloom_level": "Remember",
      "confidence": 1.0
    },
    {
      "name": "Edmond Dantès",
      "type": "Character",
      "discipline": "Literature",
      "bloom_level": "Remember",
      "confidence": 1.0,
      "properties": {"role": "protagonist", "traits": "naive turned vengeful"}
    },
    {
      "name": "Danglars",
      "type": "Character",
      "discipline": "Literature",
      "bloom_level": "Remember",
      "confidence": 1.0,
      "properties": {"role": "antagonist", "traits": "envious, greedy"}
    },
    {
      "name": "Marseilles",
      "type": "Location",
      "discipline": "Geography",
      "bloom_level": "Remember",
      "confidence": 1.0
    },
    {
      "name": "Betrayal by trusted colleagues",
      "type": "Concept",
      "chunk_ids": ["550e8400-e29b-41d4-a716-446655440000"],
      "book_id": [27],
      "highlight_id": [],
      "writing_id": [],
      "discipline": "Psychology",
      "bloom_level": "Understand",
      "confidence": 0.95
    },
    {
      "name": "Justice versus revenge",
      "type": "Concept",
      "chunk_ids": ["550e8400-e29b-41d4-a716-446655440000"],
      "book_id": [27],
      "highlight_id": [456, 789],
      "writing_id": [],
      "discipline": "Philosophy",
      "bloom_level": "Evaluate",
      "confidence": 0.9
    },
    {
      "name": "Isolation transforms personality",
      "type": "Pattern",
      "book_id": [27],
      "highlight_id": [],
      "writing_id": [],
      "discipline": "Psychology",
      "bloom_level": "Analyze",
      "confidence": 0.85
    }
  ]
}

IMPORTANT NOTES:
1. Relationships will be created separately (e.g., "The Count of Monte Cristo" WRITTEN_BY "Alexandre Dumas", "Edmond Dantès" APPEARS_IN "The Count of Monte Cristo")
2. chunk_ids in examples above (["550e8400-e29b-41d4-a716-446655440000"]) is a VALID UUID array - notice the 8-4-4-4-12 hexadecimal pattern with hyphens
3. If you receive chunk_ids values like ["80075"], ["12345"], or any plain numbers, these are INVALID - DO NOT include chunk_ids field for those nodes
4. Always verify ALL values in chunk_ids array match UUID pattern before including it
5. Entity ID fields (book_id, highlight_id, writing_id):
   - These are OPTIONAL integer arrays that link nodes to source entities
   - book_id: Array of book IDs the node is derived from (e.g., [27, 45])
   - highlight_id: Array of highlight IDs the node is derived from (e.g., [123, 456])
   - writing_id: Array of writing IDs the node is derived from (e.g., [789])
   - Empty arrays [] are acceptable when no entity IDs are available
   - A node can have multiple entity IDs if derived from multiple sources
   - These help trace nodes back to their original source entities

Example Response Format for USER DATA (NOTE: No chunk_ids for personal data):
{
  "nodes": [
    {
      "name": "Born in 1990 in Seattle",
      "type": "Identity",
      "discipline": "Personal History",
      "bloom_level": "Remember",
      "confidence": 1.0
    },
    {
      "name": "Prefers working in solitude before dawn",
      "type": "Preference",
      "discipline": "Work Habits",
      "bloom_level": "Understand",
      "confidence": 0.9
    },
    {
      "name": "Technology should serve human connection",
      "type": "Belief",
      "discipline": "Philosophy",
      "bloom_level": "Evaluate",
      "confidence": 0.95
    },
    {
      "name": "Training for marathon next spring",
      "type": "Goal",
      "discipline": "Health",
      "bloom_level": "Apply",
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
      - RELATED_TO: General semantic connection
      - EXTENDS: One concept extends or builds upon another
      - SPECIALIZES: More specific instance of a general concept
   
   B. Hierarchical Relationships:
      - PARENT_OF / CHILD_OF: Hierarchical or categorical relationship
      - PART_OF / CONTAINS: Composition relationships
      - SUBTOPIC_OF: Knowledge hierarchy
   
   C. Argumentative Relationships:
      - SUPPORTS: One concept provides evidence/support for another
      - OPPOSES / ARGUES_AGAINST: One concept contradicts or opposes another
      - EVIDENCES: Provides evidence for a claim
      - REFUTES: Disproves or contradicts
   
   D. Causal Relationships:
      - LEADS_TO / CAUSES: Direct causation
      - RESULTS_IN: Outcome or consequence
      - ENABLES: Makes something possible
      - PREVENTS: Stops or blocks something
   
   E. Temporal Relationships:
      - PRECEDES / FOLLOWS: Time-based sequence
      - HAPPENS_BEFORE / HAPPENS_AFTER: Event ordering
   
   F. Influence & Impact:
      - SHAPES / INFLUENCES: One affects the other
      - INSPIRES: Motivational or creative influence
      - MOTIVATES: Drives action or decision
      - ENHANCES: Improves or amplifies
      - WEAKENS: Diminishes or reduces
   
   G. Cognitive & Personal:
      - RESONATES_WITH: Emotional or intellectual alignment
      - CONFLICTS_WITH: Internal tension or contradiction
      - EVOLVES_INTO / TRANSFORMS_TO: Personal growth or change
      - APPLIES_TO: Practical application context
   
   H. Learning & Knowledge (PKG-specific):
      - PREREQUISITE_OF / BUILDS_ON: One concept must be understood before another
      - EXEMPLIFIES / INSTANTIATES: Concrete example of an abstract concept
      - DEFINES / CLARIFIES: One concept defines or explains another
      - COMPARES_TO / CONTRASTS_WITH: Comparative relationships for learning
      - QUESTIONS / CHALLENGES: One concept raises questions about another
      - ANSWERS / RESOLVES: One concept provides answers to questions in another
      - SYNTHESIZES: Combines multiple concepts into new understanding
      - ANNOTATES / COMMENTS_ON: Commentary or reflection on a concept
      - LEARNED_FROM: Knowledge source relationship
      - REINFORCES: Strengthens or supports existing knowledge

2. Principles for Relationship Creation:
   - Only create relationships that are strongly justified
   - Focus on relationships that reveal meaningful patterns
   - Prefer direct connections over tenuous ones
   - Consider temporal and causal flows
   - Look for relationships that help understand the user's journey

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