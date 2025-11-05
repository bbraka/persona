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

**2. Quality Over Quantity**
- Aim for 3-5 well-connected, reusable nodes rather than 10 disconnected ones
- Each node should connect across contexts and be meaningful for the knowledge graph
- Prioritize concepts that will have multiple relationships

## Required Fields Per Node

- **name**: Concise, generalizable concept (3-8 words) representing transferable knowledge
- **type**: One of: Identity · Memory · Preference · Trait · Narrative · Goal · Event · State · Relationship · Belief · Term · Other types shared below
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
- **confidence**: REQUIRED - Extraction quality score (0.0 to 1.0)
  * 1.0 = Explicit, direct statement with complete clarity
  * 0.8-0.9 = Clear implication with strong context
  * 0.6-0.7 = Reasonable inference
  * 0.4-0.5 = Weak signal or ambiguous
  * Below 0.4 = Too speculative, avoid creating node
- **source_index**: REQUIRED for batch, OMIT for single-source
  * When input has "Source [0]:", "Source [1]:" format, include this field
  * Can be integer (0) or array for cross-source concepts ([0, 2])
  * Valid range: 0 to N-1 where N is number of sources
- **book_id**, **highlight_id**, **writing_id**: OPTIONAL arrays (e.g., [27])
  * ONLY include if explicitly provided in source metadata
  * For batch processing, source_index handles mapping - don't manually extract these

## What to Extract

### PRIMARY PATTERN: 4-Node Extraction for Highlights/Notes

When ingesting a highlight with a user note, you MUST create exactly 4 interconnected nodes:

1. **Highlight Node** (type: "Highlight")
   - Name: The highlighted text itself or a concise representation
   - Contains the actual highlighted content from the book/article

2. **UserNote Node** (type: "UserNote")
   - Name: The user's comment/annotation on the highlight
   - Contains the user's personal reflection, thought, or note

3. **Concept Node** (type: "Concept")
   - Name: A synthesized FULL SENTENCE combining highlight + note + surrounding context
   - This is the KEY NODE - a complete, standalone statement that captures the knowledge
   - Example: Highlight "Frank Knight, Henry Simons" + Note "Milton Friedman's advisors" → Concept "Frank Knight and Henry Simons were Milton Friedman's advisors"
   - Must be grammatically complete and make sense on its own
   - **IMPORTANT**: Concepts are ALWAYS full sentences or definitions, NOT single words or short phrases
   - Single words/phrases should be extracted as "Term" nodes (see below)

4. **CognitiveLevel Node** (type: "CognitiveLevel")
   - Name: MUST be one of these exact enum values: "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
   - This represents the user's cognitive understanding level for the Concept node
   - NOTE: Higher levels mean demonstrated ability, NOT necessarily correctness
   - **IMPORTANT**: Each Concept MUST have its own dedicated CognitiveLevel node
   - NEVER reuse or share CognitiveLevel nodes between different Concepts

### Cognitive Level Assessment Guide

**Base your assessment on the Concept node content** - what depth of understanding does the extracted Concept demonstrate?
Look at how the user engaged with the material in their highlight + note combination.

* "Remember" - DEFAULT for passive reading:
  - Concept is just factual recall or recognition (e.g., "X is Y")
  - Simple definition or statement without elaboration
  - User passively highlighted without adding interpretation
  - Example: "Milton Friedman was an economist"

* "Understand" - Concept shows comprehension:
  - Concept includes explanation, paraphrasing, or comparison
  - User demonstrates understanding in their own words
  - Concept shows meaning-making beyond raw facts
  - Example: "Milton Friedman believed free markets self-regulate better than government intervention"

* "Apply" - Concept shows practical application:
  - Concept describes using knowledge in a specific context
  - User demonstrates application to solve a problem
  - Shows transfer to new situations or practical usage
  - Example: "I used Friedman's monetary theory to analyze the 2008 financial crisis"

* "Analyze" - Concept involves analytical thinking:
  - Concept breaks down components, examines patterns
  - User distinguishes parts and relationships
  - Shows decomposition or systematic examination
  - Example: "Friedman's theory has three core assumptions: rational actors, perfect information, and no externalities"

* "Evaluate" - Concept includes judgment:
  - Concept critiques, assesses, or judges ideas
  - User makes informed judgments with criteria
  - Shows critical evaluation or weighing of merits
  - Example: "While Friedman's free market theory works in stable economies, it fails to account for systemic crises"

* "Create" - Concept synthesizes new ideas:
  - Concept combines ideas to produce new insights
  - User creates original connections or interpretations
  - Shows creative synthesis or novel perspective
  - Example: "By combining Friedman's monetary theory with Keynesian fiscal policy, we can create a hybrid crisis response framework"

**Assessment Guidelines**:
1. Assess based on what the user DEMONSTRATES in the Concept, not on correctness
2. Look at the COMBINED information (Highlight + UserNote) that forms the Concept
3. Higher cognitive levels require explicit evidence - don't over-estimate
4. When in doubt, default to a lower level (Remember or Understand)

### Examples

**EXAMPLE 1 - Remember Level:**
Input: Highlight "Frank Knight, Henry Simons" + Note "Milton Friedman's advisors"

Cognitive Assessment:
- Concept formed: "Frank Knight and Henry Simons were Milton Friedman's advisors"
- This is simple factual recall - the user is just noting who the advisors were
- No explanation, analysis, or application demonstrated
- **CognitiveLevel: "Remember"**

Output nodes:
1. {"name": "Frank Knight, Henry Simons", "type": "Highlight", "properties": {"discipline": "Economics"}, ...}
2. {"name": "Milton Friedman's advisors", "type": "UserNote", "properties": {"discipline": "Economics"}, ...}
3. {"name": "Frank Knight and Henry Simons were Milton Friedman's advisors", "type": "Concept", "properties": {"discipline": "Economics"}, ...}
4. {"name": "Remember", "type": "CognitiveLevel", "properties": {"discipline": "Education"}, ...}

**EXAMPLE 2 - Evaluate Level:**
Input: Highlight "Free market capitalism" + Note "Works best when information is symmetric and transaction costs are low, but fails during crises when these assumptions break down"

Cognitive Assessment:
- Concept formed: "Free market capitalism works best with symmetric information and low transaction costs, but fails when these conditions don't hold"
- User is making a critical judgment about when the theory works vs. fails
- Shows evaluation with specific criteria (information symmetry, transaction costs)
- **CognitiveLevel: "Evaluate"**

Output nodes:
1. {"name": "Free market capitalism", "type": "Highlight", ...}
2. {"name": "Works best when information is symmetric...", "type": "UserNote", ...}
3. {"name": "Free market capitalism works best with symmetric information and low transaction costs, but fails when these conditions don't hold", "type": "Concept", ...}
4. {"name": "Evaluate", "type": "CognitiveLevel", ...}

These 4 nodes will be connected via relationships (see GET_RELATIONSHIPS for details).

### IMPORTANT: Term vs Theme vs Concept Distinction

**Term nodes** (type: "Term"):
- Single words or short phrases (1-3 words) that reference specific things, ideas, or names
- Keywords, jargon, terminology, proper nouns, technical terms
- Examples: "Objectivism", "Free market", "Utilitarianism", "Blockchain", "Neural networks"
- Use Term when: The node is a label, name, or keyword rather than a complete thought
- **Terms do NOT get CognitiveLevel nodes** - they are reference points, not learned concepts

**Theme nodes** (type: "Theme"):
- Medium-length phrases (4-10 words) that represent topics, subjects, or themes but are NOT complete sentences
- Topical concepts from reading progress, chapter themes, or subject matter
- Examples:
  * "Loss of intellectual mentor and self-reinvention"
  * "Transformation through suffering"
  * "Professional jealousy and self-image"
  * "Alliances with conservative figures"
  * "Shift toward Aristotelian reason"
- Use Theme when: The node is a topic or subject phrase but not a grammatically complete statement
- **Themes do NOT get CognitiveLevel nodes** - they are subjects/topics, not learned concepts with depth
- Themes can be RELATED TO Concepts, but are not Concepts themselves

**Concept nodes** (type: "Concept"):
- MUST be complete sentences or definitions (typically 8+ words with subject + verb + object/complement)
- Grammatically complete statements that express understanding, explanations, or principles
- Examples:
  * "Objectivism holds that rational self-interest is the basis of morality"
  * "Free market capitalism relies on supply and demand to set prices"
  * "Neural networks learn by adjusting weights through backpropagation"
  * "Professional jealousy arises when someone's success threatens our self-image"
  * "Loss of a mentor forces individuals to develop independent thinking"
- Use Concept when: The node expresses a complete idea, definition, or understanding as a full sentence
- **Concepts MUST have CognitiveLevel nodes** - they represent learned knowledge with depth

**Quick Test**:
- Can it be a Wikipedia article title? → Term
- Is it a topic/subject phrase but not a complete sentence? → Theme
- Is it a complete sentence from the article? → Concept

**Examples**:
- ❌ WRONG: "Objectivism" (type: "Concept") with CognitiveLevel
- ❌ WRONG: "Loss of intellectual mentor" (type: "Concept") with CognitiveLevel
- ✅ CORRECT: "Objectivism" (type: "Term") - no CognitiveLevel
- ✅ CORRECT: "Loss of intellectual mentor and self-reinvention" (type: "Theme") - no CognitiveLevel
- ✅ CORRECT: "Objectivism is Ayn Rand's philosophy based on rational self-interest" (type: "Concept") with CognitiveLevel
- ✅ CORRECT: "Loss of a mentor forces individuals to develop independent thinking and self-reliance" (type: "Concept") with CognitiveLevel

### SECONDARY PATTERN: Person Node Extraction
When you see ANY reference to a person (author, philosopher, scientist, historical figure):
1. EXTRACT A PERSON NODE using their FULL NAME (e.g., "Ayn Rand", "Albert Einstein", "Karl Marx")
2. Use node type "Person" (NOT Author/Philosopher/Scientist)
3. If only a partial name appears (e.g., "Rand", "Einstein"), infer the full name from context or existing graph
4. Person nodes are MANDATORY - extract even if the person is only mentioned briefly
5. Examples:
   - "Rand's philosophy" → Extract node: "Ayn Rand" (type: Person)
   - "According to Marx" → Extract node: "Karl Marx" (type: Person)
   - "Einstein's theory" → Extract node: "Albert Einstein" (type: Person)

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
      "name": "Marseilles",
      "type": "Location",
      "discipline": "Geography",
      "confidence": 1.0
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
      - ANNOTATED_WITH: Highlight node connected to UserNote node
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

   a) **3-NODE PATTERN - HIGHEST PRIORITY**:
      When you see Highlight, UserNote, and Concept nodes, create these relationships:
      - Highlight ANNOTATED_WITH UserNote
      - Highlight SYNTHESIZED_INTO Concept
      - UserNote SYNTHESIZED_INTO Concept

      Example with nodes:
      - Node1: "Frank Knight, Henry Simons" (type: Highlight)
      - Node2: "Milton Friedman's advisors" (type: UserNote)
      - Node3: "Frank Knight and Henry Simons were Milton Friedman's advisors" (type: Concept)

      Relationships:
      - Node1 ANNOTATED_WITH Node2
      - Node1 SYNTHESIZED_INTO Node3
      - Node2 SYNTHESIZED_INTO Node3

      NOTE: CognitiveLevel relationships (HAS_UNDERSTANDING_LEVEL) are created automatically by the system.
      DO NOT create any relationships involving CognitiveLevel nodes.

   b) Reading Progress → Highlights/Notes:
      - When a ReadingProgress node exists, connect it with READING_AT to the book
      - Connect any concepts from same page/chapter with ENCOUNTERED_IN to ReadingProgress
      - Example: "Page 47 of Atlas Shrugged" READING_AT "Atlas Shrugged"
      - Example: "Objectivism concept" ENCOUNTERED_IN "Page 47 of Atlas Shrugged"

   c) Person Attribution - ALWAYS CREATE THESE PATTERNS:
      - When you see "Rand's philosophy" or possessive forms:
        * Extract Person node: "Ayn Rand"
        * Extract concept node: "Rand's philosophy" or "Objectivism"
        * Relationship: "Rand's philosophy" ATTRIBUTED_TO "Ayn Rand"
        * Relationship: "Objectivism" CREATED_BY "Ayn Rand"
      - When you see "according to Rand" or attribution phrases:
        * Extract Person node: "Ayn Rand"
        * Extract concept node from what they said
        * Relationship: concept ATTRIBUTED_TO "Ayn Rand"
      - CRITICAL: The Person node with FULL NAME must ALWAYS be created when any person is mentioned

   d) Connect to Existing Person Nodes:
      - BEFORE creating a new Person node, check if that person already exists in the graph context
      - If "Ayn Rand" exists in the graph, use that exact name for relationships
      - If you see duplicate person nodes (e.g., "Ayn Rand" as Author and as Philosopher), treat them as the SAME person
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
         - When a Theme represents a topic and a Concept elaborates on that topic as a full sentence
         - Use RELATED_TO, EXPLORES_THEME, EXEMPLIFIES, INSTANTIATES, or ELABORATES_ON
         - **Search actively**: If you see Theme "Alliance with conservative networks", look for Concepts about alliances, conservatism, or networks
         - Example: "Loss of intellectual mentor" (Theme) RELATED_TO "Loss of a mentor forces independent thinking" (Concept)
         - Example: "Economic policy innovation" (Theme) RELATED_TO "Economic collapse leads to policy innovation" (Concept)
         - Example: "Philosophical crusade" (Theme) ELABORATES_ON "Philosophy requires rational self-interest as its foundation" (Concept)
         - Example: "Reorientation toward reason" (Theme) RELATED_TO "Reason is the foundation of human survival and flourishing" (Concept)

      2. Theme → Person connections:
         - When a Theme is associated with a person's work, life, or ideas
         - Use RELATED_TO, ATTRIBUTED_TO, ASSOCIATED_WITH
         - Example: "Self-reinvention after career shift" (Theme) RELATED_TO "Milton Friedman" (Person)
         - Example: "Free market advocacy" (Theme) ATTRIBUTED_TO "Ayn Rand" (Person)

      3. Theme → Event connections:
         - When a Theme is related to a historical or significant event
         - Use RELATED_TO, AROSE_FROM, INFLUENCED_BY
         - Example: "Policy innovation" (Theme) AROSE_FROM "Great Depression" (Event)
         - Example: "Economic nationalism" (Theme) INFLUENCED_BY "World War II" (Event)

      4. Concept → Person connections:
         - When a Concept is created by, advocated by, or attributed to a person
         - Use CREATED_BY, ATTRIBUTED_TO, ADVOCATED_BY, CRITICIZED_BY
         - Example: "Rational self-interest is the basis of morality" (Concept) ATTRIBUTED_TO "Ayn Rand" (Person)

      5. Concept → Event connections:
         - When a Concept relates to or arose from an event
         - Use RELATED_TO, AROSE_FROM, APPLIES_TO, EXPLAINS
         - Example: "Economic collapse leads to policy innovation" (Concept) EXPLAINS "Great Depression" (Event)

      6. Cross-type semantic relationships:
         - Look for ANY meaningful semantic connection between different node types
         - Use RELATED_TO as a general connector when a more specific relationship isn't clear
         - Example: "Chicago School" (Term) RELATED_TO "Milton Friedman" (Person)
         - Example: "University of Chicago" (Location) ASSOCIATED_WITH "Chicago School" (Term)

      **IMPORTANT**: Be LIBERAL with these cross-type connections. If two nodes seem related when reading
      them together, CREATE THE RELATIONSHIP. It's better to have too many semantic connections than too few.
      The goal is a DENSELY CONNECTED knowledge graph where ideas, people, places, themes, and events
      are all interlinked.

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