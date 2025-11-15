"""
Analyze Concepts to identify extractable Terms.

This script:
1. Fetches all Concept nodes
2. Identifies potential Terms within each Concept
3. Shows examples and recommendations
"""
import asyncio
import re
from persona.core.neo4j_database import Neo4jConnectionManager

# Common proper nouns and ideologies that should be Terms
KNOWN_TERMS = [
    # Ideologies
    "libertarianism", "communism", "socialism", "capitalism", "objectivism",
    "collectivism", "individualism", "altruism", "egoism", "liberalism",
    "conservatism", "anarchism", "marxism", "fascism", "democracy",

    # Economic concepts
    "free market", "market economy", "central planning", "austrian school",
    "keynesian", "monetarism", "supply and demand", "price mechanism",

    # Philosophical concepts
    "rationalism", "empiricism", "existentialism", "utilitarianism",
    "deontology", "virtue ethics", "moral relativism", "determinism",

    # People (should already be Person nodes, but might appear in concepts)
    "ayn rand", "friedrich hayek", "milton friedman", "adam smith",
    "karl marx", "john keynes", "ludwig von mises",

    # Places
    "russia", "soviet union", "united states", "america", "europe",
    "new york", "bulgaria", "louvre", "moscow", "washington",
]

def extract_potential_terms(concept_text: str) -> list[str]:
    """
    Extract potential Terms from a Concept text.
    Returns list of (term, reason) tuples.
    """
    potential_terms = []
    concept_lower = concept_text.lower()

    # Check for known terms
    for term in KNOWN_TERMS:
        if term in concept_lower:
            # Find the actual case in the original text
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            match = pattern.search(concept_text)
            if match:
                actual_term = match.group()
                potential_terms.append((actual_term.title(), f"Known ideology/concept: {term}"))

    # Look for capitalized phrases (2-3 words)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', concept_text)
    for phrase in capitalized_phrases:
        if phrase.lower() not in [t[0].lower() for t in potential_terms]:
            # Filter out common words
            if phrase.lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but']:
                potential_terms.append((phrase, "Capitalized phrase (possible proper noun)"))

    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', concept_text)
    for term in quoted:
        if term not in [t[0] for t in potential_terms]:
            potential_terms.append((term, "Quoted term"))

    return potential_terms


async def analyze_concepts():
    # Initialize Neo4j manager
    neo4j_manager = Neo4jConnectionManager()
    await neo4j_manager.initialize()

    user_id = "18"

    print(f"Analyzing Concepts for extractable Terms (user {user_id})...")
    print("=" * 80)

    async with neo4j_manager._ensure_driver().session() as session:
        # Get all Concepts
        concepts_query = """
        MATCH (n:Concept)
        WHERE n.UserId = $user_id
        RETURN n.name as concept, n.discipline as discipline
        ORDER BY n.name
        """

        result = await session.run(concepts_query, user_id=user_id)
        records = await result.values()

        if not records:
            print("No Concept nodes found!")
            await neo4j_manager.close()
            return

        print(f"\nFound {len(records)} Concept nodes")
        print("=" * 80)

        total_potential_terms = 0
        all_terms = []

        for idx, record in enumerate(records, 1):
            concept = record[0]
            discipline = record[1] if record[1] else "N/A"

            potential_terms = extract_potential_terms(concept)
            total_potential_terms += len(potential_terms)

            print(f"\n{idx}. Concept: {concept[:100]}{'...' if len(concept) > 100 else ''}")
            print(f"   Discipline: {discipline}")

            if potential_terms:
                print(f"   Potential Terms ({len(potential_terms)}):")
                for term, reason in potential_terms:
                    print(f"     • {term} ({reason})")
                    all_terms.append(term)
            else:
                print("   No obvious Terms found")

        # Check current Term nodes
        print("\n" + "=" * 80)
        print("Current Term nodes in database:")
        print("-" * 80)

        terms_query = """
        MATCH (n:Term)
        WHERE n.UserId = $user_id
        RETURN n.name as name
        ORDER BY n.name
        """

        result = await session.run(terms_query, user_id=user_id)
        current_terms = [r[0] for r in await result.values()]

        if current_terms:
            for term in current_terms:
                print(f"  • {term}")
        else:
            print("  (No Term nodes found)")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Concepts analyzed: {len(records)}")
        print(f"Current Term nodes: {len(current_terms)}")
        print(f"Potential Terms that could be extracted: {total_potential_terms}")
        print(f"Unique potential Terms: {len(set(all_terms))}")

        # Find missing terms
        missing_terms = [t for t in set(all_terms) if t.lower() not in [ct.lower() for ct in current_terms]]
        if missing_terms:
            print(f"\nMissing Terms (found in Concepts but not as Term nodes):")
            for term in sorted(missing_terms)[:20]:  # Show first 20
                print(f"  • {term}")

    await neo4j_manager.close()
    print("\n" + "=" * 80)
    print("Analysis complete!")

if __name__ == "__main__":
    asyncio.run(analyze_concepts())
