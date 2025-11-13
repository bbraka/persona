"""
Test script for cognitive level assessment.

This tests the new assess_cognitive_level() function with various UserNote examples.
"""

import asyncio
from persona.llm.llm_graph import assess_cognitive_level


async def test_cognitive_assessments():
    """Test various UserNote texts and their expected cognitive levels."""

    test_cases = [
        # (user_note_text, expected_level_range)
        ("interesting", ["Remember"]),
        ("Rand's issue is exactly being extreme!", ["Evaluate", "Analyze"]),
        ("fascinating how this applies!", ["Understand"]),
        ("I use this in my daily work", ["Apply"]),
        ("wrong - ignores rights", ["Evaluate"]),
        ("noted", ["Remember"]),
        ("reminds me of Hayek's work", ["Understand"]),
        ("this is exactly right", ["Apply"]),
        ("the flaw is being too absolutist", ["Evaluate", "Analyze"]),
        ("combining X and Y creates a new framework", ["Create"]),
    ]

    print("=" * 80)
    print("COGNITIVE LEVEL ASSESSMENT TEST")
    print("=" * 80)

    passed = 0
    failed = 0

    for note_text, expected_levels in test_cases:
        try:
            assessed_level = await assess_cognitive_level(note_text)

            if assessed_level in expected_levels:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1

            print(f"\n{status}")
            print(f"  Note: '{note_text}'")
            print(f"  Expected: {' or '.join(expected_levels)}")
            print(f"  Got: {assessed_level}")

        except Exception as e:
            print(f"\n✗ ERROR")
            print(f"  Note: '{note_text}'")
            print(f"  Error: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_cognitive_assessments())
