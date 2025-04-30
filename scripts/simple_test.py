#!/usr/bin/env python
"""
Simple Memory System Test

A minimalist test that clearly shows if the memory system is working correctly.
"""


class Memory:
    """Simplified memory system."""

    def __init__(self):
        self.data = []
        self.count = 0

    def add(self, content, is_correction=False):
        """Add content to memory."""
        self.count += 1
        self.data.append(
            {"id": f"unit_{self.count}", "content": content, "is_correction": is_correction}
        )
        return self.count

    def search(self, query):
        """Basic search for content."""
        results = []
        for item in self.data:
            # Simple word matching
            query_words = query.lower().split()
            content_words = item["content"].lower().split()

            # Count matching words
            matches = sum(1 for word in query_words if word in content_words)
            if matches > 0:
                score = matches / len(query_words)

                # Boost corrections
                if item["is_correction"]:
                    score *= 1.5

                results.append((item, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def test_memory():
    """Run simple test."""
    print("MEMORY SYSTEM TEST")
    print("-----------------")

    memory = Memory()

    # Test 1: Adding content
    print("\nTest 1: Adding content")
    memory.add("Python is a programming language")
    memory.add("JavaScript is used for web development")
    print(f"Added 2 items, memory has {len(memory.data)} items")
    print("Test 1 result:", "PASS" if len(memory.data) == 2 else "FAIL")

    # Test 2: Basic retrieval
    print("\nTest 2: Basic retrieval")
    results = memory.search("What is Python?")
    print("Query: 'What is Python?'")
    if results:
        print(f"Top result: '{results[0][0]['content']}' (Score: {results[0][1]:.2f})")
        contains_python = "python" in results[0][0]["content"].lower()
        print("Test 2 result:", "PASS" if contains_python else "FAIL")
    else:
        print("No results found")
        print("Test 2 result: FAIL")

    # Test 3: Correction prioritization
    print("\nTest 3: Correction prioritization")
    memory.add("JavaScript was created in 2005")
    memory.add("JavaScript was created in 1995 by Brendan Eich", True)  # Correction

    results = memory.search("When was JavaScript created?")
    print("Query: 'When was JavaScript created?'")
    if results:
        print(f"Top result: '{results[0][0]['content']}' (Score: {results[0][1]:.2f})")
        is_correction = results[0][0]["is_correction"]
        contains_1995 = "1995" in results[0][0]["content"]
        print("Top result is correction:", "YES" if is_correction else "NO")
        print("Test 3 result:", "PASS" if is_correction and contains_1995 else "FAIL")
    else:
        print("No results found")
        print("Test 3 result: FAIL")

    # Summary
    print("\nTEST SUMMARY")
    print("-----------")
    passed = (
        len(memory.data) == 4
        and results
        and results[0][0]["is_correction"]
        and "1995" in results[0][0]["content"]
    )
    print("All tests:", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    test_memory()
