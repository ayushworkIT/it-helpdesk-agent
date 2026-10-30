"""
Step 2.6 - Test retrieval on its own (no LLM): does each question find the right chunk?
Score = distance: LOWER means MORE similar.
"""
from ingest import get_vector_store

# (question, document we EXPECT to come top) - wording deliberately differs from the docs
TEST_QUESTIONS = [
    ("I'm working from home and can't reach the shared drives", "vpn_guide.md"),
    ("I typed my password wrong too many times and now I'm stuck", "password_policy.md"),
    ("Who do I contact when the printer runs out of toner?", "printer_troubleshooting.md"),
    ("I need Photoshop for a design project", "software_requests.md"),
    ("My visitor needs internet access", "wifi_and_network.md"),
    ("The whole office lost the network at 9pm on a Saturday", "it_support_contacts.md"),
    ("What's a good recipe for pizza dough?", None),   # off-topic trap: nothing should match well
]

store = get_vector_store()

for question, expected in TEST_QUESTIONS:
    results = store.similarity_search_with_score(question, k=3)
    top_source = results[0][0].metadata["source"]

    if expected is None:
        verdict = "(off-topic - check the scores)"
    elif top_source == expected:
        verdict = "PASS"
    else:
        verdict = f"FAIL (expected {expected})"

    print(f"\nQ: {question}")
    print(f"   {verdict}")
    for doc, score in results:
        print(f"   {score:.3f}  {doc.metadata['source']:<28} > {doc.metadata['section']}")