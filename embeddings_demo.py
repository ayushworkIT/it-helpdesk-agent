"""
Step 2.1 - See embeddings in action: similar meanings -> similar numbers.
"""
import math

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# The embedding model turns text into a list of numbers (a vector)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

sentences = [
    "How do I reset my password?",            # 0
    "I forgot my login credentials",          # 1 - same meaning, different words
    "The printer on floor 2 is out of toner", # 2 - unrelated topic
]

# Turn all three sentences into embeddings in one API call
vectors = embeddings.embed_documents(sentences)

print("Each sentence became a list of", len(vectors[0]), "numbers")
print("First 5 numbers of sentence 0:", [round(x, 3) for x in vectors[0][:5]])


def cosine_similarity(a, b):
    """How close two vectors point in the same direction (1.0 = identical meaning)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


print(f"\npassword vs login credentials: {cosine_similarity(vectors[0], vectors[1]):.2f}")
print(f"password vs printer toner:     {cosine_similarity(vectors[0], vectors[2]):.2f}")