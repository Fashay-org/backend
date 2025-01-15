from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("fashayrecommnder")

# Sample product IDs to fetch
product_ids = [
    "899bd773-e79c-4778-9b67-b4bcb9de5716",  # Sample ID 1
    "9d58ddc9-e75c-4cb6-86b3-7e660f9142c5"   # Sample ID 2
]

try:
    # Fetch the vectors
    results = index.fetch(
        ids=product_ids
    )

    # Display results
    print("\nFetched Products:")
    if hasattr(results, 'vectors'):
        for vector_id, vector_data in results.vectors.items():
            print(f"\nProduct ID: {vector_id}")
            if hasattr(vector_data, 'metadata'):
                print(f"Text Description: {vector_data.metadata.get('text', 'No text available')}")
            if hasattr(vector_data, 'values'):
                print(f"Vector dimension: {len(vector_data.values)}")
                print(f"First 5 vector values: {vector_data.values[:5]}")
            print("-" * 80)
    else:
        print("No vectors found in the response")

    # Print index statistics
    stats = index.describe_index_stats()
    print("\nIndex Statistics:")
    print(f"Total vectors in index: {stats.total_vector_count if hasattr(stats, 'total_vector_count') else 'Unknown'}")
    print(f"Vector dimension: {stats.dimension if hasattr(stats, 'dimension') else 'Unknown'}")

except Exception as e:
    print(f"Error fetching vectors: {str(e)}")