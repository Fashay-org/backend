import pickle
import numpy as np

def read_embeddings():
    try:
        # Load the pickle file
        print("Loading embeddings from pickle file...")
        with open('embeddings_backup_1000.pkl', 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        # Print summary
        print(f"\nTotal embeddings loaded: {len(embeddings_dict)}")
        
        # Print first few entries
        print("\nFirst 3 entries:")
        for i, (ref_id, data) in enumerate(embeddings_dict.items()):
            if i >= 3:
                break
                
            print(f"\nReference ID: {ref_id}")
            print(f"Text: {data['text']}")
            print(f"Embedding (first 5 dimensions): {data['embedding'][:5]}...")
            print(f"Embedding dimension: {len(data['embedding'])}")
            print("-" * 80)
        
        # Print random entry
        random_id = np.random.choice(list(embeddings_dict.keys()))
        print("\nRandom entry:")
        print(f"Reference ID: {random_id}")
        print(f"Text: {embeddings_dict[random_id]['text']}")
        print(f"Embedding (first 5 dimensions): {embeddings_dict[random_id]['embedding'][:5]}...")
        
        return embeddings_dict
        
    except FileNotFoundError:
        print("Error: product_embeddings.pkl not found!")
    except Exception as e:
        print(f"Error reading pickle file: {str(e)}")

if __name__ == "__main__":
    print("Starting to read embeddings...")
    embeddings = read_embeddings()
    print("\nProcess complete")