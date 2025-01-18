import pickle
import numpy as np

class EmbeddingsReader:
    def __init__(self):
        self.embeddings_dict = {}
        self.product_ids = []
        self.embeddings_array = None

    def load_embeddings(self):
        try:
            print("Loading embeddings from pickle file...")
            with open('Final_embeddings\product_embeddings_final.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, dict):
                self.embeddings_dict = loaded_data.get('embeddings', loaded_data)
            else:
                raise ValueError("Unexpected format in embeddings file")

            print(f"\nInitial embeddings count: {len(self.embeddings_dict)}")

            # Validate embeddings
            invalid_ids = [
                pid for pid, data in list(self.embeddings_dict.items())
                if not isinstance(data, dict) or 'embedding' not in data or 'text' not in data
            ]
            
            print(f"Found {len(invalid_ids)} invalid entries")
            
            for pid in invalid_ids:
                del self.embeddings_dict[pid]

            self.product_ids = list(self.embeddings_dict.keys())
            self.embeddings_array = np.array([
                self.embeddings_dict[pid]['embedding'] 
                for pid in self.product_ids
            ])

            # Print summary
            print(f"\nFinal valid embeddings count: {len(self.product_ids)}")
            print(f"Embeddings array shape: {self.embeddings_array.shape}")
            
            # Show some examples
            print("\nFirst 3 entries:")
            for i, pid in enumerate(self.product_ids[:3]):
                print(f"\nProduct ID: {pid}")
                print(f"Text: {self.embeddings_dict[pid]['text']}")
                print(f"Embedding length: {len(self.embeddings_dict[pid]['embedding'])}")  # Changed from shape to len
                print(f"First 5 embedding values: {self.embeddings_dict[pid]['embedding'][:5]}")
                print("-" * 80)

        except FileNotFoundError:
            print("Error: Embeddings file not found!")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")

if __name__ == "__main__":
    reader = EmbeddingsReader()
    reader.load_embeddings()