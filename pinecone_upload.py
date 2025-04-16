import pickle
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time

load_dotenv()

class PineconeUploader:
    def __init__(self, index_name="fashayrecommnder"):
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.index_name = index_name
        self.batch_size = 100
        self.embeddings_dict = {}
        self.product_ids = []
        self.embeddings_array = None

    def load_embeddings(self):
        """Load and process embeddings from pickle file"""
        print("Loading embeddings from pickle file...")
        try:
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
            
            # Convert to numpy array
            print("Converting embeddings to numpy array...")
            self.embeddings_array = np.array([
                self.embeddings_dict[pid]['embedding'] 
                for pid in self.product_ids
            ])
            
            print(f"Embeddings array shape: {self.embeddings_array.shape}")
            return True

        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False

    def upload_embeddings(self):
        """Upload embeddings to Pinecone"""
        if not self.load_embeddings():
            return

        total_items = len(self.product_ids)
        print(f"\nUploading {total_items} embeddings...")
        
        successful_uploads = 0
        failed_batches = []

        for start_idx in tqdm(range(0, total_items, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, total_items)
            batch_vectors = []

            # Prepare batch
            for idx in range(start_idx, end_idx):
                pid = self.product_ids[idx]
                vector = self.embeddings_array[idx].tolist()  # Convert numpy array to list
                
                batch_vectors.append({
                    'id': str(pid),
                    'values': vector,
                    'metadata': {
                        'text': self.embeddings_dict[pid]['text'],
                        'product_id': str(pid)
                    }
                })

            # Upload batch
            try:
                self.index.upsert(vectors=batch_vectors)
                successful_uploads += len(batch_vectors)
                time.sleep(0.1)  # Small delay to avoid rate limits
            except Exception as e:
                print(f"\nError uploading batch {start_idx}-{end_idx}: {str(e)}")
                failed_batches.append((start_idx, end_idx))
                continue

        print(f"\nUpload completed:")
        print(f"Successfully uploaded: {successful_uploads}/{total_items} embeddings")
        if failed_batches:
            print("Failed batch ranges:", failed_batches)

        self.verify_upload()

    def verify_upload(self, sample_size=5):
        """Verify uploaded vectors"""
        print("\nVerifying upload...")
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"Total vectors in index: {stats.total_vector_count}")
            
            # Query random samples
            print("\nSample vectors:")
            sample_ids = self.product_ids[:sample_size]
            results = self.index.fetch(
                ids=sample_ids,
                include_values=True,
                include_metadata=True
            )
            
            for vector_id, vector_data in results['vectors'].items():
                print(f"\nProduct ID: {vector_id}")
                print(f"Text: {vector_data.metadata['text']}")
                print(f"Vector length: {len(vector_data.values)}")
                print(f"First 5 values: {vector_data.values[:5]}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error during verification: {str(e)}")

if __name__ == "__main__":
    uploader = PineconeUploader()
    uploader.upload_embeddings()