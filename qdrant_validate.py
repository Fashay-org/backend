import pickle
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

class QdrantUploader:
    def __init__(self, collection_name="product_embeddings"):
        self.client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=300
        )
        self.collection_name = collection_name
        self.batch_size = 100
        self.vector_size = 3072
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

    def recreate_collection(self):
        """Recreate the collection from scratch"""
        try:
            # Delete if exists
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)

            print(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print("Collection created successfully")
            return True
        except Exception as e:
            print(f"Error with collection creation: {str(e)}")
            return False

    def upload_embeddings(self):
        """Upload embeddings to Qdrant"""
        if not self.load_embeddings():
            return
        
        if not self.recreate_collection():
            return

        total_items = len(self.product_ids)
        print(f"\nUploading {total_items} embeddings...")
        
        successful_uploads = 0
        failed_batches = []

        for start_idx in tqdm(range(0, total_items, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, total_items)
            batch_points = []

            # Prepare batch
            for idx in range(start_idx, end_idx):
                pid = self.product_ids[idx]
                vector = self.embeddings_array[idx].tolist()  # Convert numpy array to list
                
                point = PointStruct(
                    id=pid,
                    vector=vector,
                    payload={
                        "text_data": self.embeddings_dict[pid]['text'],
                        "product_id": pid
                    }
                )
                batch_points.append(point)

            # Upload batch
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                    wait=True
                )
                successful_uploads += len(batch_points)
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
        """Verify uploaded points"""
        print("\nVerifying upload...")
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Total points in collection: {collection_info.points_count}")
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=sample_size
            )
            
            print("\nSample points:")
            for point in results[0]:
                print(f"\nProduct ID: {point.payload.get('product_id', 'NO_ID')}")
                print(f"Text: {point.payload.get('text_data', 'NO_TEXT')}")
                print(f"Vector length: {len(point.vector) if point.vector is not None else 'None'}")
                if point.vector is not None:
                    print(f"First 5 values: {point.vector[:5]}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error during verification: {str(e)}")

if __name__ == "__main__":
    uploader = QdrantUploader()
    uploader.upload_embeddings()