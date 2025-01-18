import pickle
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json
import time

load_dotenv()

class EmbeddingTransfer:
    def __init__(self):
        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        self.batch_size = 50  # Adjust based on your needs

    def load_pickle_file(self, file_path):
        """Load embeddings from pickle file"""
        print("Loading pickle file...")
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data.get('embeddings', data)
            return None
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
            return None

    def prepare_embedding_batch(self, embeddings_dict, start_idx, end_idx):
        """Prepare a batch of embeddings for upload"""
        batch_items = []
        items = list(embeddings_dict.items())[start_idx:end_idx]
        
        for product_id, data in items:
            if not isinstance(data, dict) or 'embedding' not in data or 'text' not in data:
                continue
                
            embedding = data['embedding']
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Split the embedding into three parts
            embedding_part1 = embedding[:1000]
            embedding_part2 = embedding[1000:2000]
            embedding_part3 = embedding[2000:]
            
            # Prepare additional data
            additional_data = {k: v for k, v in data.items() if k not in ['embedding', 'text']}
            
            batch_items.append({
                "product_id": str(product_id),
                "embedding_part1": embedding_part1,
                "embedding_part2": embedding_part2,
                "embedding_part3": embedding_part3,
                "text_data": data['text'],
                "additional_data": additional_data
            })
        
        return batch_items

    async def transfer_embeddings(self, file_path):
        """Transfer embeddings from pickle to Supabase"""
        embeddings_dict = self.load_pickle_file(file_path)
        if not embeddings_dict:
            print("Failed to load embeddings")
            return

        total_items = len(embeddings_dict)
        print(f"Found {total_items} embeddings to transfer")

        # Process in batches
        for start_idx in tqdm(range(0, total_items, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, total_items)
            batch = self.prepare_embedding_batch(embeddings_dict, start_idx, end_idx)
            
            if not batch:
                continue

            try:
                # Upload batch to Supabase
                response = self.supabase.table("product_embeddings").insert(batch).execute()
                
                if not response.data:
                    print(f"Warning: No data returned for batch {start_idx}-{end_idx}")
                
                # Add delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error uploading batch {start_idx}-{end_idx}: {str(e)}")
                continue

    def verify_transfer(self):
        """Verify the transfer was successful"""
        try:
            response = self.supabase.table("product_embeddings").select("count").execute()
            count = response.count
            print(f"Successfully transferred {count} embeddings to Supabase")
            
            # Verify random samples
            sample_response = self.supabase.table("product_embeddings")\
                .select("product_id, embedding")\
                .limit(5)\
                .execute()
                
            print("\nSample verification:")
            for item in sample_response.data:
                print(f"Product ID: {item['product_id']}")
                print(f"Embedding length: {len(item['embedding'])}")
                print("---")
                
        except Exception as e:
            print(f"Error verifying transfer: {str(e)}")

if __name__ == "__main__":
    import asyncio
    
    # File path to your pickle file
    PICKLE_FILE = "Final_embeddings\product_embeddings_final.pkl"
    
    transfer = EmbeddingTransfer()
    
    # Run the transfer
    asyncio.run(transfer.transfer_embeddings(PICKLE_FILE))
    
    # Verify the transfer
    transfer.verify_transfer()