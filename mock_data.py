import pickle
import numpy as np
from typing import Dict, Any

def load_real_embeddings(file_path: str = "Final_embeddings/embeddings_backup_8900.pkl"):
    """Load the actual embeddings file"""
    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        if isinstance(loaded_data, dict):
            if 'embeddings' in loaded_data:
                embeddings_dict = loaded_data['embeddings']
            else:
                embeddings_dict = loaded_data
                
            print(f"Successfully loaded {len(embeddings_dict)} embeddings")
            return embeddings_dict
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None

# Mock user profile data
MOCK_USER_PROFILES = {
    "test_user_123": {
        "unique_id": "test_user_123",
        "gender": "male",
        "favorite_styles": ["casual", "modern", "minimalist"],
        "favorite_colors": ["black", "navy", "white"],
        "body_shape_info": "Athletic build, 5'11\", broad shoulders",
        "style_determined_1": "Prefers clean lines and minimal patterns",
        "style_determined_2": "Values comfort without sacrificing style",
        "favorite_materials": ["cotton", "wool", "leather"]
    }
}

# Mock retailer products data (using actual product IDs from embeddings)
def create_mock_retailer_data(embeddings_dict):
    """Create mock retailer data based on actual embedding IDs"""
    mock_retailer_products = {}
    mock_product_references = {}
    
    # Sample some products from the embeddings
    sampled_products = list(embeddings_dict.keys())[:10]  # Take first 10 products
    
    # Create mock retailer tables
    retailers = ["zara_products", "uniqlo_products", "levis_products"]
    for idx, product_id in enumerate(sampled_products):
        retailer = retailers[idx % len(retailers)]
        
        # Create retailer table if it doesn't exist
        if retailer not in mock_retailer_products:
            mock_retailer_products[retailer] = {}
            
        # Add product to retailer
        mock_retailer_products[retailer][f"{retailer}_id_{idx}"] = {
            "id": f"{retailer}_id_{idx}",
            "gender": "male" if idx % 3 == 0 else "unisex",
            "image_urls": [
                f"https://example.com/{retailer}/item{idx}_1.jpg",
                f"https://example.com/{retailer}/item{idx}_2.jpg"
            ]
        }
        
        # Add product reference
        mock_product_references[product_id] = {
            "id": product_id,
            "retailer_table": retailer,
            "product_id": f"{retailer}_id_{idx}"
        }
    
    return mock_retailer_products, mock_product_references

class MockSupabase:
    """Mock Supabase client for testing"""
    def __init__(self, embeddings_dict):
        self.mock_retailer_products, self.mock_product_references = create_mock_retailer_data(embeddings_dict)
        self.data = {
            "user_profile": MOCK_USER_PROFILES,
            "product_references": self.mock_product_references,
            **self.mock_retailer_products
        }

    def table(self, table_name: str):
        return MockTable(self.data, table_name)

class MockTable:
    """Mock Supabase table operations"""
    def __init__(self, data: Dict[str, Any], table_name: str):
        self.data = data
        self.table_name = table_name
        self._filters = []
        self._selected_fields = None

    def select(self, *fields):
        self._selected_fields = fields if fields else None
        return self

    def eq(self, field: str, value: Any):
        self._filters.append(("eq", field, value))
        return self

    def in_(self, field: str, values: list):
        self._filters.append(("in", field, values))
        return self

    def single(self):
        return self

    def execute(self):
        results = []
        
        if self.table_name == "user_profile":
            for filter_type, field, value in self._filters:
                if filter_type == "eq" and field == "unique_id":
                    if value in self.data.get("user_profile", {}):
                        results.append(self.data["user_profile"][value])
        
        elif self.table_name == "product_references":
            for filter_type, field, value in self._filters:
                if filter_type == "eq" and field == "id":
                    if value in self.data.get("product_references", {}):
                        results.append(self.data["product_references"][value])
                elif filter_type == "in" and field == "id":
                    for id_value in value:
                        if id_value in self.data.get("product_references", {}):
                            results.append(self.data["product_references"][id_value])
        
        elif self.table_name in self.data:  # Retailer tables
            for filter_type, field, value in self._filters:
                if filter_type == "eq" and field == "id":
                    if value in self.data[self.table_name]:
                        results.append(self.data[self.table_name][value])
        
        return MockResponse(results)

class MockResponse:
    """Mock Supabase response"""
    def __init__(self, data):
        self.data = data

def setup_mock_environment():
    """Set up mock environment with real embeddings"""
    # Load real embeddings
    embeddings = load_real_embeddings()
    if not embeddings:
        raise Exception("Failed to load embeddings")
    
    # Create mock Supabase client with real embedding IDs
    mock_supabase = MockSupabase(embeddings)
    
    return embeddings, mock_supabase

if __name__ == "__main__":
    # Test the mock setup
    embeddings, supabase = setup_mock_environment()
    print("Mock environment created successfully!")
    print(f"Number of real embeddings: {len(embeddings)}")
    
    # Test Supabase mock
    response = supabase.table("user_profile").select("*").eq("unique_id", "test_user_123").execute()
    print("\nTest user profile:", response.data[0] if response.data else "Not found")
    
    # Test product references
    sample_product_id = list(embeddings.keys())[0]
    response = supabase.table("product_references").select("*").eq("id", sample_product_id).execute()
    print(f"\nProduct reference for {sample_product_id}:", response.data[0] if response.data else "Not found")