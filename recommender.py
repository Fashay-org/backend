import numpy as np
from openai import OpenAI
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json

class ProductRecommender:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2"))
        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        
        print("Loading product embeddings...")
        try:
            with open('Final_embeddings/embeddings_backup_8900.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, dict):
                self.embeddings_dict = loaded_data.get('embeddings', loaded_data)
            else:
                raise ValueError("Unexpected format in embeddings file")

            # Validate embeddings
            invalid_ids = [
                pid for pid, data in list(self.embeddings_dict.items())
                if not isinstance(data, dict) or 'embedding' not in data or 'text' not in data
            ]
            for pid in invalid_ids:
                del self.embeddings_dict[pid]

            self.product_ids = list(self.embeddings_dict.keys())
            self.embeddings_array = np.array([
                self.embeddings_dict[pid]['embedding'] 
                for pid in self.product_ids
            ])
            
            print(f"Successfully loaded {len(self.product_ids)} product embeddings")
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            raise

    async def get_user_profile(self, user_id: str) -> Dict:
        """Fetch user profile and preferences from database"""
        try:
            profile_response = self.supabase.table("user_profile")\
                .select("*")\
                .eq("unique_id", user_id)\
                .execute()
            
            if not profile_response.data:
                return None
                
            wardrobe_response = self.supabase.table("wardrobe")\
                .select("gender")\
                .eq("unique_id", user_id)\
                .execute()
            
            profile_data = profile_response.data[0]
            profile_data['gender'] = wardrobe_response.data[0].get('gender') if wardrobe_response.data else None
            
            return profile_data
            
        except Exception as e:
            print(f"Error fetching user profile: {str(e)}")
            return None

    def get_gpt_recommendations(self, query: str, profile: Dict) -> Dict[str, str]:
        """Get outfit recommendations from GPT-4 in category format"""
        try:
            system_prompt = (
                "You are a professional fashion stylist. Based on the query and user profile, "
                "suggest specific items for each clothing category needed for a complete outfit. "
                "Return ONLY a JSON dictionary where keys are basic clothing categories "
                "(like 'shirt', 'pants', 'dress', 'shoes', 'accessories' etc.) and values are detailed "
                "descriptions of recommended items. Each description should include color, style, fit, and material. "
                "Example format: {'dress': 'A flowy midi dress in emerald green silk with...', "
                "'shoes': 'Strappy beige sandals with...'}. Only include relevant categories for the specific outfit. "
                "Consider the user's gender when making recommendations."
            )

            user_prompt = (
                f"User Query: {query}\n\n"
                f"Profile Information:\n"
                f"- Gender: {profile.get('gender', 'Not specified')}\n"
                f"- Favorite Styles: {', '.join(profile.get('favorite_styles', []))}\n"
                f"- Favorite Colors: {', '.join(profile.get('favorite_colors', []))}\n"
                f"- Body Shape: {profile.get('body_shape_info', '')}\n"
                f"- Preferred Materials: {', '.join(profile.get('favorite_materials', []))}\n"
                f"- Style Preferences: {', '.join(filter(None, [profile.get(f'style_determined_{i}', '') for i in range(1, 6)]))}"
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                response_format={ "type": "json_object" }
            )
            
            # Parse the JSON response
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations

        except Exception as e:
            print(f"Error getting GPT recommendations: {str(e)}")
            return None

    def get_embedding(self, text: str | dict) -> List[float]:
            """Get embedding for text using OpenAI API"""
            try:
                # print("text", text, type(text))
                
                # Handle dictionary input
                if isinstance(text, dict):
                    # Extract text from dictionary
                    input_text = text.get('description', '') if 'description' in text else str(text)
                else:
                    input_text = str(text)
                
                # Clean the text
                clean_text = input_text.replace("\n", " ").strip()
                
                # Get embedding
                response = self.client.embeddings.create(
                    input=[clean_text],
                    model="text-embedding-3-large",
                    dimensions=3072
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                return None

    def get_category_recommendations(self, category_suggestions: Dict[str, str], user_gender: str = None) -> Dict[str, Dict]:
        """Get product recommendations for each category, considering gender"""
        category_products = {}
        
        for category, description in category_suggestions.items():
            # Get embedding for category description
            embedding = self.get_embedding(description)
            if embedding:
                # Get initial matches
                query_embedding = np.array(embedding).reshape(1, -1)
                similarities = cosine_similarity(query_embedding, self.embeddings_array)[0]
                
                # Get top matches sorted by similarity
                top_indices = similarities.argsort()[::-1]
                
                # Find the first product that matches the gender requirement
                product_found = False
                for idx in top_indices:
                    product_id = self.product_ids[idx]
                    
                    try:
                        # Extract retailer table and original product id
                        reference_response = self.supabase.table("product_references")\
                            .select("retailer_table, product_id")\
                            .eq("id", product_id)\
                            .execute()
                            
                        if not reference_response.data or not reference_response.data[0]:
                            continue
                            
                        ref_data = reference_response.data[0]  # Get first item from response data
                        retailer_table = ref_data['retailer_table']
                        original_product_id = ref_data['product_id']
                        
                        # Get product details including gender
                        product_response = self.supabase.table(retailer_table)\
                            .select("*")\
                            .eq("id", original_product_id)\
                            .execute()
                            
                        
                        # exit(-1)
                        if not product_response.data or not product_response.data[0]:
                            continue
                            
                        prod_data = product_response.data[0]  # Get first item from response data
                        product_gender = prod_data.get('gender', '')
                        
                        # Check if gender matches or if no gender preference
                        if not user_gender or product_gender == user_gender or product_gender == 'unisex':
                            category_products[category] = {
                                'product_id': product_id,
                                'similarity_score': float(similarities[idx]),
                                'product_text': self.embeddings_dict[product_id]['text'],
                                'gender': product_gender,
                                'retailer': retailer_table,
                                'image_urls': prod_data.get('image_urls', []),
                                'url': prod_data.get('url', ''),  # Product URL
                                'price': prod_data.get('price', ''),  # Product price
                                'name': prod_data.get('name', ''),  # Product name
                                'description': prod_data.get('description', '')
                            }
                            # print(category_products[category], " category_products[category]")
                            product_found = True
                            break
                            
                    except Exception as e:
                        print(f"Error checking product: {str(e)}")
                        continue
                
                if not product_found:
                    print(f"Warning: No gender-appropriate product found for category {category}")
        
        return category_products
    def get_final_recommendations(self, query: str, category_suggestions: Dict[str, str], 
                                    category_products: Dict[str, Dict]) -> str:
            """Get final styled recommendations in strict JSON format"""
            try:
                system_prompt = (
                    "You are a fashion recommendation system. Return a JSON object with EXACTLY this structure:\n"
                    "{\n"
                    "    'items': [\n"
                    "        {\n"
                    "            'category': 'Category name',\n"
                    "            'product_id': 'ID of the product',\n"
                    "            'styling_tips': ['Tip 1', 'Tip 2']\n"
                    "        }\n"
                    "    ],\n"
                    "    'query_fit': 'How this outfit matches the query'\n"
                    "}\n\n"
                    "Ensure evrything is in the exact JSON structure. Keep it concise and relevant to the query."
                )

                # Format products with all available information
                products_list = []
                for category, product in category_products.items():
                    products_list.append(
                        f"{category.title()}:\n"
                        f"- Product: {product['product_text']}\n"
                        f"- ID: {product['product_id']}\n"
                        f"- Gender: {product['gender']}\n"
                        f"- Original Suggestion: {category_suggestions[category]}"
                    )
                
                products_formatted = "\n\n".join(products_list)

                user_prompt = (
                    f"Query: {query}\n\n"
                    f"Available Products:\n{products_formatted}\n\n"
                    "Create a cohesive outfit recommendation in the required JSON format:\n"
                    "1. Include every selected item with its ID\n"
                    "2. Provide specific styling tips for each piece\n"
                    "3. Explain how the pieces work together\n"
                    "4. Focus on how the outfit fulfills the original query\n"
                    "5. Maintain the exact JSON structure specified"
                )

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent structure
                    max_tokens=800,
                    response_format={ "type": "json_object" }
                )
                
                # Parse and validate JSON structure
                try:
                    recommendation = json.loads(response.choices[0].message.content)
                    
                    # Validate required fields
                    required_fields = ['items', 'query_fit']
                    for field in required_fields:
                        if field not in recommendation:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Validate items structure
                    for item in recommendation['items']:
                        required_item_fields = ['product_id', 'styling_tips']
                        for field in required_item_fields:
                            if field not in item:
                                raise ValueError(f"Missing required item field: {field}")
                    
                    # Return the validated JSON as a string
                    return json.dumps(recommendation, indent=2)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing GPT response as JSON: {str(e)}")
                    return None
                    
                except ValueError as e:
                    print(f"Invalid JSON structure: {str(e)}")
                    return None

            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                return None

    async def get_product_images(self, product_reference_ids: List[str]) -> Dict[str, List[str]]:
        """Fetch product images from appropriate retailer tables"""
        try:
            # First get product_id and retailer_table for each reference
            response = self.supabase.table("product_references")\
                .select("id, product_id, retailer_table")\
                .in_("id", product_reference_ids)\
                .execute()
                
            if not response.data:
                return {}
                
            # Group products by retailer table
            retailer_products = {}
            id_mapping = {}  # Maps reference_id to {retailer, product_id}
            
            for item in response.data:
                if item['retailer_table'] not in retailer_products:
                    retailer_products[item['retailer_table']] = []
                retailer_products[item['retailer_table']].append(item['product_id'])
                id_mapping[item['id']] = {
                    'retailer': item['retailer_table'],
                    'product_id': item['product_id']
                }
            
            # Get images from each retailer table
            images_dict = {}
            
            for retailer, product_ids in retailer_products.items():
                retailer_response = self.supabase.table(retailer)\
                    .select("id, image_urls")\
                    .in_("id", product_ids)\
                    .execute()
                    
                if retailer_response.data:
                    for product in retailer_response.data:
                        for ref_id, mapping in id_mapping.items():
                            if mapping['retailer'] == retailer and mapping['product_id'] == product['id']:
                                images_dict[ref_id] = product.get('image_urls', [])
            
            return images_dict
            
        except Exception as e:
            print(f"Error fetching product images: {str(e)}")
            return {}

    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Main function to process user query and return recommendations"""
        try:
            # Get user profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {"error": "User profile not found"}
            
            # Get user gender
            user_gender = profile.get('gender', None)
            
            # Get initial category-based GPT recommendations
            category_suggestions = self.get_gpt_recommendations(query, profile)
            if not category_suggestions:
                return {"error": "Failed to generate GPT recommendations"}
            
            # Get product recommendations for each category with gender matching
            import time
            timenew = time.time()
            category_products = self.get_category_recommendations(category_suggestions, user_gender)
            print("Time taken for get_category_recommendations: ", time.time()-timenew)
            if not category_products:
                return {"error": "No suitable products found matching gender preference"}
            
            # Get final styled recommendations
            final_recommendations = self.get_final_recommendations(
                query, 
                category_suggestions,
                category_products
            )

            # Get product images
            product_ids = [prod['product_id'] for prod in category_products.values()]
            product_images = await self.get_product_images(product_ids)
            
            # Prepare output
            products_output = []
            for category, suggestion in category_suggestions.items():
                if category in category_products:
                    product = category_products[category]
                    products_output.append({
                        'category': category,
                        'suggestion': suggestion,
                        'product_id': product['product_id'],
                        'product_text': product['product_text'],
                        'similarity_score': product['similarity_score'],
                        'gender': product.get('gender'),
                        'retailer': product.get('retailer'),
                        'image_urls': product_images.get(product['product_id'], []),
                        'url': product.get('url', ''),
                        'price': product.get('price', ''),
                        'name': product.get('name', ''),
                        'description': product.get('description', '')
                    })
            

            
            return {
                "success": True,
                "category_suggestions": category_suggestions,
                "final_recommendation": final_recommendations,
                "products": products_output,
                "user_gender": user_gender
            }
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {"error": str(e)}