import numpy as np
from openai import OpenAI
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
class ProductRecommender:
    def __init__(self, thread_manager: Dict = None):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2"))
        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        
        # Use shared thread management if provided, otherwise create own
        if thread_manager:
            self._get_or_create_thread = thread_manager['get_thread']
            self.user_threads = thread_manager['user_threads']
            self.last_interaction = thread_manager['last_interaction']
            self.conversation_context = thread_manager['conversation_context']
        else:
            self.user_threads = {}
            self.last_interaction = {}
            self.conversation_context = {}
        print("Loading product embeddings...")
        self.assistant = self._create_assistant() 
        try:
            with open('Final_embeddings/product_embeddings_final.pkl', 'rb') as f:
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

    def _create_assistant(self):
        """Create an OpenAI assistant for product recommendations"""
        try:
            assistant = self.client.beta.assistants.create(
                name="Product Recommender",
                instructions="""You are a product recommendation assistant specializing in fashion products. Your role is to analyze queries and suggest specific clothing items and accessories based on user needs.

                Key Requirements:
                1. ALWAYS return recommendations in valid JSON format
                2. For each clothing category, provide ONE detailed description
                3. Focus on practical, specific item descriptions that can be used for product matching
                4. Consider seasonal appropriateness, occasion, and style preferences
                5. Include color, material, and style details in descriptions

                Response Format Example:
                {
                    "shirt": "A light blue cotton Oxford button-down with a slim fit and white buttons",
                    "pants": "Charcoal grey wool dress slacks with a tapered fit and flat front",
                    "shoes": "Brown leather cap-toe Oxford shoes with Goodyear welted soles",
                    "accessories": "Silver stainless steel chronograph watch with a black leather strap"
                }

                Guidelines:
                - Keep descriptions clear and specific
                - Include material, color, and style details
                - Focus on one clear item per category
                - Ensure descriptions are detailed enough for matching
                - Consider gender appropriateness when specified
                - Include fit details where relevant
                - Consider the occasion and context of the request""",
                            model="gpt-4o-mini",
                            tools=[]
                        )
            return assistant
        except Exception as e:
            print(f"Error creating assistant: {str(e)}")
            raise
    # def _get_thread_key(self, unique_id, stylist_id):
    #     return f"{unique_id}_{stylist_id}"
    # def _cleanup_old_threads(self, max_age_hours=24):
    #     current_time = time.time()
    #     threads_to_remove = []
        
    #     for thread_key, last_time in self.last_interaction.items():
    #         if current_time - last_time > max_age_hours * 3600:
    #             threads_to_remove.append(thread_key)
        
    #     for thread_key in threads_to_remove:
    #         del self.user_threads[thread_key]
    #         del self.last_interaction[thread_key]
    # def _get_or_create_thread(self, unique_id, stylist_id):
    #     thread_key = self._get_thread_key(unique_id, stylist_id)
    #     current_time = time.time()
        
    #     if thread_key in self.user_threads:
    #         self.last_interaction[thread_key] = current_time
    #         return self.user_threads[thread_key]
        
    #     thread = self.client.beta.threads.create()
    #     self.user_threads[thread_key] = thread.id
    #     self.last_interaction[thread_key] = current_time
        
    #     self._cleanup_old_threads()
        
    #     return thread.id
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

    def get_gpt_recommendations(self, query: str, profile: Dict, user_id: str, stylist_id: str) -> Dict[str, str]:
        """Get outfit recommendations with shared context awareness"""
        try:
            # Get thread key and shared context
            thread_key = f"{user_id}_{stylist_id}"
            # Initialize empty list if needed
            if thread_key not in self.conversation_context:
                self.conversation_context[thread_key] = []
                
            conversation_messages = self.conversation_context[thread_key]


            # Create context-aware prompt
            system_prompt = """You are a product recommendation assistant specializing in fashion products. 

            Consider the following context:
            Previous Conversation:
            {prev_conversation}
            
            User Profile:
            - Gender: {gender}
            - Favorite Styles: {styles}
            - Favorite Colors: {colors}
            - Body Shape: {body_shape}
            - Preferred Materials: {materials}
            - Style Preferences: {style_prefs}

            If the current query asks for alternatives or different suggestions,
            maintain the same occasion but provide different outfit recommendations.
            
            Key Requirements:
            1. ALWAYS return recommendations in valid JSON format
            2. For each clothing category, provide ONE detailed description
            3. Focus on practical, specific item descriptions that can be used for product matching
            4. Consider seasonal appropriateness, occasion, and style preferences
            5. Include color, material, and style details in descriptions
            
            Return ONLY a JSON with this exact structure:
            {{
                "occasion": "current occasion or previous if not specified",
                "shirt": "detailed description of recommended shirt",
                "pants": "detailed description of recommended pants",
                "shoes": "detailed description of recommended shoes",
                "accessories": "detailed description of recommended accessories"
            }}
            
            Guidelines:
            - Keep descriptions clear and specific
            - Include material, color, and style details
            - Focus on one clear item per category
            - Ensure descriptions are detailed enough for matching
            - Consider gender appropriateness when specified
            - Include color, material, and style details""".format(
                prev_conversation=chr(10).join(conversation_messages) if conversation_messages else "No previous conversation",
                query=query,
                gender=profile.get('gender', 'Not specified'),
                styles=', '.join(profile.get('favorite_styles', [])),
                colors=', '.join(profile.get('favorite_colors', [])),
                body_shape=profile.get('body_shape_info', ''),
                materials=', '.join(profile.get('favorite_materials', [])),
                style_prefs=', '.join(filter(None, [profile.get(f'style_determined_{i}', '') for i in range(1, 6)]))
            )

            # Get thread
            thread_id = self._get_or_create_thread(user_id, stylist_id)

            # Create message in thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": system_prompt}]
            )
            
            # Create and wait for run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                time.sleep(1)
            
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            recommendations = json.loads(messages.data[0].content[0].text.value)
            self.conversation_context[thread_key].append(f"System: {messages.data[0].content[0].text.value}")


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
        print("\n=== Starting category recommendations ===")
        category_products = {}
        
        for category, description in category_suggestions.items():
            # print(f"\nProcessing category: {category}")
            try:
                # Create fresh Supabase client for each category to avoid connection timeouts
                # print(f"Creating new Supabase client for {category}")
                supabase = create_client(
                    os.environ.get("SUPABASE_URL"),
                    os.environ.get("SUPABASE_KEY")
                )
                
                # print(f"Getting embedding for {category}")
                embedding = self.get_embedding(description)
                if embedding:
                    # print("Got embedding, calculating similarities")
                    query_embedding = np.array(embedding).reshape(1, -1)
                    similarities = cosine_similarity(query_embedding, self.embeddings_array)[0]
                    top_indices = similarities.argsort()[::-1]
                    
                    # print(f"Starting product search for {category}")
                    for idx in top_indices:
                        product_id = self.product_ids[idx]
                        # print(f"Checking product ID: {product_id}")
                        
                        try:
                            # print("Querying product_references table")
                            reference_response = supabase.table("product_references")\
                                .select("retailer_table, product_id")\
                                .eq("id", product_id)\
                                .execute()
                                
                            if not reference_response.data or not reference_response.data[0]:
                                print("No reference data found")
                                continue
                                
                            ref_data = reference_response.data[0]
                            retailer_table = ref_data['retailer_table']
                            original_product_id = ref_data['product_id']
                            
                            # print(f"Querying retailer table: {retailer_table}")
                            product_response = supabase.table(retailer_table)\
                                .select("*")\
                                .eq("id", original_product_id)\
                                .execute()
                            
                            if not product_response.data or not product_response.data[0]:
                                print("No product data found")
                                continue
                                
                            # print("Found product data, checking gender match")
                            prod_data = product_response.data[0]
                            product_gender = prod_data.get('gender', '')
                            
                            if not user_gender or product_gender == user_gender or product_gender == 'unisex':
                                # print(f"Found matching product for {category}")
                                category_products[category] = {
                                    'product_id': product_id,
                                    'similarity_score': float(similarities[idx]),
                                    'product_text': self.embeddings_dict[product_id]['text'],
                                    'gender': product_gender,
                                    'retailer': retailer_table,
                                    'brand': prod_data.get('brand', ''),
                                    'image_urls': prod_data.get('image_urls', []),
                                    'url': prod_data.get('url', ''),
                                    'price': prod_data.get('price', ''),
                                    'name': prod_data.get('name', ''),
                                    'description': prod_data.get('description', ''),
                                    'colors': prod_data.get('colors', []),
                                }
                                break
                            else:
                                print(f"Gender mismatch: Product={product_gender}, User={user_gender}")
                                
                        except Exception as e:
                            print(f"Error during product check: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Error processing category {category}: {str(e)}")
                continue

            if category not in category_products:
                print(f"Warning: No suitable product found for category {category}")
        
        print("\n=== Finished category recommendations ===")
        return category_products
    def get_final_recommendations(self, query: str, category_suggestions: Dict[str, str], 
                                category_products: Dict[str, Dict], user_id: str, stylist_id: str) -> str:
        """Get final styled recommendations in strict JSON format"""
        try:
            thread_id = self._get_or_create_thread(user_id, stylist_id)
            
            system_prompt = """You are a fashion recommendation system. Return a JSON object with EXACTLY this structure:
            {
                'items': [
                    {
                        'category': 'Category name',
                        'product_id': 'ID of the product',
                        'styling_tips': ['Tip 1', 'Tip 2']
                    }
                ],
                'query_fit': 'How this outfit matches the query'
            }"""

            # Format products with all available information
            products_list = []
            for category, product in category_products.items():
                products_list.append(
                    f"{category.title()}:\n"
                    f"- Product: {product['product_text']}\n"
                    f"- ID: {product['product_id']}\n"
                    f"- Gender: {product.get('gender', 'Not specified')}\n"
                    f"- Original Suggestion: {category_suggestions[category]}"
                )
            
            products_formatted = "\n\n".join(products_list)

            user_prompt = f"""Query: {query}

            Available Products:
            {products_formatted}

            Create a cohesive outfit recommendation in the required JSON format:
            1. Include every selected item with its ID
            2. Provide specific styling tips for each piece
            3. Explain how the pieces work together
            4. Focus on how the outfit fulfills the original query
            5. Maintain the exact JSON structure specified"""

            # Create message in thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": system_prompt + "\n\n" + user_prompt}]
            )
            
            # Create and wait for run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                time.sleep(1)
            
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0].content[0].text.value
            
            # Parse and validate JSON
            recommendation = json.loads(latest_message)
            required_fields = ['items', 'query_fit']
            for field in required_fields:
                if field not in recommendation:
                    raise ValueError(f"Missing required field: {field}")
            
            for item in recommendation['items']:
                required_item_fields = ['product_id', 'styling_tips']
                for field in required_item_fields:
                    if field not in item:
                        raise ValueError(f"Missing required item field: {field}")
            
            return json.dumps(recommendation, indent=2)
            
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

    async def process_query(self, user_id: str, stylist_id: str, query: str) -> Dict[str, Any]:
        """Main function to process user query and return recommendations"""
        try:
            # Get user profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {"error": "User profile not found"}
            
            # Get user gender
            user_gender = profile.get('gender', None)
            
            print("getting gpt recommendations")
            # Get initial category-based GPT recommendations
            category_suggestions = self.get_gpt_recommendations(
                query=query,
                profile=profile,
                user_id=user_id,
                stylist_id=stylist_id  # This will use the same thread as FashionAssistant
            )
            if not category_suggestions:
                return {"error": "Failed to generate GPT recommendations"}
            
            # Get product recommendations for each category with gender matching
            import time
            timenew = time.time()
            # print("getting category recommendations")
            category_products = self.get_category_recommendations(category_suggestions, user_gender)
            # print("completed category recommendations")
            # print("Time taken for get_category_recommendations: ", time.time()-timenew)
            if not category_products:
                return {"error": "No suitable products found matching gender preference"}
            # print("getting final recommendations")
            # Get final styled recommendations
            final_recommendations = self.get_final_recommendations(
                query, 
                category_suggestions,
                category_products,
                user_id,
                stylist_id
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
                        # 'product_text': product['name'],
                        'product_text': suggestion,
                        'similarity_score': product['similarity_score'],
                        'gender': product.get('gender'),
                        'retailer': product.get('retailer'),
                        'brand': product.get('brand', ''),
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