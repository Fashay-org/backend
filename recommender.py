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
from pinecone import Pinecone
class ProductRecommender:
    def __init__(self, thread_manager: Dict = None):
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2"))
        self.supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = self.pc.Index("fashayrecommnder")
        
        # Thread management setup
        if thread_manager:
            self._get_or_create_thread = thread_manager['get_thread']
            self.user_threads = thread_manager['user_threads']
            self.last_interaction = thread_manager['last_interaction']
            self.conversation_context = thread_manager['conversation_context']
        else:
            self.user_threads = {}
            self.last_interaction = {}
            self.conversation_context = {}

        self.assistant = self._create_assistant()

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
                "category 1": "detailed description of recommended category 1",
                "category 2": "detailed description of recommended category 2",
                "category 3": "detailed description of recommended category 3",
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
            try:
                # Create fresh Supabase client for each category
                supabase = create_client(
                    os.environ.get("SUPABASE_URL"),
                    os.environ.get("SUPABASE_KEY")
                )
                
                embedding = self.get_embedding(description)
                if embedding:
                    # Query Pinecone directly with the embedding
                    query_response = self.index.query(
                        vector=embedding,
                        top_k=50,  # Get more results to account for gender filtering
                        include_metadata=True
                    )
                    
                    for match in query_response['matches']:
                        product_id = match['id']
                        
                        try:
                            # Get product reference data
                            reference_response = supabase.table("product_references")\
                                .select("retailer_table, product_id")\
                                .eq("id", product_id)\
                                .execute()
                                
                            if not reference_response.data:
                                continue
                                
                            ref_data = reference_response.data[0]
                            retailer_table = ref_data['retailer_table']
                            original_product_id = ref_data['product_id']
                            
                            # Get product details
                            product_response = supabase.table(retailer_table)\
                                .select("*")\
                                .eq("id", original_product_id)\
                                .execute()
                            
                            if not product_response.data:
                                continue
                                
                            prod_data = product_response.data[0]
                            product_gender = prod_data.get('gender', '')
                            
                            if not user_gender or product_gender == user_gender or product_gender == 'unisex':
                                category_products[category] = {
                                    'product_id': product_id,
                                    'similarity_score': float(match['score']),
                                    'product_text': match['metadata']['text'],
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
                                
                        except Exception as e:
                            print(f"Error during product check: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Error processing category {category}: {str(e)}")
                continue

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

    async def process_query(self, user_id: str, stylist_id: str, query: str, selected_item: Dict = None) -> Dict[str, Any]:
        """Main function to process user query and return recommendations"""
        try:
            # Get user profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                return {"error": "User profile not found"}
            
            # Get user gender
            user_gender = profile.get('gender', None)
            selected_item_text = (f"ID: {selected_item['token_name']} | Caption: {selected_item['caption']}" if selected_item else 'None')
            enhanced_query = query if not selected_item else f"Find items that coordinate with: {selected_item_text}. Context: {query}"
            print("getting gpt recommendations")
            # Get initial category-based GPT recommendations
            print("enhanced_query", enhanced_query)
            category_suggestions = self.get_gpt_recommendations(
                query=enhanced_query,
                profile=profile,
                user_id=user_id,
                stylist_id=stylist_id  # This will use the same thread as FashionAssistant
            )
            print("completed gpt recommendations", category_suggestions)
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
            print(final_recommendations, "final_recommendations")
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
            
            wardrobe_items = []
            if selected_item:
                wardrobe_items.append(selected_item['token_name'])
            
            return {
                "success": True,
                "category_suggestions": category_suggestions,
                "final_recommendation": final_recommendations,
                "products": products_output,
                "user_gender": user_gender,
                "value": wardrobe_items
            }
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {"error": str(e)}