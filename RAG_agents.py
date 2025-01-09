from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Optional
from langgraph.graph import Graph, StateGraph, START, END
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import time
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langgraph.graph.message import add_messages
import asyncio

# Import the ProductRecommender
from recommender import ProductRecommender

# Load environment variables and setup
load_dotenv()
client = wrap_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY2")))

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fashay"

# Global dict to store assistant instances
_fashion_assistant_instances = {}

# Define the state structure
class State(TypedDict):
    messages: Annotated[List[str], add_messages]
    stylist_id: str
    user_query: str
    unique_id: str
    image_id: str
    wardrobe_data: List[Dict[str, str]]
    response: str
    needs_recommendations: bool
    product_recommendations: Optional[Dict[str, Any]]
    shopping_analysis: Optional[Dict[str, Any]]
    wardrobe_response: Optional[Dict[str, Any]]

class FashionAssistant:
    def __init__(self, stylist_id="reginald"):
            self.current_stylist_id = stylist_id.lower()
            self.user_threads = {}
            self.conversation_context = {}
            self.last_interaction = {}
            self.max_history = 10
            self.stylist_personalities = {
                "reginald": """I am Reginald, a men's fashion expert with a keen eye for sophisticated yet practical styling. 
                I specialize in creating versatile looks that combine classic elements with modern trends.""",
                
                "eliza": """I am Eliza, a fashion curator with an eye for elevated, sophisticated style. 
                I excel at creating polished looks that seamlessly blend timeless elegance with contemporary fashion.""",
                
                "lilia": """I am Lilia, a body-positive fashion stylist who celebrates individual beauty. 
                I specialize in creating flattering looks that make people feel confident and comfortable."""
            }
            
            # Initialize the product recommender with shared thread management
            # self.product_recommender = ProductRecommender()
            self.product_recommender = ProductRecommender(
                thread_manager={
                    'get_thread': self._get_or_create_thread,
                    'user_threads': self.user_threads,
                    'last_interaction': self.last_interaction,
                    'conversation_context': self.conversation_context
                }
            )
            
            # Initialize the graph
            self.workflow = StateGraph(State)
            
            # Add nodes
            self.workflow.add_node("process_input", self.process_user_input)
            self.workflow.add_node("handle_question", self.handle_question)  # New node
            self.workflow.add_node("check_recommendation_need", self.check_recommendation_need)
            self.workflow.add_node("generate_wardrobe_response", self.generate_wardrobe_response)
            self.workflow.add_node("get_product_recommendations", self.get_product_recommendations)
            self.workflow.add_node("format_product_response", self.format_product_response)
            
            # Add edges
            self.workflow.add_edge(START, "process_input")
            # self.workflow.add_edge("process_input", "check_recommendation_need")
            # Add conditional edges from process_input based on question need
            self.workflow.add_conditional_edges(
                "process_input",
                self.should_ask_question,  # New condition
                {
                    True: "handle_question",  # If we need more info, go to question handler
                    False: "check_recommendation_need"  # Otherwise continue normal flow
                }
            )

            # Add edge from question handler to END
            self.workflow.add_edge("handle_question", END)

            # Add conditional edges based on shopping need
            self.workflow.add_conditional_edges(
                "check_recommendation_need",
                self.should_get_recommendations,
                {
                    True: "get_product_recommendations",
                    False: "generate_wardrobe_response"
                }
            )
            
            # For product recommendation path
            self.workflow.add_edge("get_product_recommendations", "format_product_response")
            
            # Add final edges
            self.workflow.add_conditional_edges(
                "format_product_response",
                self.should_end,
                [END]
            )
            
            self.workflow.add_edge("generate_wardrobe_response", END)  # Direct path to END for wardrobe-only response
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY2")) 
            # Compile the graph
            self.graph = self.workflow.compile()
            self.assistant = self._create_assistant()

    def _create_assistant(self):
        """Create the fashion stylist assistant with updated instructions"""
        stylist_personality = self.stylist_personalities.get(
            self.current_stylist_id,
            "I am your personal fashion stylist, focused on helping you create stylish and confident looks."
        )

        instructions = f"""You are an expert fashion stylist assistant named {self.current_stylist_id.capitalize()}.\n\n{stylist_personality}\n\n"""

        try:
            assistant = client.beta.assistants.create(
                name="Fashion Stylist",
                instructions=instructions + r"""
                You help users with outfit recommendations using chain of thought reasoning. Follow this process:

                THOUGHT PROCESS:
                1. Analyze what information you have:
                - Occasion/event details
                - Style preferences
                - Wardrobe items
                - Target item (if any)
                
                2. If information is missing, respond with:
                {
                    "text": "Your question to gather more information",
                    "value": [],
                    "needs_info": true,
                    "context": {
                        "understood": ["what you know"],
                        "missing": ["what you need to know"],
                        "question": "your specific question"
                    }
                }

                3. If you have enough information, respond with:
                {
                    "text": "Your styling advice and explanation",
                    "value": [
                        {
                            "category": "item category (shirt/pants/etc)",
                            "product_id": "item identifier",
                            "styling_tips": ["tip 1", "tip 2"]
                        }
                    ],
                    "recommendations": {
                        "category_suggestions": {
                            "occasion": "event description",
                            "shirt": "detailed shirt description",
                            "pants": "detailed pants description",
                            "shoes": "detailed shoes description",
                            "accessories": "detailed accessories description"
                        },
                        "products": [
                            // Product details will be filled by the system
                        ]
                    }
                }

                Example - When Gathering Information:
                User: "I need an outfit"
                Assistant's Thought Process:
                1. Very general request, missing critical details
                2. Need occasion and style preferences
                3. Occasion is most important for formality level
                Response: {
                    "text": "I'll help you create the perfect outfit. First, what occasion are you dressing for?",
                    "value": [],
                    "needs_info": true,
                    "context": {
                        "understood": ["needs outfit recommendation"],
                        "missing": ["occasion", "style preferences"],
                        "question": "What occasion are you dressing for?"
                    }
                }

                Example - When Providing Recommendations:
                User: "I need an outfit for a dinner date at a nice restaurant"
                Assistant's Thought Process:
                1. Have key information:
                - Occasion: Formal dinner date
                - Setting: Upscale restaurant
                2. Can provide complete recommendation
                Response: {
                    "text": "I've put together a sophisticated dinner date look...",
                    "value": [
                        {
                            "category": "category name",
                            "product_id": "product id will be here",
                            "styling_tips": ["suggestion tip 1", "suggestion tip 2"]
                        }
                    ],
                    "recommendations": {
                        "category_suggestions": {
                            "categroy 1": "suggestion for category 1",
                            "category 2": "suggestion for category 2",
                        }
                    }
                }""",
                model="gpt-4o-mini",
                tools=[],
            )
            return assistant
        except Exception as e:
            print(f"Error creating assistant: {str(e)}")
            raise

    def should_ask_question(self, state: State) -> bool:
        """Check if we need to ask for more information"""
        try:
            if state.get("response"):
                response = json.loads(state.get("response"))
                return response.get("needs_info", False)
            return False
        except:
            return False

    @traceable
    async def handle_question(self, state: State) -> Dict[str, Any]:
        """Handle cases where we need to ask for more information"""
        try:
            # Just return the question response
            return {
                "messages": state.get("messages", []),
                "response": state.get("response")  # This contains the question format
            }
        except Exception as e:
            print(f"Error in handle_question: {str(e)}")
            error_response = {
                "text": "I apologize, but I encountered an error processing your request.",
                "value": [],
                "needs_info": False
            }
            return {
                "messages": state.get("messages", []),
                "response": json.dumps(error_response)
            }
    def _get_thread_key(self, unique_id, stylist_id):
        return f"{unique_id}_{stylist_id}"

    def _cleanup_old_threads(self, max_age_hours=24):
        current_time = time.time()
        threads_to_remove = []
        
        for thread_key, last_time in self.last_interaction.items():
            if current_time - last_time > max_age_hours * 3600:
                threads_to_remove.append(thread_key)
        
        for thread_key in threads_to_remove:
            del self.user_threads[thread_key]
            del self.last_interaction[thread_key]

    def _get_or_create_thread(self, unique_id, stylist_id):
        thread_key = self._get_thread_key(unique_id, stylist_id)
        current_time = time.time()
        
        # Initialize context with only messages
        if thread_key not in self.conversation_context:
            self.conversation_context[thread_key] = []  # Just a simple list for messages
        
        if thread_key in self.user_threads:
            self.last_interaction[thread_key] = current_time
            return self.user_threads[thread_key]
        
        thread = self.client.beta.threads.create()
        self.user_threads[thread_key] = thread.id
        self.last_interaction[thread_key] = current_time
        
        self._cleanup_old_threads()
        return thread.id

    def reset_conversation(self, unique_id: str, stylist_id: str):
        thread_key = self._get_thread_key(unique_id, stylist_id)
        
        if thread_key in self.user_threads:
            del self.user_threads[thread_key]
        if thread_key in self.last_interaction:
            del self.last_interaction[thread_key]
        if thread_key in self.conversation_context:
            self.conversation_context[thread_key] = []  # Reset to empty list
                
        thread = client.beta.threads.create()
        self.user_threads[thread_key] = thread.id
        self.last_interaction[thread_key] = time.time()
        
        return "Hello! I'm your refreshed stylist. How can I help you today?"
    @traceable
    async def format_product_response(self, state: State) -> Dict[str, Any]:
        """Format the product recommendations response"""
        try:
            recommendations = state.get("product_recommendations", {})
            stylist_name = state["stylist_id"].capitalize()
            
            # Extract data from recommendations
            category_suggestions = recommendations.get("category_suggestions", {})
            final_recommendation = json.loads(recommendations.get("final_recommendation", "{}"))
            products = recommendations.get("products", [])
            
            # Create formatted response
            formatted_response = {
                "text": final_recommendation.get("query_fit", f"Here are {stylist_name}'s recommendations for your outfit:"),
                "value": final_recommendation.get("items", []),
                "recommendations": {
                    "category_suggestions": category_suggestions,
                    "products": []
                }
            }
            
            # Format product details
            for product in products:
                formatted_product = {
                    "category": product["category"],
                    "product_id": product["product_id"],
                    "product_text": product["product_text"],
                    "retailer": product["retailer"],
                    "similarity_score": product["similarity_score"],
                    "image_urls": product.get("image_urls", []),
                    "url": product.get("url", ""),  # Product URL
                    "price": product.get("price", ""),  # Product price
                    "brand": product.get("brand", ""),  # Product brand
                    "suggestion": category_suggestions.get(product["category"], ""),
                    "gender": product.get("gender", "")
                }
                
                # Add styling tips from final recommendation if available
                for item in final_recommendation.get("items", []):
                    if item["product_id"] == product["product_id"]:
                        formatted_product["styling_tips"] = item.get("styling_tips", [])
                        break
                
                formatted_response["recommendations"]["products"].append(formatted_product)
            
            # print(formatted_response, "formatted_response printing")
            return {
                "messages": state["messages"],
                "response": json.dumps(formatted_response)
            }
            
        except Exception as e:
            print(f"Error formatting product response: {str(e)}")
            error_response = {
                "text": f"Error formatting recommendations: {str(e)}",
                "value": [],
                "recommendations": recommendations  # Return raw recommendations on error
            }
            return {
                "messages": state["messages"],
                "response": json.dumps(error_response)
            }
    @traceable
    async def process_user_input(self, state: State) -> Dict[str, Any]:
        thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
        thread_key = self._get_thread_key(state["unique_id"], state["stylist_id"])
        
        # Initialize if needed
        if thread_key not in self.conversation_context:
            self.conversation_context[thread_key] = []
        # conversation_messages = self.conversation_context[thread_key]
    
        # Create the system and user message
        context_message = f"""
        Previous Conversation:
        {chr(10).join(self.conversation_context[thread_key]) if self.conversation_context[thread_key] else "No previous conversation"}

        Current User Query: {state['user_query']}

        Follow your chain of thought process and respond in EXACTLY this format:
        {{
            "text": "your message or question",
            "value": [],
            "needs_info": true/false,
            "context": {{
                "understood": ["what you know"],
                "missing": ["what you need"],
                "question": "your question"
            }}
        }}
        """
        
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=[{"type": "text", "text": context_message}]
        )
        
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
            await asyncio.sleep(1)
        
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        response_text = messages.data[0].content[0].text.value
        

        print(response_text, "response_text printingggggggggggggggggggggggggggggg")
        try:
            json_start = response_text.find('{"text":')
            if json_start == -1:
                json_start = response_text.find('{')
                
            json_text = response_text[json_start:]
            response = json.loads(json_text)
            
            # Store the conversation flow
            self.conversation_context[thread_key].append(f"User: {state['user_query']}")
            self.conversation_context[thread_key].append(f"Assistant: {response['text']}")
            
            # Keep only last N messages for context
            if len(self.conversation_context[thread_key]) > self.max_history * 2:  # *2 because each turn has user + assistant
                self.conversation_context[thread_key] = self.conversation_context[thread_key][-self.max_history * 2:]
            
            return {
                "messages": state.get("messages", []),
                "response": json.dumps(response),
                "user_query": state['user_query']
            }
                    
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {response_text}")
            print(f"JSON error: {str(e)}")
            error_response = {
                "text": "I apologize, but I encountered an error processing your request.",
                "value": [],
                "needs_info": False
            }
            return {
                "messages": state.get("messages", []),
                "response": json.dumps(error_response),
                "user_query": state['user_query']
            }

    @traceable
    async def check_recommendation_need(self, state: State) -> Dict[str, Any]:
        """Determine if external product recommendations are needed"""
        try:
            # First check if wardrobe is empty
            if not state["wardrobe_data"]:
                return {
                    **state,
                    "needs_recommendations": True,
                    "shopping_analysis": {
                        "needs_shopping": True,
                        "confidence": 1.0,
                        "reasoning": "No items in wardrobe, recommendations needed",
                        "categories": []
                    }
                }
            # Get or create thread
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])

            system_prompt = """You are a shopping intent analyzer. Determine if the user's query indicates 
            interest in purchasing or shopping for new items. Consider both explicit mentions and implicit intent.
            Return ONLY a JSON with this exact structure:
            {
                "needs_shopping": boolean,
                "confidence": float,
                "reasoning": "brief explanation",
                "categories": ["category1", "category2"]
            }
            """

            wardrobe_info = "\n".join([f"- {item['caption']}" for item in state["wardrobe_data"]])
            user_context = f"""User Query: {state["user_query"]}
            
            Their Current Wardrobe Items:
            {wardrobe_info}
            
            Analyze if they need new items or can be styled with existing wardrobe.
            If the wardrobe lacks essential items to create a complete outfit, indicate that shopping is needed."""

            # Create message in thread
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": system_prompt + "\n\n" + user_context}]
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
                await asyncio.sleep(1)
            
            messages = self.client.beta.threads.messages.list(thread_id=thread_id)
            analysis = json.loads(messages.data[0].content[0].text.value)
            needs_recommendations = analysis.get("needs_shopping", False)

            # If wardrobe response exists but has no items referenced, trigger recommendations
            if state.get("wardrobe_response") and not state["wardrobe_response"].get("value"):
                needs_recommendations = True
                analysis["needs_shopping"] = True
                analysis["reasoning"] += " (No suitable items found in wardrobe)"
            
            return {
                **state,
                "needs_recommendations": needs_recommendations,
                "shopping_analysis": analysis
            }
                
        except Exception as e:
            print(f"Error in shopping intent analysis: {str(e)}")
            return {
                **state,
                "needs_recommendations": True,
                "shopping_analysis": {
                    "needs_shopping": True,
                    "confidence": 0.5,
                    "reasoning": "Fallback due to analysis error",
                    "categories": []
                }
            }

    def should_get_recommendations(self, state: State) -> bool:
        """Conditional edge handler for recommendation flow"""
        return state.get("needs_recommendations", False)

    @traceable
    async def generate_wardrobe_response(self, state: State) -> Dict[str, Any]:
        """Generate initial response using only wardrobe items"""
        try:
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
            
            # Format wardrobe items with clear IDs
            wardrobe_info = "\n".join(
                f"ID: {item['token_name']} | {item['caption']}" 
                for item in state["wardrobe_data"]
            )
            
            # Create focused context message
            context_message = f"""You are {state['stylist_id'].capitalize()}. Maintain your unique styling perspective.

                            Available Wardrobe Items:
                            {wardrobe_info}

                            Target Item ID: {state['image_id']}
                            User Query: {state['user_query']}

                            CRITICAL INSTRUCTION:
                            Respond with ONLY a JSON string in this exact format:
                            {{
                                "text": "<your complete styling advice>",
                                "value": ["token_name1", "token_name2"]
                            }}

                            - The text field should contain your complete styling advice
                            - The value array MUST contain the token_names of items you referenced
                            - Do not include wardrobe item token names in the styling advice
                            - Include ALL wardrobe items you mentioned in your advice in the value array
                            - Do not add any text outside the JSON structure
                            """
            
            # Get styling advice
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": context_message}]
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                await asyncio.sleep(1)
            
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0].content[0].text.value
            
            try:
                # Ensure we're parsing only the JSON part
                json_str = latest_message.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:-3]  # Remove ```json and ``` markers
                
                wardrobe_response = json.loads(json_str)
                
                # Validate response format
                if not isinstance(wardrobe_response, dict):
                    raise ValueError("Response must be a dictionary")
                if "text" not in wardrobe_response or "value" not in wardrobe_response:
                    raise ValueError("Response missing required fields")
                if not isinstance(wardrobe_response["value"], list):
                    wardrobe_response["value"] = []
                
                # Add response to state
                state["response"] = json.dumps(wardrobe_response)
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Invalid response format: {str(e)}\nResponse: {latest_message}")
                wardrobe_response = {
                    "text": "I apologize, but I couldn't generate a proper response. Please try again.",
                    "value": []
                }
                state["response"] = json.dumps(wardrobe_response)
            
            return {
                **state,
                "wardrobe_response": wardrobe_response
            }
            
        except Exception as e:
            print(f"Error in generate_wardrobe_response: {str(e)}")
            error_response = {
                "text": f"Error generating wardrobe advice: {str(e)}",
                "value": []
            }
            state["response"] = json.dumps(error_response)
            return {
                **state,
                "wardrobe_response": error_response
            }
    @traceable
    async def get_product_recommendations(self, state: State) -> Dict[str, Any]:
        """Get product recommendations if needed"""
        try:
            if not state.get("needs_recommendations", False):
                return state

            enhanced_query = state["user_query"]
            if state["image_id"] and state["image_id"] != "general_chat":
                target_item = next(
                    (item for item in state["wardrobe_data"] if item["token_name"] == state["image_id"]),
                    None
                )
                if target_item:
                    enhanced_query += f"\nPlease consider coordinating with: {target_item['caption']}"

            recommendations = await self.product_recommender.process_query(
                user_id=state["unique_id"],
                stylist_id=state["stylist_id"],
                query=enhanced_query
            )
            
            # print(recommendations, "recommendations printing")
            # exit(-1)
            return {
                **state,
                "product_recommendations": recommendations
            }
            
        except Exception as e:
            print(f"Error getting product recommendations: {str(e)}")
            return {
                **state,
                "product_recommendations": {"error": str(e)}
            }

    @traceable
    async def generate_final_response(self, state: State) -> Dict[str, Any]:
        """Generate final response incorporating both wardrobe and product recommendations"""
        try:
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
            
            # Combine wardrobe response and product recommendations
            base_content = (
                f"Based on the wardrobe styling: {state.get('wardrobe_response', {}).get('text', '')}\n\n"
                f"User Query: {state['user_query']}\n"
            )
            
            if state.get("product_recommendations"):
                recs = state["product_recommendations"]
                if "products" in recs and not "error" in recs:
                    product_content = "\nRecommended Products to Complete the Look:\n"
                    for product in recs.get("products", []):
                        product_content += (
f"\n- {product['category'].upper()}:"
                            f"\n  {product['product_text']}"
                            f"\n  Retailer: {product['retailer']}"
                            f"\n  Product ID: {product['product_id']}"
                            f"\n  Image URLs Available: {len(product.get('image_urls', []))}"
                        )
                    base_content += product_content
            
            # Get final styling advice
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": base_content}]
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id
            )
            
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                await asyncio.sleep(1)
            
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0].content[0].text.value
            
            try:
                final_response = json.loads(latest_message)
                # Add recommendations and wardrobe response to final response
                final_response["wardrobe_styling"] = state.get("wardrobe_response", {})
                if state.get("product_recommendations"):
                    final_response["recommendations"] = state["product_recommendations"]
                if state.get("shopping_analysis"):
                    final_response["shopping_analysis"] = state["shopping_analysis"]
                
            except json.JSONDecodeError:
                final_response = {
                    "text": latest_message,
                    "value": [],
                    "wardrobe_styling": state.get("wardrobe_response", {}),
                    "recommendations": state.get("product_recommendations", {}),
                    "shopping_analysis": state.get("shopping_analysis", {})
                }
            
            return {
                "messages": state["messages"] + [json.dumps(final_response)],
                "response": json.dumps(final_response)
            }
            
        except Exception as e:
            print(f"Error in generate_final_response: {str(e)}")
            error_response = {
                "text": f"An error occurred: {str(e)}",
                "value": [],
                "wardrobe_styling": state.get("wardrobe_response", {}),
                "recommendations": state.get("product_recommendations", {}),
                "shopping_analysis": state.get("shopping_analysis", {})
            }
            return {
                "messages": state["messages"] + [json.dumps(error_response)],
                "response": json.dumps(error_response)
            }

    def should_end(self, state: State) -> str:
        """Determine if the workflow should end"""
        if state.get("response"):
            return END
        return "generate_final_response"

def get_or_create_assistant(stylist_id: str) -> FashionAssistant:
    """Get existing assistant instance or create new one"""
    if stylist_id not in _fashion_assistant_instances:
        _fashion_assistant_instances[stylist_id] = FashionAssistant(stylist_id=stylist_id)
    return _fashion_assistant_instances[stylist_id]

async def chat_with_stylist(
    query: str,
    unique_id: str,
    stylist_id: str,
    image_id: str,
    wardrobe_data: List[Dict[str, str]]
) -> str:
    """Main function to interact with the fashion assistant"""
    fashion_assistant = get_or_create_assistant(stylist_id)
    
    initial_state = {
        "messages": [],
        "user_query": query,
        "unique_id": unique_id,
        "stylist_id": stylist_id,
        "image_id": image_id,
        "wardrobe_data": wardrobe_data,
        "response": "",
        "needs_recommendations": False,
        "product_recommendations": None,
        "shopping_analysis": None,
        "wardrobe_response": None
    }
    
    config = {"recursion_limit": 10}
    result = await fashion_assistant.graph.ainvoke(initial_state, config=config)
    
    return result["response"]

if __name__ == "__main__":
    async def main():
        query = "I need a new outfit for a summer wedding, give something from outside the wardrobe."
        unique_id = "9ac36b7d-ada8-4fc5-8c08-976731903d3c"
        stylist_id = "reginald"
        image_id = "item4"
        wardrobe_data = [
            {"token_name": "item1", "caption": "Blue denim jeans"},
            {"token_name": "item2", "caption": "White cotton t-shirt"},
            {"token_name": "item3", "caption": "Brown leather jacket"},
            {"token_name": "item4", "caption": "Black dress shoes"},
        ]
        
        try:
            response = await chat_with_stylist(
                query=query,
                unique_id=unique_id,
                stylist_id=stylist_id,
                image_id=image_id,
                wardrobe_data=wardrobe_data
            )
            response_dict = json.loads(response)
            
            # First, print raw JSON structure
            print(f"\n{'='*50}")
            print("RAW RESPONSE")
            print(f"{'='*50}")
            print(json.dumps(response_dict, indent=2))
            
            # Then print detailed recommendations
            print(f"\n{'='*50}")
            print("DETAILED RECOMMENDATIONS")
            print(f"{'='*50}")
            
            # Print styling advice
            print("\nStyling Overview:")
            print(response_dict['text'])
            
            # Print referenced wardrobe items
            print("\nReferenced Wardrobe Items:")
            for item_id in response_dict.get('value', []):
                item = next((i for i in wardrobe_data if i['token_name'] == item_id), None)
                if item:
                    print(f"- {item['caption']} (ID: {item_id})")
                else:
                    print(f"- Unknown item (ID: {item_id})")
            
            # Print recommendations if available
            if "recommendations" in response_dict:
                recs = response_dict["recommendations"]
                
                # Print category suggestions
                print(f"\n{'='*50}")
                print("PRODUCT SUGGESTIONS")
                print(f"{'='*50}")
                
                if "category_suggestions" in recs:
                    for category, suggestion in recs["category_suggestions"].items():
                        print(f"\n{category.upper()}:")
                        print(f"Suggestion: {suggestion}")
                
                # Print product details
                print(f"\n{'='*50}")
                print("PRODUCT DETAILS")
                print(f"{'='*50}")
                
                for product in recs.get("products", []):
                    print(f"\n{product['category'].upper()}")
                    print("-" * 30)
                    print(f"Product: {product['product_text']}")
                    print(f"Retailer: {product['retailer']}")
                    print(f"brand: {product['brand']}")
                    print(f"Product ID: {product['product_id']}")
                    print(f"Product URL: {product.get('url', 'Not available')}")
                    print(f"Match Score: {product['similarity_score']:.2f}")
                    if 'styling_tips' in product:
                        print("\nStyling Tips:")
                        for tip in product['styling_tips']:
                            print(f"- {tip}")
                    if product.get('image_urls'):
                        print("\nImage URLs:")
                        for url in product['image_urls']:
                            print(f"- {url}")
                    
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise
    
    asyncio.run(main())