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
    response_type: str

class FashionAssistant:
    def __init__(self, stylist_id="reginald"):
            self.current_stylist_id = stylist_id.lower()
            self.user_threads = {}
            self.conversation_context = {}
            self.last_interaction = {}
            self.max_history = 25
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
            self.workflow.add_edge("format_product_response", END)
            # Add final edges
            # self.workflow.add_conditional_edges(
            #     "format_product_response",
            #     self.should_end,
            #     [END]
            # )
            
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
                You are a dedicated fashion and style advisor. Your sole purpose is to help users discover 
                their unique personal style through thoughtful questioning and provide tailored 
                outfit recommendations. You will not engage with non-fashion related queries. Follow this process:

                THOUGHT PROCESS:
                1. Begin with foundational analysis:
                <contemplator>
                - What is the core request?
                - What style information do we already have?
                - What critical details are missing?
                </contemplator>
                
                2. Style Profile Building:
                a. Basic Style (Question each element):
                - Desired style words
                - Color preferences
                - Style inspiration
                - Key pieces
                - Boundaries

                3. Response Types:
                You must respond with one of three modes using the needs_info field:
                - "ask": When you need more information from the user
                - "speak": When you just need to communicate something without recommendations
                - "suggest": When you're ready to provide product suggestions

                4. Response Format:
                For "ask" mode (when you need to know about the user style profile, ask only one question at a time):
                {
                    "text": "Your question to the user must be present here", // ask only 1 question at a time
                    "value": [], // Referenced items
                    "needs_info": "ask",
                    "context": {
                        "understood": ["what you know"],
                        "missing": ["what you need"],
                        "question": "your question if asking"
                    }
                }
                For "speak" mode (include all explanations in the text field):
                {
                    "text":  "Here is your message to the user, explain everything in detail. No need to ask questions or give recommendations.",
                    "value": [], // Referenced items
                    "needs_info": "speak",
                    "context": {
                        "understood": ["what you know"],
                        "missing": [],
                        "question": "" 
                    },
                }

                For "suggest" mode (when you are recommending something or giving suggestions):
                {
                    "text": "Your styling advice and explanation", 
                    "value": [
                    token name 1,
                    token name 2
                    ], //for recommendations from within the wardrobe.
                    "needs_info": "suggest",
                    "context": {
                        "understood": ["what you know"],
                        "missing": ["what you need"],
                        "question": "your question if asking" // Empty string if not asking
                    },
                    "recommendations": {  // Only include when needs_info = "suggest"
                        "category_suggestions": {
                            "category 1": "detailed category 1 description",
                            "category 2": "detailed category 2 description",
                            "category 3": "detailed category 3 description",
                            "category 4": "detailed category 4 description",
                            "category 5": "detailed category 5 description",
                        },  
                        "products": [
                            // Product details will be filled by the system
                        ]
                        }
                    }
                }

                Examples:

                1. Asking for Information:
                {
                    "text": "Questions that help understand user style preferences like what are your stylistic inspirations?", //don't use the same question
                    "value": [],
                    "needs_info": "ask",
                    "context": {
                        "understood": ["needs outfit recommendation"],
                        "missing": ["occasion", "style preferences"],
                        "question": "what are your stylistic inspirations?"
                    }
                }

                2. Simple Communication:
                {
                    "text": "The reason for choosing the white blouse is that it is simple and elegant.",
                    "value": [],
                    "needs_info": "speak",
                    "context": {
                        "understood": ["wants casual outfit"],
                        "missing": [],
                        "question": ""
                    }
                }

                3. Providing Recommendations:
                {
                    "text": "Here's a sophisticated outfit for your business meeting",
                    "value": [
                    token name 1,
                    token name 2
                    ],
                    "needs_info": "suggest",
                    "context": {
                        "understood": ["business meeting", "formal setting"],
                        "missing": [],
                        "question": ""
                    },
                    "recommendations": {
                        "category_suggestions": {
                            "category 1": "suggestion for category 1",
                            "category 2": "suggestion for category 2",
                        }
                    }
                }
                """,
                model="gpt-4o-mini",
                tools=[],
            )
            return assistant
        except Exception as e:
            print(f"Error creating assistant: {str(e)}")
            error_response = {
                "text": f"I apologize, but I encountered an error initializing the fashion assistant: {str(e)}",
                "value": [],
                "needs_info": "speak",  # Changed from False
                "context": {
                    "error": str(e),
                    "understood": ["assistant creation attempted"],
                    "missing": [],
                    "question": ""
                }
            }
            raise Exception(json.dumps(error_response))

    def should_ask_question(self, state: State) -> bool:
        """Check if we need to ask for more information"""
        try:
            if state.get("response"):
                response = json.loads(state.get("response"))
                return response.get("needs_info") == "ask"
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
            response = json.loads(state.get("response", "{}"))
            recommendations = state.get("product_recommendations", {}) or {}
            
            # Handle speak/ask modes
            if response.get("needs_info") in ["speak", "ask"]:
                return {
                    "messages": state["messages"],
                    "response": json.dumps({
                        "text": response.get("text", ""),
                        "value": response.get("value", []),  # Preserve value array for wardrobe items
                        "needs_info": response.get("needs_info"),
                        "context": response.get("context", {})
                    })
                }
            
            # Handle suggest mode
            if response.get("needs_info") == "suggest":
                # Get category suggestions
                category_suggestions = {}
                if isinstance(recommendations, dict):
                    category_suggestions = recommendations.get("category_suggestions", {})
                if not category_suggestions and response.get("recommendations"):
                    category_suggestions = response["recommendations"].get("category_suggestions", {})
                
                # Get value items (wardrobe references) from response
                value_items = []
                if response.get("value"):
                    for item in response["value"]:
                        if isinstance(item, dict):
                            # Handle dictionary format
                            value_items.append({
                                "category": item.get("category", ""),
                                "product_id": item.get("product_id", ""),
                                "styling_tips": item.get("styling_tips", [])
                            })
                        elif isinstance(item, str):
                            # Handle string ID format
                            value_items.append(item)
                
                # Create formatted response
                formatted_response = {
                    "text": response.get("text", ""),
                    "value": value_items,  # Include wardrobe item references
                    "needs_info": "suggest",
                    "context": response.get("context", {}),
                    "recommendations": {
                        "category_suggestions": category_suggestions,
                        "products": []
                    }
                }
                
                # Only process products if they exist
                if isinstance(recommendations, dict) and recommendations.get("products"):
                    for product in recommendations["products"]:
                        if not isinstance(product, dict):
                            continue
                        
                        formatted_product = {
                            "category": product.get("category", ""),
                            "product_id": product.get("product_id", ""),
                            "product_text": product.get("product_text", ""),
                            "retailer": product.get("retailer", ""),
                            "similarity_score": product.get("similarity_score", 0.0),
                            "image_urls": product.get("image_urls", []),
                            "url": product.get("url", ""),
                            "price": product.get("price", ""),
                            "brand": product.get("brand", ""),
                            "name": product.get("name", ""),
                            "description": product.get("description", ""),
                            "suggestion": category_suggestions.get(product.get("category", ""), ""),
                            "gender": product.get("gender", ""),
                            "styling_tips": []
                        }
                        
                        formatted_response["recommendations"]["products"].append(formatted_product)
                
                return {
                    "messages": state["messages"],
                    "response": json.dumps(formatted_response)
                }
                
        except Exception as e:
            print(f"Error formatting product response: {str(e)}")
            error_response = {
                "text": f"I apologize, but I encountered an error formatting the recommendations: {str(e)}",
                "value": [],
                "needs_info": "speak",
                "context": {
                    "error": str(e),
                    "understood": ["formatting attempted"],
                    "missing": [],
                    "question": ""
                }
            }
            return {
                "messages": state["messages"],
                "response": json.dumps(error_response)
            }
    @traceable
    async def process_user_input(self, state: State) -> Dict[str, Any]:
        try:
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
            thread_key = self._get_thread_key(state["unique_id"], state["stylist_id"])
            
            # Initialize if needed
            if thread_key not in self.conversation_context:
                self.conversation_context[thread_key] = []
        
            # Create the system and user message
            context_message = f"""
            Previous Conversation:
            {chr(10).join(self.conversation_context[thread_key]) if self.conversation_context[thread_key] else "No previous conversation"}

            Current User Query: {state['user_query']}

            Follow your chain of thought process. Begin with foundational analysis and style profile building.
            Return a clean JSON response without any comments.
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

            # Clean up the response text
            def clean_json_string(text: str) -> str:
                # Find the first opening brace
                start_idx = text.find('{')
                if start_idx == -1:
                    raise ValueError("No JSON object found in response")
                    
                # Find the matching closing brace
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            return text[start_idx:i+1]
                
                raise ValueError("No complete JSON object found")

            try:
                cleaned_json = clean_json_string(response_text)
                response = json.loads(cleaned_json)
                
                # Store the conversation flow
                self.conversation_context[thread_key].append(f"User: {state['user_query']}")
                self.conversation_context[thread_key].append(f"Assistant: {response['text']}")
                
                # Keep only last N messages for context
                if len(self.conversation_context[thread_key]) > self.max_history * 2:
                    self.conversation_context[thread_key] = self.conversation_context[thread_key][-self.max_history * 2:]
                
                return {
                    "messages": state.get("messages", []),
                    "response": json.dumps(response),
                    "user_query": state['user_query']
                }
                        
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse response: {response_text}")
                print(f"JSON error: {str(e)}")
                error_response = {
                    "text": "I apologize, but I encountered an error processing your request.",
                    "value": [],
                    "needs_info": "speak",
                    "context": {
                        "error": str(e),
                        "understood": [],
                        "missing": [],
                        "question": ""
                    }
                }
                return {
                    "messages": state.get("messages", []),
                    "response": json.dumps(error_response),
                    "user_query": state['user_query']
                }
                
        except Exception as e:
            print(f"Error in process_user_input: {str(e)}")
            error_response = {
                "text": f"I apologize, but I encountered an error: {str(e)}",
                "value": [],
                "needs_info": "speak",
                "context": {
                    "error": str(e),
                    "understood": [],
                    "missing": [],
                    "question": ""
                }
            }
            return {
                "messages": state.get("messages", []),
                "response": json.dumps(error_response),
                "user_query": state['user_query']
            }
        # except json.JSONDecodeError as e:
        #     print(f"Failed to parse response: {response_text}")
        #     print(f"JSON error: {str(e)}")
        #     error_response = {
        #         "text": "I apologize, but I encountered an error processing your request.",
        #         "value": [],
        #         "needs_info": "speak",  # Changed from False
        #         "context": {
        #             "error": str(e),
        #             "understood": [],
        #             "missing": [],
        #             "question": ""
        #         }
        #     }
        #     return {
        #         "messages": state.get("messages", []),
        #         "response": json.dumps(error_response),
        #         "user_query": state['user_query']
        #     }


    @traceable
    async def check_recommendation_need(self, state: State) -> Dict[str, Any]:
        """Determine if external product recommendations are needed"""
        try:
            # Check if we have a direct response case (no recommendations needed)
            try:
                response = json.loads(state.get("response", "{}"))

                # If it's a speak mode response, don't get recommendations
                if response.get("needs_info") == "speak":
                    return {
                        **state,
                        "needs_recommendations": False,
                        "shopping_analysis": {
                            "needs_shopping": False,
                            "confidence": 1.0,
                            "reasoning": "Direct response, no recommendations needed",
                            "categories": []
                        }
                    }
            except (json.JSONDecodeError, KeyError):
                pass
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

            system_prompt = """You will be given user query and based on that determine if the user's query indicates 
            interest in getting recommendations from the wardrobe or recommendations for shopping for new items. Consider both explicit mentions and implicit intent.
            Return ONLY a JSON with this exact structure:
            {
                "needs_shopping": boolean, 
                "confidence": float,
                "reasoning": "brief explanation",
                "categories": ["category1", "category2"]
            }
            if user wants recommendations for shopping for new items, return "needs_shopping": true, otherwise return "needs_shopping": false.
            """

            wardrobe_info = "\n".join([f"- {item['caption']}" for item in state["wardrobe_data"]])
            user_context = f"""User Query: {state["user_query"]}
            
            Their Current Wardrobe Items:
            {wardrobe_info}
            
            Analyze if they need new items or can be styled with existing wardrobe.
            If the wardrobe lacks essential items to create a complete outfit, indicate that shopping is needed which means return "needs_shopping": true."""

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
            # print(analysis, "analysis printing", "check_recommendation_need")
            # If wardrobe response exists but has no items referenced, trigger recommendations
            if state.get("wardrobe_response") and not state["wardrobe_response"].get("value"):
                needs_recommendations = True
                analysis["needs_shopping"] = True
                analysis["reasoning"] += " (No suitable items found in wardrobe)"
            
            # print(state, "state printing", "check_recommendation_need")
            return {
                **state,
                "needs_recommendations": needs_recommendations,
                "shopping_analysis": analysis
            }
                
        except Exception as e:
            print(f"Error in shopping intent analysis: {str(e)}")
            error_response = {
                "text": f"I encountered an error analyzing shopping needs: {str(e)}",
                "value": [],
                "needs_info": "speak",  # Changed from True
                "context": {
                    "error": str(e),
                    "understood": ["error occurred"],
                    "missing": [],
                    "question": ""
                }
            }
            return {
                **state,
                "response": json.dumps(error_response),
                "needs_recommendations": False,
                "shopping_analysis": {
                    "needs_shopping": False,
                    "confidence": 0.0,
                    "reasoning": f"Error occurred: {str(e)}",
                    "categories": []
                }
            }

    def should_get_recommendations(self, state: State) -> bool:
        """Conditional edge handler for recommendation flow"""
        try:
            # print(state, "state printing")
            response = json.loads(state.get("response", "{}"))
            # Check if we need product suggestions
            if response.get("needs_info") == "suggest":
                return True
            # If it's speak mode or no specific needs_info, go to wardrobe response
            return False
        except (json.JSONDecodeError, KeyError):
            return False

    @traceable
    async def generate_wardrobe_response(self, state: State) -> Dict[str, Any]:
        """Generate response using wardrobe items"""
        print("generate_wardrobe_response")
        try:
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
            
            # Format wardrobe items
            wardrobe_info = "\n".join(
                f"ID: {item['token_name']} | {item['caption']}" 
                for item in state["wardrobe_data"]
            )
            
            # print(wardrobe_info, "wardrobe_info printing")
            context_message = f"""You are {state['stylist_id'].capitalize()}. 
            Available Wardrobe Items:
            {wardrobe_info}
            User Query: {state['user_query']}

            Respond with a JSON containing:
            {{
                "text": "your styling advice, no token names here in the text. Just the styling advice",
                "value": ["token_name1", "token_name2"],
                "needs_info": "suggest",
                "context": {{
                    "understood": ["what you understood"],
                    "missing": [],
                    "question": ""
                }},
                "recommendations": {{
                    "category_suggestions": {{
                        "category1": "detailed suggestion",
                        "category2": "detailed suggestion"
                    }}
                }}
            }}"""
            
            # Get styling advice from assistant
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{"type": "text", "text": context_message}]
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant.id
            )
            
            # Wait for completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                await asyncio.sleep(1)
            
            # Get response
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            wardrobe_response = json.loads(messages.data[0].content[0].text.value)
            
            # Ensure proper format
            print(wardrobe_response, "111111wardrobe_response printing")
            if "needs_info" not in wardrobe_response:
                wardrobe_response["needs_info"] = "suggest"
            if "recommendations" not in wardrobe_response:
                wardrobe_response["recommendations"] = {
                    "category_suggestions": {}
                }
            
            print(wardrobe_response, "22222wardrobe_response printing")
            return {
                **state,
                "wardrobe_response": wardrobe_response,
                "response": json.dumps(wardrobe_response)
            }
            
        except Exception as e:
            print(f"Error in generate_wardrobe_response: {str(e)}")
            error_response = {
                "text": f"I apologize, but I encountered an error generating wardrobe advice: {str(e)}",
                "value": [],
                "needs_info": "speak",  # Changed from False
                "context": {
                    "error": str(e),
                    "understood": ["wardrobe analysis attempted"],
                    "missing": [],
                    "question": ""
                }
            }
            return {
                **state,
                "wardrobe_response": error_response,
                "response": json.dumps(error_response)
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
            
            print(recommendations, "recommendations printing")
            # exit(-1)
            return {
                **state,
                "product_recommendations": recommendations
            }
            
        except Exception as e:
            print(f"Error getting product recommendations: {str(e)}")
            error_response = {
                "text": f"I encountered an error while finding product recommendations: {str(e)}",
                "value": [],
                "needs_info": "speak",  # Changed from False
                "context": {
                    "error": str(e),
                    "understood": ["product recommendation attempted"],
                    "missing": [],
                    "question": ""
                }
            }
            return {
                **state,
                "product_recommendations": None,
                "response": json.dumps(error_response)
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
                "text": f"I apologize, but I encountered an error generating the final response: {str(e)}",
                "value": [],
                "needs_info": "speak",  # Changed from False
                "context": {
                    "error": str(e),
                    "understood": ["final response generation attempted"],
                    "missing": [],
                    "question": ""
                },
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