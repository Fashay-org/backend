from typing import Dict, List, Tuple, Any, TypedDict, Annotated
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

class FashionAssistant:
    def __init__(self, stylist_id="reginald"):
        self.current_stylist_id = stylist_id.lower()
        self.user_threads = {}
        self.last_interaction = {}
        self.max_history = 10
        self.stylist_personalities = {
            "reginald": """I am Reginald, a men's fashion expert with a keen eye for sophisticated yet practical styling. 
            I specialize in creating versatile looks that combine classic elements with modern trends. 
            My approach focuses on helping men build confidence through well-coordinated outfits that suit their lifestyle and personality.
            I'm direct and professional, but also warm and encouraging.""",
            
            "eliza": """I am Eliza, a fashion curator with an eye for elevated, sophisticated style. 
            I excel at creating polished looks that seamlessly blend timeless elegance with contemporary fashion. 
            My expertise lies in helping clients develop a refined wardrobe that expresses their personal style while maintaining versatility.
            I'm graceful and articulate, with a passion for helping others discover their best look.""",
            
            "lilia": """I am Lilia, a body-positive fashion stylist who celebrates individual beauty. 
            I specialize in creating flattering looks that make people feel confident and comfortable. 
            My approach focuses on understanding each person's unique style journey and helping them embrace their body shape.
            I'm warm, supportive, and enthusiastic about helping others find their perfect style."""
        }        
        # Initialize the graph
        self.workflow = StateGraph(State)
        
        # Add nodes
        self.workflow.add_node("process_input", self.process_user_input)
        self.workflow.add_node("generate_response", self.generate_response)
        
        # Add edges
        self.workflow.add_edge(START, "process_input")
        self.workflow.add_edge("process_input", "generate_response")
        
        # Add conditional edges
        self.workflow.add_conditional_edges(
            "generate_response",
            self.should_end,
            [END]
        )
        
        # Compile the graph
        self.graph = self.workflow.compile()
        self.assistant = self._create_assistant()
    def _create_assistant(self):
        # Get the stylist's personality description
        stylist_personality = self.stylist_personalities.get(
            self.current_stylist_id,
            "I am your personal fashion stylist, focused on helping you create stylish and confident looks."
        )

        assistant = client.beta.assistants.create(
            name="Fashion Stylist",
            instructions=f"""You are an expert fashion stylist assistant specializing in providing personalized styling advice. 
            
            {stylist_personality}
            
            Your recommendations should be specific, practical, and contextually appropriate. Consider the wardrobe items provided 
            and maintain your unique perspective as a fashion stylist. Stay consistent with your personality and previous advice.
            
            IMPORTANT: Format all your responses as a valid JSON string with this exact structure:
            {{
                "text": "<your styling advice along with recommendations. No points.>",
                "value": []
            }}
            
            Always ensure your response is proper JSON and includes both the text and value fields.
            Remember previous interactions with the user and maintain continuity in your advice.""",
            model="gpt-4o-mini",
            tools=[]
        )
        return assistant

    def _get_thread_key(self, unique_id, stylist_id):
        return f"{unique_id}_{stylist_id}"

    def _cleanup_old_threads(self, max_age_hours=24):
        """Clean up threads that haven't been used for a while."""
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
        
        # Check if thread exists and when it was last used
        if thread_key in self.user_threads:
            self.last_interaction[thread_key] = current_time
            return self.user_threads[thread_key]
        
        # Create new thread
        thread = client.beta.threads.create()
        self.user_threads[thread_key] = thread.id
        self.last_interaction[thread_key] = current_time
        
        # Cleanup old threads
        self._cleanup_old_threads()
        
        return thread.id
    def reset_conversation(self, unique_id: str, stylist_id: str):
        """Reset the conversation for a specific user and stylist."""
        thread_key = self._get_thread_key(unique_id, stylist_id)
        
        # Remove existing thread if it exists
        if thread_key in self.user_threads:
            del self.user_threads[thread_key]
        if thread_key in self.last_interaction:
            del self.last_interaction[thread_key]
        
        # Create new thread with introduction
        thread = client.beta.threads.create()
        self.user_threads[thread_key] = thread.id
        self.last_interaction[thread_key] = time.time()
        
        return "Hello! I'm your refreshed stylist. How can I help you today?"

    @traceable
    async def process_user_input(self, state: State) -> Dict[str, Any]:
        print(f"[DEBUG] Processing input for stylist: {state['stylist_id']}")
        print(f"[DEBUG] Available personalities: {list(self.stylist_personalities.keys())}")
        # Get stylist personality for context
        stylist_context = self.stylist_personalities.get(
            state["stylist_id"].lower(),
            "I am your personal fashion stylist, focused on helping you create stylish and confident looks."
        )

        wardrobe_items = "\n".join(
            f"- {item['caption']} (ID: {item['token_name']})" 
            for item in state["wardrobe_data"]
        )
        
        if state["image_id"] and state["image_id"] != "general_chat":
            target_item = next(
                (item for item in state["wardrobe_data"] if item["token_name"] == state["image_id"]),
                None
            )
            
            target_context = f"""Target Item: {target_item['caption'] if target_item else 'Unknown item'} (ID: {state['image_id']})
            Please provide recommendations that:
            1. Complement this target item
            2. Create cohesive outfits using both the target item and available wardrobe items
            3. Include the target item ID in your recommendations' value array"""
        else:
            target_context = "No specific target item. Please provide recommendations based on the available wardrobe items."

        enhanced_query = f"""STYLIST CONTEXT:
        {stylist_context}

        SITUATION CONTEXT:
        {target_context}

        Available Wardrobe Items:
        {wardrobe_items}

        Instructions:
        1. When recommending items, include their IDs in the 'value' array
        2. Every item mentioned in your recommendation must have its ID in the value array
        3. Maintain your styling perspective and personality throughout the response
        4. Reference previous interactions when appropriate
        5. Stay consistent with your stylist character and previous advice

        User Query: {state['user_query']}"""
        
        return {
            "messages": state.get("messages", []) + [enhanced_query],
            "user_query": enhanced_query
        }

    @traceable
    async def generate_response(self, state: State) -> Dict[str, Any]:
        try:
            thread_id = self._get_or_create_thread(state["unique_id"], state["stylist_id"])
            
            # Create message in thread with properly formatted content
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[{
                    "type": "text",
                    "text": str(state["user_query"])
                }]
            )
            
            # Run the assistant
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
            
            # Get the latest message
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0].content[0].text.value
            
            try:
                response_json = json.loads(latest_message)
                if not isinstance(response_json, dict) or 'text' not in response_json or 'value' not in response_json:
                    response_json = {"text": latest_message, "value": []}
                if not isinstance(response_json['value'], list):
                    response_json['value'] = []
            except json.JSONDecodeError:
                response_json = {"text": latest_message, "value": []}
            
            return {
                "messages": state["messages"] + [json.dumps(response_json)],
                "response": json.dumps(response_json)
            }
        except Exception as e:
            error_response = {"text": f"An error occurred: {str(e)}", "value": []}
            return {
                "messages": state["messages"] + [json.dumps(error_response)],
                "response": json.dumps(error_response)
            }

    def should_end(self, state: State) -> str:
        """Determine if we should end or continue processing."""
        if state.get("response"):
            return END
        return "generate_response"

def get_or_create_assistant(stylist_id: str) -> FashionAssistant:
    """Get existing assistant instance or create new one."""
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
    fashion_assistant = get_or_create_assistant(stylist_id)
    
    initial_state = {
        "messages": [],
        "user_query": query,
        "unique_id": unique_id,
        "stylist_id": stylist_id,
        "image_id": image_id,
        "wardrobe_data": wardrobe_data,
        "response": ""
    }
    
    config = {"recursion_limit": 10}
    result = await fashion_assistant.graph.ainvoke(initial_state, config=config)
    
    return result["response"]

if __name__ == "__main__":
    async def main():
        query = "What outfit can I wear for a casual summer day?"
        unique_id = "user123"
        stylist_id = "reginald"
        image_id = "item4"
        wardrobe_data = [
            {"token_name": "item1", "caption": "Blue denim jeans"},
            {"token_name": "item2", "caption": "White cotton t-shirt"},            
            {"token_name": "item3", "caption": "brown furry coat"},
            {"token_name": "item4", "caption": "White hoodie with pink shades"},
            {"token_name": "item5", "caption": "Black coat"},
            {"token_name": "item6", "caption": "raincoat"},
        ]
        
        try:
            response = await chat_with_stylist(
                query=query,
                unique_id=unique_id,
                stylist_id=stylist_id,
                image_id=image_id,
                wardrobe_data=wardrobe_data
            )
            print(f"User Query: {query}")
            response_dict = json.loads(response)
            print(f"Stylist {stylist_id}'s Response: {response_dict['text']}")
            print(f"Values: {response_dict['value']}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise
    
    asyncio.run(main())