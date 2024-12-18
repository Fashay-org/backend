from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
from langsmith.wrappers import wrap_openai
from langsmith import traceable

load_dotenv()
client = wrap_openai(OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY2"),
))

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fashay"



MODEL_NAME = "text-embedding-3-small"
COLLECTION_NAME = "Fashay"

class FashionAssistant:
    def __init__(self):
        """Initialize the Fashion Assistant with OpenAI Assistant"""
        self.assistant = self._create_assistant()
        # Dictionary to store user-stylist threads
        self.user_threads = {}
        # Maximum conversation history to maintain (in messages)
        self.max_history = 10
        # Qdrant Configuration
        self.qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
    @traceable(run_type="retriever")
    def get_embedding(self, text, model=MODEL_NAME):
        response = client.embeddings.create(input=[text.replace("\n", " ")], model=model)
        return response.data[0].embedding

    # Query embeddings with stylist filter
    @traceable(run_type="retriever")
    def query_embeddings(self, query, top_k=5, stylist_id=None):
        query_embedding = self.get_embedding(query)
        
        filter_condition = None
        if stylist_id:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="stylist_id",
                        match=MatchValue(value=stylist_id)
                    )
                ]
            )
        
        search_result = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_condition
        )
        return search_result        
    def _create_assistant(self):
        """Create or load the OpenAI Assistant"""
        assistant = client.beta.assistants.create(
            name="Fashion Stylist",
            instructions = """You are an expert fashion stylist assistant providing personalized styling advice 
            based on users' wardrobes and preferences. Your recommendations should be specific, practical, 
            and contextually appropriate.

            For new conversations:
            - Ask about style preferences (casual/formal/etc.)
            - Understand occasion and lifestyle needs
            - Note any specific preferences/restrictions

            After recommendations:
            - Check if they understand and like the suggestions
            - Offer alternatives if needed
            - Confirm if the style matches their preferences

            Reference previous conversations to maintain consistency and personalization.

            IMPORTANT: Your response must always be a JSON string in this format:
            {"text": "Your styling advice here", "value": ["image_id1", "image_id2"]}

            The 'text' field should include:
            - Styling advice and/or questions about preferences
            - Natural language based item references
            - No image ids should be present here.
            - Follow-up questions about understanding

            The 'value' field should contain:
            - Array of referenced image IDs
            - Empty array if no items referenced""",
            model="gpt-4o",
            tools=[{
                "type": "function",
                "function": {
                    "name": "query_wardrobe",
                    "description": "Search through the user's wardrobe and styling history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "stylist_id": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            }]
        )
        return assistant

    def _get_thread_key(self, unique_id, stylist_id):
        """Generate a unique key for user-stylist combination"""
        return f"{unique_id}_{stylist_id}"

    def _get_or_create_thread(self, unique_id, stylist_id):
        """Get existing thread or create new one for user-stylist combination"""
        thread_key = self._get_thread_key(unique_id, stylist_id)
        if thread_key not in self.user_threads:
            thread = client.beta.threads.create()
            self.user_threads[thread_key] = thread.id
            
            # Initialize the conversation with stylist introduction
            intro_message = self._get_stylist_introduction(stylist_id)
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="assistant",
                content=json.dumps({
                    "text": intro_message,
                    "value": []
                })
            )
        return self.user_threads[thread_key]

    def _get_stylist_introduction(self, stylist_id):
        """Get personalized introduction based on stylist"""
        intros = {
            "reginald": "Hi, I'm Reginald! I specialize in men's fashion, streetwear, and creating confident, versatile looks. Let's work together to elevate your style!",
            "eliza": "Hello, I'm Eliza! I'm here to help you create sophisticated looks that combine luxury with practicality. Let's explore your style possibilities!",
            "lilia": "Hi there, I'm Lilia! I'm excited to help you find styles that celebrate your unique body shape and make you feel confident and comfortable!"
        }
        return intros.get(stylist_id.lower(), "Hello! I'm your personal fashion stylist. How can I help you today?")

    def _cleanup_old_messages(self, thread_id):
        """Clean up old messages if they exceed max_history"""
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        if len(messages.data) > self.max_history:
            # Keep only the most recent messages
            # Note: OpenAI doesn't currently support bulk message deletion
            # This is a placeholder for when that functionality becomes available
            pass

    @traceable
    def generate_response(self, query, unique_id, stylist_id, image_id, wardrobe_data):
        """
        Generate response using OpenAI Assistant with RAG and conversation history.

        If token_name and unique_id are 'general_chat', consider the entire wardrobe.
        Otherwise, use the selected item (image_id) as part of the context.
        """
        # Check if the condition for 'general_chat' is met
        is_general_chat = (unique_id == "general_chat" and image_id == "general_chat")

        # Get or create thread for this user-stylist combination
        thread_id = self._get_or_create_thread(unique_id, stylist_id)

        # Prepare context message
        image_data = "\n".join(
            f"Image ID: {item['token_name']}, Caption: {item['caption']}" 
            for item in wardrobe_data
        )
        
        # Add specific context based on whether it's general chat or normal condition
        if is_general_chat:
            context_message = (
                f"You are {stylist_id.capitalize()}. Remember to maintain your unique styling perspective.\n"
                f"Available wardrobe items:\n{image_data}\n"
                f"User query: {query}\n"
                "IMPORTANT: Respond with a JSON string in this format:\n"
                '{"text": "Your styling advice here", "value": ["image_id1", "image_id2"]}\n'
                "Include relevant image IDs in the value array. Use empty array if no specific items referenced."
            )
        else:
            context_message = (
                f"You are {stylist_id.capitalize()}. Remember to maintain your unique styling perspective.\n"
                f"Available wardrobe items:\n{image_data}\n"
                f"Selected image ID: {image_id}\n"
                f"User query: {query}\n"
                "IMPORTANT: Respond with a JSON string in this format:\n"
                '{"text": "Your styling advice here", "value": ["image_id1", "image_id2"]}\n'
                "Include relevant image IDs in the value array. Use empty array if no specific items referenced."
            )

        # Add the new message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=context_message
        )

        # Clean up old messages if necessary
        self._cleanup_old_messages(thread_id)

        # Create a run with the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant.id
        )

        # Wait for the run to complete
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status == "requires_action":
                if run_status.required_action.type == "submit_tool_outputs":
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        if tool_call.function.name == "query_wardrobe":
                            args = json.loads(tool_call.function.arguments)
                            results = self.query_embeddings(
                                args["query"],
                                args.get("top_k", 5),
                                args.get("stylist_id")
                            )
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps([{
                                    "text": r.payload.get("text", ""),
                                    "metadata": r.payload.get("metadata", {})
                                } for r in results])
                            })
                    
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            elif run_status.status == "failed":
                raise Exception(f"Run failed: {run_status.last_error}")
            
            time.sleep(1)

        # Get the assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        latest_message = messages.data[0].content[0].text.value

        # Ensure proper JSON formatting
        try:
            response_json = json.loads(latest_message)
            
            if not isinstance(response_json, dict) or "text" not in response_json or "value" not in response_json:
                response_json = {
                    "text": latest_message,
                    "value": []
                }
            
            if not isinstance(response_json["value"], list):
                response_json["value"] = []
                
        except json.JSONDecodeError:
            response_json = {
                "text": latest_message,
                "value": []
            }

        return json.dumps(response_json)

    def reset_conversation(self, unique_id, stylist_id=None):
        """Reset the conversation history for a user-stylist combination"""
        if stylist_id:
            # Reset specific user-stylist thread
            thread_key = self._get_thread_key(unique_id, stylist_id)
            if thread_key in self.user_threads:
                thread = client.beta.threads.create()
                self.user_threads[thread_key] = thread.id
        else:
            # Reset all threads for this user
            keys_to_reset = [k for k in self.user_threads.keys() if k.startswith(f"{unique_id}_")]
            for key in keys_to_reset:
                thread = client.beta.threads.create()
                self.user_threads[key] = thread.id

# Usage example
def main():
    fashion_assistant = FashionAssistant()
    
    # Example query
    query = "What outfit can I create with my blue jeans?"
    stylist_id = "reginald"
    image_id = "image456"
    wardrobe_data = [
        {"token_name": "item1", "caption": "Blue denim jeans"},
        {"token_name": "item2", "caption": "White cotton t-shirt"}
    ]
    
    response = fashion_assistant.generate_response(
        query=query,
        stylist_id=stylist_id,
        image_id=image_id,
        wardrobe_data=wardrobe_data
    )
    
    # Assuming the response is a dictionary with structured information
    print("Outfit Recommendation:")
    print(f"Query: {query}")
    print(f"Stylist: {stylist_id}")
    print(f"Image ID: {image_id}")
    print("\nWardrobe Items:")
    
    for item in wardrobe_data:
        print(f" - {item['caption']} (Token: {item['token_name']})")
    
    print("\nResponse:")
    if isinstance(response, dict):
        for key, value in response.items():
            print(f"{key.capitalize()}: {value}")
    else:
        print(response)

if __name__ == "__main__":
    main()