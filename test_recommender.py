import asyncio
import json
from typing import Dict, Any, List
from fashion_assistant import chat_with_stylist
from mock_data import setup_mock_environment
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up mock environment with real embeddings
embeddings, mock_supabase = setup_mock_environment()

# Sample wardrobe data for testing
TEST_WARDROBE = [
    {
        "token_name": "item1",
        "caption": "Black leather biker jacket with silver hardware",
        "category": "outerwear",
        "gender": "unisex"
    },
    {
        "token_name": "item2",
        "caption": "Raw denim selvedge jeans, slim fit",
        "category": "bottoms",
        "gender": "male"
    },
    {
        "token_name": "item3",
        "caption": "White Oxford cotton button-down shirt",
        "category": "tops",
        "gender": "unisex"
    },
    {
        "token_name": "item4",
        "caption": "Brown suede Chelsea boots with leather sole",
        "category": "footwear",
        "gender": "male"
    },
    {
        "token_name": "item5",
        "caption": "Navy merino wool crew neck sweater",
        "category": "tops",
        "gender": "unisex"
    }
]

async def test_scenario(scenario: Dict[str, Any]):
    """Run a single test scenario"""
    print(f"\n=== Testing Scenario: {scenario['name']} ===")
    print(f"Query: {scenario['query']}")
    print(f"Stylist: {scenario['stylist_id']}")
    
    if scenario['image_id'] != "general_chat":
        focus_item = next(
            (item['caption'] for item in TEST_WARDROBE if item['token_name'] == scenario['image_id']), 
            'Unknown'
        )
        print(f"Focus Item: {focus_item}")
    
    print("\nAvailable Wardrobe Items:")
    for item in TEST_WARDROBE:
        print(f"- {item['caption']} ({item['category']})")

    try:
        response = await chat_with_stylist(
            query=scenario['query'],
            unique_id="test_user_123",
            stylist_id=scenario['stylist_id'],
            image_id=scenario['image_id'],
            wardrobe_data=TEST_WARDROBE
        )
        
        response_dict = json.loads(response)
        
        print("\nStylist Response:")
        print("-" * 80)
        print(response_dict['text'])
        print("-" * 80)
        
        # Show selected wardrobe items
        if 'value' in response_dict and response_dict['value']:
            print("\nSelected Wardrobe Items:")
            for item_id in response_dict['value']:
                item = next(
                    (i['caption'] for i in TEST_WARDROBE if i['token_name'] == item_id), 
                    'Unknown item'
                )
                print(f"- {item}")
        
        # Show product recommendations
        if 'recommendations' in response_dict:
            recs = response_dict['recommendations']
            if recs.get('products'):
                print("\nRecommended Products to Purchase:")
                print("-" * 80)
                for product in recs['products']:
                    print(f"\nCategory: {product['category']}")
                    print(f"Product: {product['product_text']}")
                    print(f"Retailer: {product.get('retailer', 'Unknown')}")
                    print(f"Similarity Score: {product.get('similarity_score', 0):.3f}")
                    print(f"Available Images: {len(product.get('image_urls', []))}")
                    print("-" * 40)
        
        return response_dict
        
    except Exception as e:
        print(f"Error in scenario: {str(e)}")
        return None

async def main():
    """Run all test scenarios"""
    print("Starting Fashion System Tests with Real Embeddings...")
    print(f"Using {len(embeddings)} real product embeddings")
    
    # Test scenarios
    scenarios = [
        {
            "name": "Casual Weekend Outfit",
            "query": "I need a casual but stylish weekend outfit from my current wardrobe",
            "stylist_id": "eliza",
            "image_id": "general_chat"
        },
        {
            "name": "Formal Event Shopping",
            "query": "I need to buy a complete formal outfit for a wedding next month",
            "stylist_id": "reginald",
            "image_id": "general_chat"
        },
        {
            "name": "Mixed Wardrobe and Shopping",
            "query": "I want to wear my leather jacket but need new items to create a rock concert outfit",
            "stylist_id": "lilia",
            "image_id": "item1"
        },
        {
            "name": "Specific Item Focus with Shopping",
            "query": "How can I style these Chelsea boots? I'm willing to buy new items to create the perfect outfit",
            "stylist_id": "reginald",
            "image_id": "item4"
        },
        {
            "name": "Seasonal Transition",
            "query": "Help me transition my wardrobe for fall, suggesting both styling of current items and new purchases",
            "stylist_id": "eliza",
            "image_id": "general_chat"
        }
    ]
    
    for scenario in scenarios:
        await test_scenario(scenario)
    
    print("\nAll test scenarios completed!")

if __name__ == "__main__":
    asyncio.run(main())