from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, EmailStr, validator
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from langsmith.wrappers import wrap_openai
import base64
import tempfile
from langsmith import traceable
from openai import OpenAI
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client, Client
from auth_utils import hash_password, verify_password
import json
import asyncio
from starlette.responses import JSONResponse
import os
import uuid
import time
from enum import Enum
from decimal import Decimal
import base64
import smtplib
import re
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import asyncio
import secrets
from RAG_agents import chat_with_stylist, get_or_create_assistant
from dotenv import load_dotenv
import hashlib
import base64

def create_hash(password: str) -> str:
    """Create a simple but secure hash of the password."""
    # Convert password to bytes and create SHA-256 hash
    password_bytes = password.encode('utf-8')
    hash_obj = hashlib.sha256(password_bytes)
    # Convert hash to base64 string for storage
    return base64.b64encode(hash_obj.digest()).decode('utf-8')

def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verify if the provided password matches the stored hash."""
    try:
        # Create hash of provided password
        provided_hash = create_hash(provided_password)
        # Compare hashes
        return provided_hash == stored_hash
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            # Set to 28 seconds to be just under Heroku's 30-second limit
            return await asyncio.wait_for(call_next(request), timeout=28)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, 
                detail="Request processing time exceeded 28 seconds"
            )

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(TimeoutMiddleware)

# Load environment variables
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase clients
supabase: Client = create_client(supabase_url, supabase_key)
supabase_admin: Client = create_client(supabase_url, supabase_service_key)

# Initialize OpenAI client
openai_client = wrap_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY2")))

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fashay"

# FastAPI setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

# Setup templates and static files
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cache storages
registration_cache = {}
forgot_password_cache = {}





class AuthBase(BaseModel):
    email: EmailStr
    password: str
class CurrencyType(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class RefreshRequest(AuthBase):
    stylist: str
class ProductBase(BaseModel):
    id: uuid.UUID
    url: str
    name: str
    brand: str
    price: Decimal
    currency: CurrencyType
    colors: List[str]
    image_urls: List[str]
    category: str
    additional_notes: Optional[str]
    gender: Optional[str]
    created_at: datetime
    updated_at: datetime

class HMProduct(ProductBase):
    sku: str

class AmazonProduct(ProductBase):
    asin: str
    sizes: List[str]

class ProductResponse(BaseModel):
    status: str
    data: dict
class ProfilePictureResponse(BaseModel):
    status: str
    message: str
    url: Optional[str]

# New request model for the profile picture upload
class ProfilePictureRequest(BaseModel):
    email: str
    password: str
    image_data: str  # base64 string
    filename: str
    content_type: str

class SignupRequest(AuthBase):
    confirm_password: str

class VerificationRequest(AuthBase):
    verification_code: str

class PasswordResetRequest(BaseModel):
    email: EmailStr
    password: str  # Required by AuthBase
    reset_code: str
    new_password: str

class UserProfileData(AuthBase):
    favorite_styles: List[str]
    favorite_colors: List[str]
    size_preferences: Dict[str, str]
    budget_range: str
    favorite_materials: List[str]
    body_shape_info: Optional[str] = None
    style_determined_1: Optional[str] = None
    style_determined_2: Optional[str] = None
    style_determined_3: Optional[str] = None
    style_determined_4: Optional[str] = None
    style_determined_5: Optional[str] = None

class StyleResponse(BaseModel):
    unique_id: str
    stylist_id: str
    user_query: str
    stylist_response: str
    outfit_image_ids: Optional[List[str]] = None
    style_category: Optional[str] = None
    is_saved: bool = False
    preference: Optional[str] = None

class OutfitPreference(AuthBase):
    response_id: int
    preference: str  # LIKE or DISLIKE

class SaveOutfit(AuthBase):
    response_id: int
    outfit_image_ids: List[str]
    style_category: str

class ChatRequest(AuthBase):
    input_text: str
    token_name: str
    stylist: str

class ImageRequest(AuthBase):
    token_name: str

class GenderUpdateRequest(AuthBase):
    gender: str

# Response Models
class StandardResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None

class ProductRecommendation(BaseModel):
    category: str
    product_id: str
    price: float
    product_text: str
    retailer: str
    brand: str
    similarity_score: float
    image_urls: List[str]
    url: str
    suggestion: str
    gender: str
    styling_tips: Optional[List[str]] = None
class Recommendations(BaseModel):
    category_suggestions: Dict[str, str]
    products: List[ProductRecommendation]

class WardrobeImage(BaseModel):
    image_id: str
    image_url: str
    token_name: str
class ChatResponse(BaseModel):
    reply: str
    images: List[WardrobeImage]
    recommendations: Optional[Recommendations] = None
    shopping_analysis: Optional[Dict[str, Union[bool, float, str, List[str]]]] = None


class ProfileResponse(BaseModel):
    status: str
    data: Dict
    message: Optional[str] = None

# Helper Functions
def generate_verification_code() -> str:
    return ''.join(secrets.choice('0123456789') for _ in range(6))

def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, ""

# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
async def async_generate_with_retry(*args, **kwargs):
    return await chat_with_stylist(*args, **kwargs)

# Email Functions
def send_verification_email(email: str, code: str) -> bool:
    try:
        sender_email = "fashay.contact@gmail.com"
        sender_password = os.environ.get("EMAIL_PASSWORD")
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = "Email Verification Code"
        body = f"Your verification code is: {code}"
        message.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False






@app.get("/products", response_model=ProductResponse)
async def get_all_products(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=40, ge=1, le=100)
):
    """Get paginated products from all retailers"""
    try:
        # Calculate pagination
        start = (page - 1) * per_page
        end = start + per_page - 1

        # Get paginated product references
        refs_response = supabase.table("product_references")\
            .select("*", count="exact")\
            .order('created_at', desc=True)\
            .range(start, end)\
            .execute()

        if not refs_response.data:
            return ProductResponse(
                status="success",
                data={
                    "products": [],
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total_pages": 0,
                        "total_count": 0,
                        "has_more": False
                    }
                }
            )

        total_count = refs_response.count

        # Fetch products and combine with reference data
        all_products = []
        for ref in refs_response.data:
            try:
                # Get product from appropriate retailer table
                product_response = supabase.table(ref['retailer_table'])\
                    .select("*")\
                    .eq("id", ref['product_id'])\
                    .single()\
                    .execute()

                if product_response.data:
                    retailer = ref['retailer_table'].replace('_products', '')
                    product_data = {
                        **product_response.data,
                        'retailer': retailer,
                        'reference_id': ref['id'],
                    }
                    all_products.append(product_data)
            except Exception as e:
                print(f"Error fetching product {ref['product_id']}: {str(e)}")
                continue

        # Calculate pagination info
        total_pages = (total_count + per_page - 1) // per_page
        has_more = page < total_pages

        return ProductResponse(
            status="success",
            data={
                "products": all_products,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "total_count": total_count,
                    "has_more": has_more
                }
            }
        )

    except Exception as e:
        print(f"Error in get_all_products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/products/categories")
async def get_product_categories():
    """Get all unique product categories across retailers"""
    try:
        # Get categories from both tables
        categories = set()
        
        # Get HM categories
        hm_response = supabase.table("hm_products").select("category").execute()
        if hm_response.data:
            categories.update(item.get("category") for item in hm_response.data if item.get("category"))
        
        # Get Amazon categories
        amazon_response = supabase.table("amazon_products").select("category").execute()
        if amazon_response.data:
            categories.update(item.get("category") for item in amazon_response.data if item.get("category"))

        return {
            "status": "success",
            "data": sorted(list(categories))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.get("/products/brands")
async def get_product_brands():
    """Get all unique brands across retailers"""
    try:
        brands = set()
        
        # Get HM brands
        hm_response = supabase.table("hm_products").select("brand").execute()
        if hm_response.data:
            brands.update(item.get("brand") for item in hm_response.data if item.get("brand"))
        
        # Get Amazon brands
        amazon_response = supabase.table("amazon_products").select("brand").execute()
        if amazon_response.data:
            brands.update(item.get("brand") for item in amazon_response.data if item.get("brand"))

        return {
            "status": "success",
            "data": sorted(list(brands))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get brands: {str(e)}")

@app.get("/products/{product_id}")
async def get_product_details(product_id: str):
    """Get detailed information about a specific product"""
    try:
        # First get the product reference to determine the retailer
        ref_response = supabase.table("product_references")\
            .select("*")\
            .eq("product_id", product_id)\
            .single()\
            .execute()

        if not ref_response.data:
            raise HTTPException(status_code=404, detail="Product not found")

        retailer_table = ref_response.data["retailer_table"]
        
        # Get the product details with the product reference included
        product_response = supabase.table(retailer_table)\
            .select("*, product_references!inner(*)")\
            .eq("id", product_id)\
            .single()\
            .execute()

        if not product_response.data:
            raise HTTPException(status_code=404, detail="Product not found")

        # Add retailer information
        product_data = {
            **product_response.data,
            'retailer': 'hm' if retailer_table == 'hm_products' else 'amazon',
            'product_code': product_response.data.get('sku') if retailer_table == 'hm_products' 
                          else product_response.data.get('asin')
        }

        return {
            "status": "success",
            "data": product_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get product details: {str(e)}")

@app.get("/products/similar/{product_id}")
async def get_similar_products(product_id: str, limit: int = 5):
    """Get similar products based on category and price range"""
    try:
        # Get the original product details
        ref_response = supabase.table("product_references")\
            .select("*")\
            .eq("product_id", product_id)\
            .single()\
            .execute()

        if not ref_response.data:
            raise HTTPException(status_code=404, detail="Product not found")

        retailer_table = ref_response.data["retailer_table"]
        
        # Get original product details
        product = supabase.table(retailer_table)\
            .select("*")\
            .eq("id", product_id)\
            .single()\
            .execute()

        if not product.data:
            raise HTTPException(status_code=404, detail="Product not found")

        price = float(product.data["price"] or 0)
        price_range = 0.2  # 20% price range
        min_price = price * (1 - price_range)
        max_price = price * (1 + price_range)

        similar_products = []
        
        # Get similar products from HM
        hm_similar = supabase.table("hm_products")\
            .select("*, product_references!inner(*)")\
            .eq("category", product.data["category"])\
            .gte("price", min_price)\
            .lte("price", max_price)\
            .neq("id", product_id)\
            .limit(limit)\
            .execute()

        if hm_similar.data:
            similar_products.extend([
                {**item, 'retailer': 'hm', 'product_code': item.get('sku')}
                for item in hm_similar.data
            ])

        # Get similar products from Amazon
        amazon_similar = supabase.table("amazon_products")\
            .select("*, product_references!inner(*)")\
            .eq("category", product.data["category"])\
            .gte("price", min_price)\
            .lte("price", max_price)\
            .neq("id", product_id)\
            .limit(limit)\
            .execute()

        if amazon_similar.data:
            similar_products.extend([
                {**item, 'retailer': 'amazon', 'product_code': item.get('asin')}
                for item in amazon_similar.data
            ])

        # Sort by price similarity and limit results
        similar_products.sort(key=lambda x: abs(float(x.get("price", 0) or 0) - price))
        similar_products = similar_products[:limit]

        return {
            "status": "success",
            "data": similar_products
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get similar products: {str(e)}")
# Update route handlers
@app.get("/view")
async def get_view_page(request: Request, email: str):
    try:
        # Get user data
        response = supabase.table("wardrobe").select("*").eq("email", email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
            
        unique_id = response.data[0]["unique_id"]
        
        # Get images data
        image_response = supabase.table("image_data").select("*").eq("unique_id", unique_id).execute()
        
        # Ensure proper template context
        return templates.TemplateResponse(
            "view.html",
            context={
                "request": request,  # Required by Jinja2
                "items": image_response.data or [],  # Ensure items is always iterable
                "profile_image": response.data[0].get("profile_image", ""),  # Default empty string
                "email": email
            }
        )
    except Exception as e:
        print(f"Template error: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

# Block direct template access
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    if request.url.path.endswith((".html", ".htm")):
        return JSONResponse(
            status_code=403,
            content={"message": "Direct template access forbidden"}
        )
    return JSONResponse(
        status_code=404,
        content={"message": "Resource not found"}
    )
@app.get("/login")
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
# Update root and login endpoints
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})
# Auth Endpoints
@app.post("/signup", response_model=StandardResponse)
async def signup(data: SignupRequest):
    try:
        # Check if email exists
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if response.data:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Validate password
        is_valid, error_message = validate_password(data.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Check if passwords match
        if data.password != data.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        # Generate hash and unique ID
        hashed_password = create_hash(data.password)
        unique_id = str(uuid.uuid4())
        
        # Generate verification code
        verification_code = generate_verification_code()
        
        # Store in registration cache
        registration_cache[verification_code] = {
            "email": data.email,
            "password": hashed_password,
            "unique_id": unique_id,
            "is_verified": False
        }
        
        # Send verification email
        if not send_verification_email(data.email, verification_code):
            raise HTTPException(status_code=500, detail="Failed to send verification email")
        
        return StandardResponse(
            status="pending_verification",
            message="Please check your email for verification code"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create account")
def save_to_db(email, hashed_password, unique_id):
    data = {
        "unique_id": unique_id,
        "email": email,
        "password": hashed_password  # Store the hashed password
    }
    response = supabase.table("wardrobe").insert(data).execute()
    
    if response.data:
        print("Data successfully saved to Supabase.")
        return True
    else:
        print("Error inserting data into Supabase:", response)
        return False
# Authentication Endpoints Continued
@app.post("/verify", response_model=StandardResponse)
async def verify(data: VerificationRequest):
    # Check if verification code exists in cache
    if data.verification_code not in registration_cache:
        raise HTTPException(status_code=400, detail="Invalid verification code")

    user_data = registration_cache[data.verification_code]
    try:
        # Add a check to ensure email matches
        if user_data["email"] != data.email:
            raise HTTPException(status_code=400, detail="Email mismatch")

        # Save to database using the password from cache
        saved = save_to_db(
            user_data["email"], 
            user_data["password"],
            user_data["unique_id"]
        )
        
        if not saved:
            raise HTTPException(status_code=500, detail="Failed to save user data")

        # Verify the data was actually saved by querying it
        verify_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not verify_response.data:
            raise HTTPException(status_code=500, detail="Data verification failed")

        # Remove from cache only after confirming save
        del registration_cache[data.verification_code]
        
        return StandardResponse(
            status="success",
            message="Account verified successfully",
            data={
                "email": data.email,
                "password": data.password  # This will be used for auto-login
            }
        )
    except Exception as e:
        print(f"Verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify account")
@app.post("/login", response_model=StandardResponse)
async def login(data: AuthBase):
    try:
        # Get user record
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")

        record = response.data[0]
        stored_password = record.get("password")  # Get the stored password hash

        # If no password in record
        if not stored_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Convert input password to hash and compare
        if verify_password(stored_password, data.password):
            return StandardResponse(
                status="success",
                message="Login successful. Redirecting to home."
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
def clean_expired_codes():
    """Remove expired verification codes from cache"""
    current_time = datetime.now()
    expired_codes = [
        code for code, data in forgot_password_cache.items()
        if current_time > data["expiry"]
    ]
    for code in expired_codes:
        del forgot_password_cache[code]

@app.post("/forgot-password", response_model=StandardResponse)
async def forgot_password(data: AuthBase):
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    verification_code = generate_verification_code()
    forgot_password_cache[verification_code] = {
        "email": data.email,
        "expiry": datetime.now() + timedelta(minutes=30)
    }
    
    if not send_verification_email(data.email, verification_code):
        raise HTTPException(status_code=500, detail="Failed to send verification code")

    clean_expired_codes()
    return StandardResponse(
        status="success",
        message="Verification code sent to your email"
    )
@app.post("/reset-password", response_model=StandardResponse)
async def reset_password(data: PasswordResetRequest):
    try:
        print(f"Received reset request for email: {data.email}")  # Debug log
        clean_expired_codes()
        
        # Verify reset code exists and is valid
        if data.reset_code not in forgot_password_cache:
            print(f"Invalid reset code: {data.reset_code}")  # Debug log
            raise HTTPException(status_code=400, detail="Invalid or expired verification code")

        stored_data = forgot_password_cache[data.reset_code]
        if stored_data["email"] != data.email:
            print(f"Email mismatch: {data.email} vs {stored_data['email']}")  # Debug log
            raise HTTPException(status_code=400, detail="Email mismatch")
        
        if datetime.now() > stored_data["expiry"]:
            del forgot_password_cache[data.reset_code]
            raise HTTPException(status_code=400, detail="Code expired")

        # Validate new password
        is_valid, error_message = validate_password(data.new_password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)

        # Hash the new password using the same hashlib method
        hashed_password = create_hash(data.new_password)
        
        # Update password in database
        response = supabase.table("wardrobe").update({
            "password": hashed_password
        }).eq("email", data.email).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to update password")

        # Clear the reset code from cache
        del forgot_password_cache[data.reset_code]
        
        return StandardResponse(
            status="success",
            message="Password reset successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Password reset error: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))
@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        # Replace asyncio.timeout with asyncio.wait_for for compatibility with Python <3.11
        return await asyncio.wait_for(call_next(request), timeout=75)  # Set a timeout of 75 seconds
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        # Log other exceptions for debugging
        print(f"Error in timeout_middleware: {e}")
        raise
# Chat and Styling Endpoints
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(data: ChatRequest):
    try:
        print("Step 1: Starting chat request")
        print("Received data:", data.dict())

        print("Step 2: Verifying credentials")
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        record = response.data[0]
        stored_hash = record["password"]
        
        print("Step 3: Verifying password")
        if not verify_password(stored_hash, data.password):
            raise HTTPException(status_code=401, detail="Invalid password")

        unique_id = record["unique_id"]
        
        print("Step 4: Fetching wardrobe data")
        image_data = supabase.table("image_data")\
            .select("token_name, image_caption, image_url")\
            .eq("unique_id", unique_id)\
            .execute()
            
        wardrobe_data = [
            {
                "token_name": record["token_name"],
                "caption": record["image_caption"],
                "image_url": record["image_url"]
            } 
            for record in image_data.data
        ]
        print(f"Found {len(wardrobe_data)} wardrobe items")

        print("Step 5: Generating response")
        try:
            
            start_time = time.time()
            # Replace asyncio.timeout with asyncio.wait_for
            result = await async_generate_with_retry(
                    query=data.input_text,
                    unique_id=unique_id,
                    stylist_id=data.stylist.lower(),
                    image_id=data.token_name,
                    wardrobe_data=wardrobe_data
                )
            print(f"Step X: Duration: {time.time() - start_time} seconds")

            #     timeout=0 # Timeout duration in seconds
            # )
        except asyncio.TimeoutError:
            print("Response generation timed out")
            raise HTTPException(status_code=504, detail="Response generation timed out")
        
        print("Step 6: Parsing response")
        try:
            parsed_result = json.loads(result)
            text = parsed_result.get("text", "No response text found.")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {str(e)}")
            print(f"Raw result: {result}")
            raise HTTPException(status_code=500, detail="Failed to parse response")

        print("Step 7: Processing images")
        images = []
        print(parsed_result, "parsed_result")
        value_items = parsed_result.get("value", [])
        if value_items and isinstance(value_items, list):
            product_ids = set()
            
            for item in value_items:
                if isinstance(item, dict) and "product_id" in item:
                    product_ids.add(item["product_id"])
                elif isinstance(item, str):
                    product_ids.add(item)
            
            for product_id in product_ids:
                matching_item = next(
                    (item for item in wardrobe_data if item["token_name"] == product_id),
                    None
                )
                if matching_item:
                    images.append(WardrobeImage(
                        image_id=product_id,
                        image_url=matching_item["image_url"],
                        token_name=matching_item["token_name"]
                    ))
        print(f"Processed {len(images)} images")

        print("Step 8: Processing recommendations")
        recommendations = None
        if "recommendations" in parsed_result:
            recs = parsed_result["recommendations"]
            products = []
            
            for product in recs.get("products", []):
                try:
                    if isinstance(product.get("product_text"), dict):
                        product_text = " ".join(str(v) for v in product["product_text"].values() if v)
                    else:
                        product_text = str(product.get("product_text", ""))

                    price = str(product.get("price", ""))
                    if isinstance(price, (int, float)):
                        price = str(price)

                    products.append(ProductRecommendation(
                        category=product.get("category", ""),
                        product_id=product.get("product_id", ""),
                        product_text=product_text,
                        price=price,
                        brand=product.get("brand", ""),
                        retailer=product.get("retailer", ""),
                        similarity_score=float(product.get("similarity_score", 0.0)),
                        image_urls=product.get("image_urls", []),
                        url=product.get("url", ""),
                        suggestion=recs.get("category_suggestions", {}).get(product.get("category", ""), ""),
                        gender=product.get("gender", ""),
                        styling_tips=product.get("styling_tips", [])
                    ))
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
                    continue

            if products:
                recommendations = Recommendations(
                    category_suggestions=recs.get("category_suggestions", {}),
                    products=products
                )
        
        print("Step 9: Preparing final response")
        shopping_analysis = parsed_result.get("shopping_analysis")

        return ChatResponse(
            reply=text,
            images=images,
            recommendations=recommendations,
            shopping_analysis=shopping_analysis
        )
    
    except Exception as e:
        print(f"Error in handle_chat: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
# @traceable
class ImageUploadRequest(BaseModel):
    email: str
    password: str
    token_name: str
    image: str  # base64 encoded image

    @validator('image')
    def validate_base64(cls, v):
        # Remove any potential data URI prefix
        if ';base64,' in v:
            v = v.split(';base64,')[1]
        # Check if the remaining string is valid base64
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError('Invalid base64 string')
# File Upload Endpoints
class ImageUploadResponse(BaseModel):
    token_name: str
    status: str
    caption: str
    image_url: str
    unique_id: str
    
def encode_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    buffered = BytesIO()
    input_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
@traceable
def get_apparel_features(image_path):
    base64_image = encode_image(image_path)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the fashion apparel in the image in terms of color, apparel type, texture, design pattern and style information. \
                    The output must be a sentence without adjectives and no json output. Statement should be less than 15 words"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content, base64_image
def save_image_data(unique_id: str, token_name: str, image_caption: str, image_url: str):
    data = {
        "unique_id": unique_id,
        "token_name": token_name,
        "image_caption": image_caption,
        "image_url": image_url  # Store URL instead of base64
    }
    
    response = supabase.table("image_data").insert(data).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to save image data")
@app.post("/upload", response_model=ImageUploadResponse)
@traceable
async def upload_image(data: ImageUploadRequest):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    record = response.data[0]
    unique_id = record["unique_id"]
    
    try:
        print(f"Received upload request for token: {data.token_name}")
        # Decode base64 image
        image_content = base64.b64decode(data.image)
        timestamp = str(int(time.time()))
        storage_path = f"{unique_id}/{timestamp}_{data.token_name}.png"
        
        # Upload to storage
        storage_result = supabase_admin.storage\
            .from_("wardrobe")\
            .upload(storage_path, image_content, {"content-type": "image/png"})
            
        if not storage_result:
            raise HTTPException(status_code=500, detail="Failed to upload image")
            
        # Get public URL
        file_url = supabase_admin.storage\
            .from_("wardrobe")\
            .get_public_url(storage_path)
            
        # Create a temporary file for the image content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_content)
            temp_file_path = temp_file.name
            
        try:
            # Get image caption
            image_caption, _ = get_apparel_features(temp_file_path)
            
            # Save image data
            save_image_data(unique_id, data.token_name, image_caption, file_url)
            
            return ImageUploadResponse(
                status="success",
                token_name=data.token_name,
                caption=image_caption,
                image_url=file_url,
                unique_id=unique_id
            )
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        if 'storage_path' in locals():
            try:
                supabase_admin.storage.from_("wardrobe").remove([storage_path])
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
    
# Profile and Style Management Endpoints
@app.post("/user_profile", response_model=ProfileResponse)
async def create_or_update_profile(data: UserProfileData):
    try:
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        record = response.data[0]
        stored_hash = record["password"]
        
        if not bcrypt.verify(data.password, stored_hash):
            raise HTTPException(status_code=401, detail="Invalid password")

        unique_id = record["unique_id"]
        
        # Prepare profile data
        profile_data = {
            "unique_id": unique_id,
            "favorite_styles": data.favorite_styles,
            "favorite_colors": data.favorite_colors,
            "size_preferences": data.size_preferences,
            "budget_range": data.budget_range,
            "favorite_materials": data.favorite_materials,
            "body_shape_info": data.body_shape_info,
            "style_determined_1": data.style_determined_1,
            "style_determined_2": data.style_determined_2,
            "style_determined_3": data.style_determined_3,
            "style_determined_4": data.style_determined_4,
            "style_determined_5": data.style_determined_5,
            "last_updated": datetime.now().isoformat()
        }

        # Update or create profile
        existing_profile = supabase.table("user_profile").select("*").eq("unique_id", unique_id).execute()
        
        if existing_profile.data:
            response = supabase.table("user_profile")\
                .update(profile_data)\
                .eq("unique_id", unique_id)\
                .execute()
        else:
            profile_data["created_at"] = datetime.now().isoformat()
            response = supabase.table("user_profile")\
                .insert(profile_data)\
                .execute()

        return ProfileResponse(
            status="success",
            message="Profile updated successfully",
            data=response.data[0]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

@app.get("/user_profile/{unique_id}", response_model=ProfileResponse)
async def get_user_profile(unique_id: str, data: AuthBase):
    try:
        # Verify credentials
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        # Get profile data
        profile_response = supabase.table("user_profile").select("*").eq("unique_id", unique_id).execute()
        if not profile_response.data:
            raise HTTPException(status_code=404, detail="Profile not found")

        return ProfileResponse(
            status="success",
            data=profile_response.data[0]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile: {str(e)}")

# View and Delete Endpoints
class ViewImagesResponse(BaseModel):
    items: List[Dict[str, str]]
    profile_image: Optional[str]

@app.post("/view", response_model=ViewImagesResponse)
async def view_images(data: AuthBase = None):
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    record = response.data[0]
    unique_id = record["unique_id"]
    
    # Fetch images
    image_response = supabase.table("image_data").select("*").eq("unique_id", unique_id).execute()
    
    items = [
        {
            "token_name": row["token_name"],
            "image_caption": row["image_caption"],
            "image_url": row["image_url"],
            "unique_id": row["unique_id"]
        } 
        for row in image_response.data
    ]
    
    # Get profile image
    profile_image_record = supabase.table("wardrobe").select("profile_image").eq("email", data.email).execute()
    profile_image = profile_image_record.data[0]["profile_image"] if profile_image_record.data else None

    return ViewImagesResponse(items=items, profile_image=profile_image)

class DeleteItemResponse(BaseModel):
    success: bool
    message: str

async def delete_from_storage(bucket: str, file_path: str):
    """
    Delete file from Supabase storage
    """
    try:
        response = supabase.storage.from_(bucket).remove([file_path])
        return response
    except Exception as e:
        print(f"Error deleting from storage: {str(e)}")
        raise
@app.delete("/delete_item/{token_name}", response_model=DeleteItemResponse)
async def delete_item(token_name: str, data: AuthBase):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    record = response.data[0]
    unique_id = record["unique_id"]
    
    try:
        # Get image data
        image_data = supabase.table("image_data").select("*").eq("token_name", token_name).execute()
        if not image_data.data:
            raise HTTPException(status_code=404, detail="Item not found")

        # Delete from storage
        storage_path = f"{unique_id}/{token_name}"
        await delete_from_storage("wardrobe", storage_path)
        
        # Delete from database
        delete_response = supabase.table("image_data").delete().eq("token_name", token_name).execute()
        
        return DeleteItemResponse(
            success=True,
            message="Item deleted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {str(e)}")

# Gender Management Endpoints
@app.post("/update_gender", response_model=StandardResponse)
async def update_gender(data: GenderUpdateRequest):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    try:
        update_response = supabase.table("wardrobe")\
            .update({"gender": data.gender})\
            .eq("email", data.email)\
            .execute()
            
        return StandardResponse(
            status="success",
            message="Gender identity updated successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update gender: {str(e)}")

@app.get("/get_gender")
async def get_gender(email: str):
    response = supabase.table("wardrobe").select("gender").eq("email", email).execute()
    gender = "other"
    if response.data and "gender" in response.data[0]:
        gender = response.data[0]["gender"]
    
    return {"gender": gender}

# Stylist Management Endpoint
@app.post("/refresh_stylist", response_model=StandardResponse)
@traceable
async def refresh_stylist(data: RefreshRequest):
    try:
        # Verify credentials
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        record = response.data[0]
        unique_id = record["unique_id"]
        
        # Get and reset the assistant
        fashion_assistant = get_or_create_assistant(data.stylist.lower())
        intro_message = fashion_assistant.reset_conversation(unique_id, data.stylist.lower())
        print(f"Refreshed stylist: {intro_message}")
        return StandardResponse(
            status="success",
            message="Stylist refreshed successfully",
            data={"initial_message": intro_message}
        )
        
    except Exception as e:
        print(f"Error in refresh_stylist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh stylist: {str(e)}")


# Profile Picture Management
class ProfilePictureResponse(BaseModel):
    status: str
    message: str
    url: Optional[str]

@app.post("/upload_profile_picture", response_model=ProfilePictureResponse)
async def upload_profile_picture(data: ProfilePictureRequest):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    try:
        # Process image
        try:
            image_content = base64.b64decode(data.image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        timestamp = str(int(time.time()))
        file_extension = os.path.splitext(data.filename)[1].lower()
        storage_path = f"profile_pictures/{timestamp}_{data.email}{file_extension}"
        
        # Upload to storage
        storage_response = supabase_admin.storage\
            .from_("wardrobe")\
            .upload(storage_path, image_content, {"content-type": data.content_type})
            
        if not storage_response:
            raise HTTPException(status_code=500, detail="Failed to upload to storage")
            
        # Get public URL
        profile_url = supabase_admin.storage\
            .from_("wardrobe")\
            .get_public_url(storage_path)
            
        # Update database
        update_response = supabase.table("wardrobe")\
            .update({"profile_image": profile_url})\
            .eq("email", data.email)\
            .execute()
            
        return ProfilePictureResponse(
            status="success",
            message="Profile picture updated successfully",
            url=profile_url
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in upload_profile_picture: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile picture: {str(e)}")


@app.post("/style/response", response_model=StandardResponse)
async def add_style_response(data: StyleResponse):
    try:
        # Verify user exists
        user_response = supabase.table("wardrobe").select("*").eq("unique_id", data.unique_id).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")

        # Prepare the data
        style_data = {
            "unique_id": data.unique_id,
            "stylist_id": data.stylist_id,
            "user_query": data.user_query,
            "stylist_response": data.stylist_response,
            "outfit_image_ids": data.outfit_image_ids or [],
            "style_category": data.style_category,
            "is_saved": data.is_saved,
            "preference": data.preference,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Insert into database
        response = supabase.table("user_style_outfits").insert(style_data).execute()

        return StandardResponse(
            status="success",
            message="Style response saved successfully",
            data={"response_id": response.data[0]["id"]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save style response: {str(e)}")

@app.post("/style/preference", response_model=StandardResponse)
async def update_preference(data: OutfitPreference):
    try:
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        stored_hash = record["password"]
        
        # Replace with
        if not verify_password(stored_hash, data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify preference value
        if data.preference not in ["LIKE", "DISLIKE"]:
            raise HTTPException(status_code=400, detail="Invalid preference value")

        # Update preference
        response = supabase.table("user_style_outfits")\
            .update({"preference": data.preference, "updated_at": datetime.now().isoformat()})\
            .eq("id", data.response_id)\
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Style response not found")

        return StandardResponse(
            status="success",
            message=f"Preference updated to {data.preference}",
            data={"response_id": data.response_id}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preference: {str(e)}")

@app.post("/style/save-outfit", response_model=StandardResponse)
async def save_outfit(data: SaveOutfit):
    try:
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        stored_hash = record["password"]
        
        if not verify_password(stored_hash, data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Update outfit data
        update_data = {
            "outfit_image_ids": data.outfit_image_ids,
            "style_category": data.style_category,
            "is_saved": True,
            "saved_date": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        response = supabase.table("user_style_outfits")\
            .update(update_data)\
            .eq("id", data.response_id)\
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Style response not found")

        return StandardResponse(
            status="success",
            message="Outfit saved successfully",
            data={"response_id": data.response_id}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save outfit: {str(e)}")

# Optional: Add a getter endpoint to retrieve saved outfits
@app.get("/style/saved-outfits/{unique_id}", response_model=StandardResponse)
async def get_saved_outfits(unique_id: str, data: AuthBase):
    try:
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        stored_hash = record["password"]
        
        if not verify_password(stored_hash, data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Get saved outfits
        response = supabase.table("user_style_outfits")\
            .select("*")\
            .eq("unique_id", unique_id)\
            .eq("is_saved", True)\
            .order("saved_date", desc=True)\
            .execute()

        return StandardResponse(
            status="success",
            message="Saved outfits retrieved successfully",
            data={"outfits": response.data}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve saved outfits: {str(e)}")

# Server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)