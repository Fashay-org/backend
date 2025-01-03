from fastapi import FastAPI, UploadFile, HTTPException, File, Form, Body, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from langsmith.wrappers import wrap_openai
import base64
import tempfile
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from openai import OpenAI
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, HTTPException, File, Form, Body, Request, Depends
from urllib.parse import parse_qsl
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client, Client
import bcrypt
import json
import os
import uuid
import time
import base64
import smtplib
import random
import string
import re
import secrets
from RAG_agents import chat_with_stylist, get_or_create_assistant
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

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


# Pydantic Models
class AuthBase(BaseModel):
    email: EmailStr
    password: str

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

class PasswordResetRequest(AuthBase):
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
    product_text: str
    retailer: str
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
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



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if response.data:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    is_valid, error_message = validate_password(data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
    unique_id = str(uuid.uuid4())
    verification_code = generate_verification_code()
    
    registration_cache[verification_code] = {
        "email": data.email,
        "hashed_password": hashed_password,
        "unique_id": unique_id,
        "is_verified": False
    }
    
    if not send_verification_email(data.email, verification_code):
        raise HTTPException(status_code=500, detail="Failed to send verification email")
    
    return StandardResponse(
        status="pending_verification",
        message="Please check your email for verification code"
    )
def save_to_db(email, hashed_password, unique_id):
    data = {
        "unique_id": unique_id,
        "email": email,
        "password": hashed_password.decode('utf-8')
    }
    response = supabase.table("wardrobe").insert(data).execute()
    
    if response.data:
        print("Data successfully saved to Supabase.")
    else:
        print("Error inserting data into Supabase:", response.json())
# Authentication Endpoints Continued
@app.post("/verify", response_model=StandardResponse)
async def verify(data: VerificationRequest):
    if data.verification_code not in registration_cache:
        raise HTTPException(status_code=400, detail="Invalid verification code")

    user_data = registration_cache[data.verification_code]
    try:
        save_to_db(
            user_data["email"], 
            user_data["hashed_password"], 
            user_data["unique_id"]
        )
        del registration_cache[data.verification_code]
        return StandardResponse(
            status="success",
            message="Account verified successfully",
            data={"email": data.email, "password": data.password}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")

@app.post("/login", response_model=StandardResponse)
async def login(data: AuthBase):
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")

    record = response.data[0]
    hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
    
    if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    return StandardResponse(
        status="success",
        message="Login successful. Redirecting to home."
    )

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
    clean_expired_codes()
    if data.reset_code not in forgot_password_cache:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")

    stored_data = forgot_password_cache[data.reset_code]
    if stored_data["email"] != data.email:
        raise HTTPException(status_code=400, detail="Email mismatch")
    
    if datetime.now() > stored_data["expiry"]:
        del forgot_password_cache[data.reset_code]
        raise HTTPException(status_code=400, detail="Code expired")

    is_valid, error_message = validate_password(data.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)

    hashed_password = bcrypt.hashpw(data.new_password.encode('utf-8'), bcrypt.gensalt())
    response = supabase.table("wardrobe").update({
        "password": hashed_password.decode('utf-8')
    }).eq("email", data.email).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to update password")

    del forgot_password_cache[data.reset_code]
    return StandardResponse(
        status="success",
        message="Password reset successfully"
    )

# Chat and Styling Endpoints
@app.post("/chat", response_model=ChatResponse)
@traceable
async def handle_chat(data: ChatRequest):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    record = response.data[0]
    hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
    
    if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    unique_id = record["unique_id"]
    
    # Get wardrobe data with full information including URLs
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

    try:
        result = await async_generate_with_retry(
            query=data.input_text,
            unique_id=unique_id,
            stylist_id=data.stylist.lower(),
            image_id=data.token_name,
            wardrobe_data=wardrobe_data
        )
        
        parsed_result = json.loads(result)
        text = parsed_result.get("text", "No response text found.")
        image_ids = parsed_result.get("value", [])

        # Process wardrobe images more efficiently
        images = []
        for image_id in image_ids:
            matching_item = next(
                (item for item in wardrobe_data if item["token_name"] == image_id),
                None
            )
            if matching_item:
                images.append(WardrobeImage(
                    image_id=image_id,
                    image_url=matching_item["image_url"],
                    token_name=matching_item["token_name"]
                ))

        # Process recommendations if available
        recommendations = None
        if "recommendations" in parsed_result:
            recs = parsed_result["recommendations"]
            
            # Format product recommendations
            products = []
            for product in recs.get("products", []):
                products.append(ProductRecommendation(
                    category=product["category"],
                    product_id=product["product_id"],
                    product_text=product["product_text"],
                    retailer=product["retailer"],
                    similarity_score=product["similarity_score"],
                    image_urls=product.get("image_urls", []),
                    url=product.get("url", ""),
                    suggestion=recs.get("category_suggestions", {}).get(product["category"], ""),
                    gender=product.get("gender", ""),
                    styling_tips=product.get("styling_tips", [])
                ))
            
            recommendations = Recommendations(
                category_suggestions=recs.get("category_suggestions", {}),
                products=products
            )

        # Include shopping analysis if available
        shopping_analysis = parsed_result.get("shopping_analysis")

        return ChatResponse(
            reply=text,
            images=images,
            recommendations=recommendations,
            shopping_analysis=shopping_analysis
        )
    
    except Exception as e:
        # logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
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
        # Verify credentials
        response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Email not found")
        
        record = response.data[0]
        hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
        
        if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
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
async def refresh_stylist(data: ChatRequest):
    # Verify credentials
    response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Email not found")
    
    record = response.data[0]
    unique_id = record["unique_id"]
    
    try:
        fashion_assistant = get_or_create_assistant(data.stylist.lower())
        intro_message = fashion_assistant.reset_conversation(unique_id, data.stylist.lower())
        
        return StandardResponse(
            status="success",
            message="Stylist refreshed successfully",
            data={"initial_message": intro_message}
        )
        
    except Exception as e:
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
        # Verify user credentials
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
        
        if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
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
        # Verify user credentials
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
        
        if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
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
        # Verify user credentials
        user_response = supabase.table("wardrobe").select("*").eq("email", data.email).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail="User not found")
        
        record = user_response.data[0]
        hashed_password = record["password"].encode('utf-8') if isinstance(record["password"], str) else record["password"]
        
        if not bcrypt.checkpw(data.password.encode('utf-8'), hashed_password):
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