from fastapi import FastAPI, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
import os
import bcrypt
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
import json
from langsmith import Client
import functools
import base64
from PIL import Image
from RAG_agents import FashionAssistant
import uuid
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
# from process import generate_mask
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client, Client
import os
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import re
import secrets
from datetime import datetime, timedelta

app = FastAPI()
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")  # anon key
supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Public client for regular operations
supabase: Client = create_client(supabase_url, supabase_key)

# Admin client for storage operations
supabase_admin: Client = create_client(supabase_url, supabase_service_key)



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")  # If this is how it's named in your .env
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fashay"

print(os.environ.get("OPENAI_API_KEY2"), "OPENAI_API_KEY2")
openai_client = wrap_openai(OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY2"),
))
# Cache to store registration information temporarily
registration_cache = {}

# Initialize FashionAssistant
fashion_assistant = FashionAssistant()
# Add this to store temporary verification data
forgot_password_cache = {}

def generate_verification_code() -> str:
    """Generate a 6-digit verification code"""
    return ''.join(secrets.choice('0123456789') for _ in range(6))

def send_forgot_password_email(email: str, code: str) -> bool:
    """
    Send password reset verification code via email
    Returns True if email was sent successfully, False otherwise
    """
    try:
        sender_email = "fashay.contact@gmail.com"
        sender_password = os.environ.get("EMAIL_PASSWORD")

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = "Fashay - Password Reset Code"

        body = f"""
        Hello,

        You have requested to reset your password for your Fashay account.
        Your verification code is: {code}

        This code will expire in 30 minutes.
        If you did not request this password reset, please ignore this email.

        Best regards,
        The Fashay Team
        """
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
            
        return True
    except Exception as e:
        print(f"Error sending forgot password email: {str(e)}")
        return False

def clean_expired_codes():
    """Remove expired verification codes from cache"""
    current_time = datetime.now()
    expired_codes = [
        code for code, data in forgot_password_cache.items()
        if current_time > data["expiry"]
    ]
    for code in expired_codes:
        del forgot_password_cache[code]

@app.post("/forgot-password")
async def forgot_password(email: str = Form(...)):
    """
    Handle forgot password requests
    """
    try:
        # Check if email exists in database
        response = supabase.table("wardrobe").select("*").eq("email", email).execute()
        if not response.data:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No account found with this email address"
                },
                status_code=404
            )

        # Generate verification code
        verification_code = generate_verification_code()
        
        # Store in cache with 30-minute expiry
        forgot_password_cache[verification_code] = {
            "email": email,
            "expiry": datetime.now() + timedelta(minutes=30)
        }

        # Send verification email
        if not send_forgot_password_email(email, verification_code):
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Failed to send verification code. Please try again."
                },
                status_code=500
            )

        # Clean up expired codes
        clean_expired_codes()

        return JSONResponse(
            content={
                "status": "success",
                "message": "Verification code sent to your email"
            }
        )

    except Exception as e:
        print(f"Error in forgot_password: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": "An error occurred. Please try again."
            },
            status_code=500
        )

@app.get("/contact")
async def contact():
    contact_path = os.path.join(FRONTEND_DIR, "contact.html")
    if not os.path.exists(contact_path):
        raise HTTPException(status_code=404, detail="Contact page not found")
    return FileResponse(contact_path, media_type="text/html")
@app.post("/reset-password")
async def reset_password(
    email: str = Form(...),
    reset_code: str = Form(...),
    new_password: str = Form(...)
):
    """
    Handle password reset with verification code
    """
    try:
        # Clean expired codes first
        clean_expired_codes()

        # Verify the code exists and matches
        if reset_code not in forgot_password_cache:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Invalid or expired verification code"
                },
                status_code=400
            )

        # Get stored data
        stored_data = forgot_password_cache[reset_code]

        # Verify email matches
        if stored_data["email"] != email:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Email address doesn't match verification code"
                },
                status_code=400
            )

        # Verify code hasn't expired
        if datetime.now() > stored_data["expiry"]:
            del forgot_password_cache[reset_code]
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Verification code has expired"
                },
                status_code=400
            )

        # Validate new password
        is_valid, error_message = validate_password(new_password)
        if not is_valid:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": error_message
                },
                status_code=400
            )

        # Hash new password
        hashed_password = bcrypt.hashpw(
            new_password.encode('utf-8'),
            bcrypt.gensalt()
        )

        # Update password in database
        response = supabase.table("wardrobe").update({
            "password": hashed_password.decode('utf-8')
        }).eq("email", email).execute()

        if not response.data:
            raise Exception("Failed to update password in database")

        # Remove used verification code
        del forgot_password_cache[reset_code]

        return JSONResponse(content={
            "status": "success",
            "message": "Password has been reset successfully"
        })

    except Exception as e:
        print(f"Error in reset_password: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": "An error occurred while resetting password"
            },
            status_code=500
        )
def generate_verification_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password against security requirements.
    Returns (is_valid, error_message)
    """
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
def send_verification_email(email, code):
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
async def upload_to_storage(file_bytes: bytes, bucket: str, file_path: str) -> str:
    """
    Upload file to Supabase storage and return the public URL
    """
    try:
        # Upload the file to storage
        response = supabase.storage.from_(bucket).upload(
            file_path,
            file_bytes,
            {"content-type": "image/jpeg"}  # Adjust content-type as needed
        )
        
        # Get the public URL
        public_url = supabase.storage.from_(bucket).get_public_url(file_path)
        return public_url
    
    except Exception as e:
        print(f"Error uploading to storage: {str(e)}")
        raise

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



# Get the absolute path to the frontend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
print(FRONTEND_DIR, "FRONTEND_DIR")
templates = Jinja2Templates(directory=FRONTEND_DIR)

@app.get("/login")
def read_login():
    login_path = os.path.join(FRONTEND_DIR, "login.html")
    if not os.path.exists(login_path):
        raise HTTPException(status_code=404, detail="Login page not found")
    return FileResponse(login_path, media_type="text/html")

@app.get("/")
def read_root():
    login_path = os.path.join(FRONTEND_DIR, "app.html")
    if not os.path.exists(login_path):
        raise HTTPException(status_code=404, detail="App page not found")
    return FileResponse(login_path, media_type="text/html")


@app.post("/signup")
async def signup(email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    # print("Signup attempt with email:", email)

    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    if response.data:
        return JSONResponse(
            content={"status": "error", "message": "Email already exists. Please sign in."}, 
            status_code=400
        )

    # Validate password
    is_valid, error_message = validate_password(password)
    if not is_valid:
        return JSONResponse(
            content={"status": "error", "message": error_message},
            status_code=400
        )
    if password != confirm_password:
        return JSONResponse(
            content={"status": "error", "message": "Passwords do not match"}, 
            status_code=400
        )
    
    # Validate email format
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return JSONResponse(
            content={"status": "error", "message": "Invalid email format"},
            status_code=400
        )
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    unique_id = str(uuid.uuid4())
    verification_code = generate_verification_code()

    registration_cache[verification_code] = {
        "email": email,
        "hashed_password": hashed_password,
        "unique_id": unique_id,
        "is_verified": False  # Add verification status
    }

    send_verification_email(email, verification_code)

    return JSONResponse(
        content={
            "status": "pending_verification",  # Changed status to be more specific
            "message": "Please check your email for verification code"
        }
    )

@app.post("/verify")
async def verify(email: str = Form(...), password: str = Form(...), verification_code: str = Form(...)):
    if verification_code not in registration_cache:
        return JSONResponse(
            content={"status": "error", "message": "Invalid verification code"}, 
            status_code=400
        )

    user_data = registration_cache[verification_code]
    try:
        save_to_db(
            user_data["email"], 
            user_data["hashed_password"], 
            user_data["unique_id"]
        )
        del registration_cache[verification_code]
        return JSONResponse(
            content={
                "status": "success",
                "message": "Account verified successfully.",
                "email": email,
                "password": password  # Send back the plain password for localStorage
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": f"Failed to register user: {str(e)}"}, 
            status_code=500
        )


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    record = response.data

    if record:
        hashed_password = record[0]["password"]
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')

        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return JSONResponse(content={"status": "success", "message": "Login successful. Redirecting to home."})
        else:
            return JSONResponse(content={"status": "error", "message": "Invalid password"}, status_code=401)
    else:
        return JSONResponse(content={"status": "error", "message": "Email not found"}, status_code=404)



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

import time
@app.post("/upload")
@traceable
async def upload_image(
    email: str = Form(...),
    password: str = Form(...),
    token_name: str = Form(...),
    image: UploadFile = Form(...)
):
    temp_input_path = None
    temp_output_path = None
    storage_path = None
    def resize_image_with_padding(image_path, target_size=(300, 300)):
        """Resize image maintaining aspect ratio and add white padding"""
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Calculate scaling factor to fit within target size
        width_ratio = target_size[0] / img.width
        height_ratio = target_size[1] / img.height
        resize_ratio = min(width_ratio, height_ratio)
        
        # Calculate new size maintaining aspect ratio
        new_size = (
            int(img.width * resize_ratio),
            int(img.height * resize_ratio)
        )
        
        # Resize image with LANCZOS resampling
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        # Create white background image
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate position to paste resized image (center)
        paste_pos = (
            (target_size[0] - new_size[0]) // 2,
            (target_size[1] - new_size[1]) // 2
        )
        
        # Paste resized image onto white background
        background.paste(resized_img, paste_pos)
        
        return background
    try:
        # Verify user credentials
        auth_response = supabase.table("wardrobe").select("*").eq("email", email).execute()
        record = auth_response.data

        if not record:
            raise HTTPException(status_code=404, detail="Email not found")

        hashed_password = record[0]["password"]
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')

        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            raise HTTPException(status_code=401, detail="Invalid password")

        unique_id = record[0]["unique_id"]

        # Read image content and save temporarily
        image_content = await image.read()

        os.makedirs("upload", exist_ok=True)
        temp_input_path = os.path.join("upload", f"temp_input_{int(time.time())}{os.path.splitext(image.filename)[1]}")
        
        with open(temp_input_path, "wb") as buffer:
            buffer.write(image_content)

        # Resize image
        resized_image = resize_image_with_padding(temp_input_path)
        resized_temp_path = os.path.join("upload", f"resized_{int(time.time())}.png")
        resized_image.save(resized_temp_path, format="PNG")

        # Replace original temp file with resized version
        os.remove(temp_input_path)
        temp_input_path = resized_temp_path
        # Generate masked image
        # base64_image = encode_image(temp_input_path)
        # classification_response = openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": "Classify the given image into fashion apparel and accessories. Fashion apparel means a piece of \
        #                  clothing like jeans, jacket, hoodie etc that is worn by a person and accessories means any accessory like sneakers, jewelry, \
        #                  sunglasses etc. that is worn by a person. Only return one word (apparel) or (accessory)"},
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/jpeg;base64,{base64_image}"
        #                 }
        #             },
        #             ],
        #         }
        #     ],
        #     max_tokens=300,
        # )
        
        # if "apparel" in str(classification_response.choices[0].message.content):
        #     masked_image = generate_mask(temp_input_path)
        # else:
        masked_image = Image.open(temp_input_path).convert('RGB')
        
        # Save masked image
        temp_output_path = os.path.join("upload", f"temp_output_{int(time.time())}.png")
        masked_image.save(temp_output_path, format="PNG")

        # Read masked image for upload
        with open(temp_output_path, "rb") as masked_file:
            masked_content = masked_file.read()

        # Generate storage path
        timestamp = str(int(time.time()))
        storage_path = f"{unique_id}/{timestamp}_{token_name}.png"

        # Upload masked image to Supabase Storage
        try:
            storage_result = supabase_admin.storage \
                .from_("wardrobe") \
                .upload(
                    storage_path,  # path parameter
                    masked_content,  # file parameter
                    {"content-type": "image/png"}  # file_options parameter
                )
            
            if not storage_result:
                raise Exception("Storage upload failed")
                
        except Exception as storage_error:
            print(f"Storage upload error: {str(storage_error)}")
            raise Exception(f"Failed to upload to storage: {str(storage_error)}")

        # Get public URL
        try:
            file_url = supabase_admin.storage \
                .from_("wardrobe") \
                .get_public_url(storage_path)
        except Exception as url_error:
            print(f"Error getting public URL: {str(url_error)}")
            # Try to clean up the uploaded file
            try:
                supabase_admin.storage.from_("wardrobe").remove([storage_path])
            except:
                pass
            raise Exception(f"Failed to get public URL: {str(url_error)}")

        # Get image caption
        try:
            image_caption, _ = get_apparel_features(temp_output_path)
        except Exception as caption_error:
            print(f"Error getting image caption: {str(caption_error)}")
            # Try to clean up the uploaded file
            try:
                supabase_admin.storage.from_("wardrobe").remove([storage_path])
            except:
                pass
            raise Exception(f"Failed to get image caption: {str(caption_error)}")

        # Save image data
        try:
            save_image_data(unique_id, token_name, image_caption, file_url)
        except Exception as db_error:
            print(f"Error saving to database: {str(db_error)}")
            # Try to clean up the uploaded file
            try:
                supabase_admin.storage.from_("wardrobe").remove([storage_path])
            except:
                pass
            raise Exception(f"Failed to save image data: {str(db_error)}")

        return JSONResponse(content={
            'status': 'success',
            'caption': image_caption,
            'image_url': file_url,
            'unique_id': unique_id
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        # Clean up storage if upload succeeded but later steps failed
        if storage_path:
            try:
                supabase_admin.storage.from_("wardrobe").remove([storage_path])
            except Exception as delete_error:
                print(f"Error cleaning up storage: {str(delete_error)}")
        
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': f'Failed to process image: {str(e)}',
                'error_type': 'processing_error'
            }
        )

    finally:
        # Clean up temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e:
                print(f"Error removing temp input file: {str(e)}")
        
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception as e:
                print(f"Error removing temp output file: {str(e)}")
@app.post("/view")
async def view_images(email: str = Form(...), password: str = Form(...), request: Request = None):
    # Verify user credentials
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    records = response.data

    if not records:
        raise HTTPException(status_code=404, detail="Email not found")

    hashed_password = records[0]["password"]
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')

    if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        unique_id = records[0]["unique_id"]
        
        # Fetch images from image_data table
        image_response = supabase.table("image_data").select("*").eq("unique_id", unique_id).execute()
        images = image_response.data
        
        # Use image_url instead of image_base64
        items = [
            {
                "token_name": row["token_name"],
                "image_caption": row["image_caption"],
                "image_url": row["image_url"],  # Changed from image_base64 to image_url
                "unique_id": row["unique_id"]
            } 
            for row in images
        ]
        
        # Get profile image
        profile_image_record = supabase.table("wardrobe").select("profile_image").eq("email", email).execute().data
        profile_image = profile_image_record[0]["profile_image"] if profile_image_record else None
        
        return templates.TemplateResponse(
            "view.html", 
            {
                "request": request, 
                "items": items, 
                "profile_image": profile_image
            }
        )
    else:
        raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/refresh_stylist")
@traceable
async def refresh_stylist(
    email: str = Form(...),
    password: str = Form(...),
    stylist: str = Form(...)
):
    # Verify user credentials
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    records = response.data
    
    if not records:
        return JSONResponse(content={"error": "Email not found"}, status_code=404)
    
    hashed_password = records[0]["password"]
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    
    if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        return JSONResponse(content={"error": "Invalid password"}, status_code=401)
        
    try:
        # Fetch user unique ID based on email
        user_unique_id_response = supabase.table("wardrobe").select("unique_id").eq("email", email).execute()
        # print("user", user_unique_id_response)
        user_data = user_unique_id_response.data
        # Get the unique_id
        unique_id = user_data[0]["unique_id"]
        # Reset the conversation for the specific stylist
        print("stylist", stylist, "UNIQUE_ID", unique_id)
        fashion_assistant.reset_conversation(unique_id, stylist.lower())
        
        try:
            # parsed_result = json.loads(result)
            text = "Hello! I'm your refreshed stylist. How can I help you today?"
            
            return JSONResponse(content={
                "status": "success",
                "message": "Stylist refreshed successfully",
                "initial_message": text
            })
            
        except json.JSONDecodeError:
            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Stylist refreshed successfully",
                    "initial_message": "Hello! I'm your refreshed stylist. How can I help you today?"
                }
            )
            
    except Exception as e:
        print(f"Error in refresh_stylist: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Failed to refresh stylist: {str(e)}"
            },
            status_code=500
        )
@app.post("/chat")
@traceable
async def handle_chat(
    input_text: str = Form(...), 
    token_name: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...),
    stylist: str = Form(...)  # Add stylist parameter
):
    print(f"Debug chat: Received request with stylist: {stylist}")
    print(f"Debug chat: input_text={input_text}, token_name={token_name}, email={email}, password={'*' * len(password)}")
    # print(f"Debug chat: input_text={input_text}, unique_id={unique_id}, token_name={token_name}, email={email}, password={'*' * len(password)}")

    # Verify user credentials
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    print(response)
    records = response.data
    if not records:
        return JSONResponse(content={"error": "Email not found"}, status_code=404)
    
    hashed_password = records[0]["password"]
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    
    if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        return JSONResponse(content={"error": "Invalid password"}, status_code=401)

    # Fetch user unique ID based on email
    user_unique_id_response = supabase.table("wardrobe").select("unique_id").eq("email", email).execute()
    print("user", user_unique_id_response)

    # Access the data attribute to get the unique_id
    user_data = user_unique_id_response.data
    if not user_data:
        raise ValueError("No unique_id found for the given email.")

    # Get the unique_id
    unique_id = user_data[0]["unique_id"]

    # Fetch image data based on the unique_id
    image_data_response = supabase.table("image_data").select("token_name", "image_caption").eq("unique_id", unique_id).execute()
    image_data = image_data_response.data
    wardrobe_data = [{"token_name": record["token_name"], "caption": record["image_caption"]} for record in image_data]
    
    try:
        # Pass token_name and unique_id as "general_chat" if applicable
        if token_name == "general_chat" and unique_id == "general_chat":
            result = fashion_assistant.generate_response(
                query=input_text,
                unique_id="general_chat",
                stylist_id=stylist.lower(),
                image_id="general_chat",
                wardrobe_data=wardrobe_data
            )
        else:
            result = fashion_assistant.generate_response(
                query=input_text,
                unique_id=unique_id,
                stylist_id=stylist.lower(),
                image_id=token_name,
                wardrobe_data=wardrobe_data
            )
        
        try:
            parsed_result = json.loads(result)
            print(parsed_result, "Debug pasrsed result")
            text = parsed_result.get("text", "No response text found.")
            image_ids = parsed_result.get("value", [])

            images = []
            for image_id in image_ids:
                image_response = supabase.table("image_data").select("image_url").eq("token_name", image_id).execute()
                print(image_response.data[0]["image_url"], "url", image_id, "image id")
                if image_response.data:
                    images.append({
                        "image_id": image_id, 
                        "image_url": image_response.data[0]["image_url"]
                    })
            print(images, "images")
            return JSONResponse(content={"reply": text, "images": images})
            
        except json.JSONDecodeError:
            return JSONResponse(
                content={
                    "error": "Failed to parse JSON from response", 
                    "result": result
                }, 
                status_code=500
            )
            
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return JSONResponse(
            content={
                "error": f"Failed to generate response: {str(e)}", 
                "result": None
            }, 
            status_code=500
        )


@app.post("/delete_item/{token_name}")
async def delete_item(token_name: str, email: str = Form(...), password: str = Form(...)):
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    record = response.data

    if not record:
        raise HTTPException(status_code=404, detail="Email not found")
    
    unique_id = record[0]["unique_id"]
    hashed_password = record[0]["password"]
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')

    if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        try:
            # Get the image data first
            image_data = supabase.table("image_data")\
                .select("*")\
                .eq("token_name", token_name)\
                .execute()

            if image_data.data:
                # Delete from storage first
                storage_path = f"{unique_id}/{token_name}"
                supabase_admin.storage \
                    .from_("wardrobe") \
                    .remove([storage_path])
                
                # Then delete from database
                delete_response = supabase.table("image_data")\
                    .delete()\
                    .eq("token_name", token_name)\
                    .execute()
                
                if delete_response.data:
                    return JSONResponse(content={"success": True, "message": "Item deleted successfully"})
                
            return JSONResponse(content={"success": False, "message": "Item not found"}, status_code=404)
            
        except Exception as e:
            return JSONResponse(
                status_code=500, 
                content={"success": False, "message": f"Failed to delete item: {str(e)}"}
            )
    else:
        raise HTTPException(status_code=401, detail="Invalid password")

@app.post("/update_gender")
async def update_gender(
    email: str = Form(...),
    password: str = Form(...),
    gender: str = Form(...)  # Gender identity field
):
    # Check if user exists and verify password
    response = supabase.table("wardrobe").select("*").eq("email", email).execute()
    record = response.data

    if not record:
        raise HTTPException(status_code=404, detail="Email not found")

    hashed_password = record[0]["password"]
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')

    if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Update gender identity in the wardrobe table
    update_response = supabase.table("wardrobe").update({"gender": gender}).eq("email", email).execute()
    if update_response.data:
        return JSONResponse(content={'success': True, 'message': 'Gender identity updated successfully'})
    else:
        return JSONResponse(content={'success': False, 'message': 'Failed to update gender identity'}, status_code=500)
@app.get("/get_gender")
async def get_gender(email: str):
    response = supabase.table("wardrobe").select("gender").eq("email", email).execute()
    if response.data and "gender" in response.data[0]:
        gender = response.data[0]["gender"]
        return JSONResponse(content={"gender": gender})
    else:
        return JSONResponse(content={"gender": "other"})  # Default to "other" if gender is missing



@app.post("/upload_profile_picture")
async def upload_profile_picture(
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = Form(...)
):
    try:
        print("Uploading profile picture... debug", email, password, image)
        # Verify user
        response = supabase.table("wardrobe").select("*").eq("email", email).execute()
        record = response.data

        if not record:
            raise HTTPException(status_code=404, detail="Email not found")

        hashed_password = record[0]["password"]
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')

        if not bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            raise HTTPException(status_code=401, detail="Invalid password")

        # Read image content
        image_content = await image.read()
        
        # Create unique filename
        timestamp = str(int(time.time()))
        file_extension = os.path.splitext(image.filename)[1].lower()
        storage_path = f"profile_pictures/{timestamp}_{email}{file_extension}"

        try:
            # Upload to storage using bytes directly
            storage_response = supabase_admin.storage \
                .from_("wardrobe") \
                .upload(
                    storage_path,  # Remove 'path=' as it's a positional argument
                    image_content, # Direct bytes instead of file object
                    {"content-type": image.content_type}  # Simplified file options
                )

            if not storage_response:  # Check if upload failed
                raise Exception("Failed to upload to storage")

            # Get public URL
            profile_url = supabase_admin.storage \
                .from_("wardrobe") \
                .get_public_url(storage_path)

            # Update profile URL in database
            update_response = supabase.table("wardrobe") \
                .update({"profile_image": profile_url}) \
                .eq("email", email) \
                .execute()

            if not update_response.data:
                # If database update fails, try to clean up the uploaded file
                try:
                    supabase_admin.storage \
                        .from_("wardrobe") \
                        .remove([storage_path])
                except:
                    pass  # Silently fail cleanup
                raise Exception("Failed to update database")

            return JSONResponse(
                content={
                    "status": "success",
                    "message": "Profile picture updated successfully",
                    "url": profile_url
                }
            )

        except Exception as storage_error:
            print(f"Storage error: {str(storage_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Storage error: {str(storage_error)}"
            )

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "error_type": "processing_error"
            }
        )
# At the bottom of main.py, change the run block to:
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
