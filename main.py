import base64
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(title="Aura Style Studio API")

# Allow your index.html website to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from your local tablet environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Structured JSON Response Schema
class PhotoshootConcept(BaseModel):
    custom_aesthetic_name: str = Field(description="The unique name given to this user-defined aesthetic style.")
    aesthetic_summary: str = Field(description="A brief 1-2 sentence description of the vibe and fashion mood.")
    location_suggestions: list[str] = Field(description="Three specific environments or backdrops matching the clothing and aesthetic.", min_items=3, max_items=3)
    posing_guide: list[str] = Field(description="Three actionable modeling or framing poses tailored to the clothing style.", min_items=3, max_items=3)
    lighting_and_camera_vibe: str = Field(description="Detailed description of lighting, gear feel, editing style, color grading, or film stock.")
    prop_ideas: list[str] = Field(description="Two or three small items to hold or use in the frame to enhance the story.", min_items=2, max_items=3)
    image_prompt: str = Field(description="A detailed, descriptive text prompt optimized for diffusion models (Imagen 3) to generate a representative preview photo.")

# 2. Incoming Data Schema (What your index.html will send)
class StyleRequest(BaseModel):
    detected_clothing: str
    style_elements: list[str]
    custom_aesthetic_name: str

@app.post("/generate-studio")
async def generate_studio_concept(request: StyleRequest):
    """API Endpoint that processes front-end requests and returns text brief + base64 image data."""
    try:
        # Initialise the Google GenAI client (Reads GEMINI_API_KEY from environment)
        client = genai.Client()

        system_instruction = (
            "You are an expert fashion photographer and creative director for top-tier Instagram influencers. "
            "Your task is to take user-defined custom fashion styles, outfit items, and custom aesthetic names, "
            "and produce a comprehensive, structured photoshoot concept along with an ideal image-generation prompt."
        )

        styles_formatted = ", ".join(request.style_elements)
        prompt = f"""
        Create a complete photoshoot concept based on this custom aesthetic build:
        - Custom Aesthetic Name: "{request.custom_aesthetic_name}"
        - Style Elements / Influences: {styles_formatted}
        - Outfit / Clothing: {request.detected_clothing}

        Ensure all suggestions feel cohesive with the user's custom aesthetic vision.
        """

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PhotoshootConcept,
            temperature=0.75,
        )

        # Generate text brief
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        brief_data = json.loads(response.text)

        # Generate visual preview image using Imagen 3
        image_prompt = brief_data["image_prompt"]
        image_response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=image_prompt,
            config=types.GenerateImageConfig(
                number_of_images=1,
                aspect_ratio="3:4",
                output_mime_type="image/jpeg",
            ),
        )

        # Convert image bytes directly to a web-friendly text format (Base64 string)
        if image_response.generated_images:
            img_bytes = image_response.generated_images[0].image.image_bytes
            base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            # This format injects straight into an HTML <img> tag src attribute
            brief_data["image_data_url"] = f"data:image/jpeg;base64,{base64_encoded}"
        else:
            brief_data["image_data_url"] = None

        return brief_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
