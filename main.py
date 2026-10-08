import base64
import json
import os
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StyleMe by MDX API")

# Setup safe communication access for your tablet's local browser environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Strict Structured JSON Output Profile
class PhotoshootConcept(BaseModel):
    custom_aesthetic_name: str = Field(description="The unique name given to this user-defined aesthetic style.")
    aesthetic_summary: str = Field(description="A brief 1-2 sentence description of the vibe and fashion mood.")
    location_suggestions: list[str] = Field(description="Three specific environments or backdrops matching the clothing and aesthetic.", min_items=3, max_items=3)
    posing_guide: list[str] = Field(description="Three actionable modeling or framing poses tailored to the clothing style.", min_items=3, max_items=3)
    lighting_and_camera_vibe: str = Field(description="Detailed description of lighting, gear feel, editing style, color grading, or film stock.")
    prop_ideas: list[str] = Field(description="Two or three small items to hold or use in the frame to enhance the story.", min_items=2, max_items=3)
    image_prompt: str = Field(description="A detailed, descriptive text prompt optimized for diffusion models (Imagen 3) to generate a representative preview photo.")

@app.post("/generate-studio")
async def generate_studio_concept(
    custom_aesthetic_name: str = Form(...),
    style_elements: str = Form(...),
    manual_clothing: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None)
):
    """API Endpoint processing text descriptors or image file packets to run styling matrices."""
    try:
        client = genai.Client()
        style_elements_list = json.loads(style_elements)
        detected_clothing = manual_clothing or ""

        # Trigger Gemini Vision scan matrix if an image file structure was sent from the web app
        if image_file:
            image_bytes = await image_file.read()
            
            vision_prompt = (
                "Analyze this fashion image thoroughly. Extract and list the exact clothing items, "
                "color breakdowns, textures, materials, layer structures, footwear, and accessory details you see."
            )
            
            vision_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=image_file.content_type),
                    vision_prompt
                ]
            )
            detected_clothing = vision_response.text

        # Base prompt mapping logic parameters assembly
        system_instruction = (
            "You are an expert fashion photographer and creative director for top-tier Instagram influencers. "
            "Your task is to take user-defined custom fashion styles, outfit items, and custom aesthetic names, "
            "and produce a comprehensive, structured photoshoot concept along with an ideal image-generation prompt."
        )

        styles_formatted = ", ".join(style_elements_list)
        prompt = f"""
        Create a complete photoshoot concept based on this custom aesthetic build:
        - Custom Aesthetic Name: "{custom_aesthetic_name}"
        - Style Elements / Influences: {styles_formatted}
        - Detected Outfit / Clothing Profile: {detected_clothing}

        Ensure all suggestions feel highly detailed and completely cohesive with the user's custom aesthetic vision.
        """

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PhotoshootConcept,
            temperature=0.75,
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config=config
        )
        brief_data = json.loads(response.text)

        # Execute automated graphic composition preview via Imagen 3 model engines
        image_prompt = brief_data["image_prompt"]
        image_response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=image_prompt,
            config=types.GenerateImageConfig(
                number_of_images=1,
                aspect_ratio="3:4",  # Perfect vertical frame layout composition for social feeds
                output_mime_type="image/jpeg",
            ),
        )

        # Transpile output image bytes layout straight into data strings for fluid network delivery
        if image_response.generated_images:
            img_bytes = image_response.generated_images[0].image.image_bytes
            base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
            brief_data["image_data_url"] = f"data:image/jpeg;base64,{base64_encoded}"
        else:
            brief_data["image_data_url"] = None

        return brief_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
