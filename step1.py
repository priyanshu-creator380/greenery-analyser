import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



def get_mime_type(file_path):
    extension = file_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp'
    }
    if extension in mime_types:
        return mime_types[extension]
    raise ValueError(f"Unsupported file format '{extension}'. Supported formats: {', '.join(mime_types.keys())}")

def caption_image(image_path, prompt="Caption this image.", api_key=None):
    try:
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Set it in .env or pass explicitly.")

        genai.configure(api_key=api_key)

        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        mime_type = get_mime_type(image_path)

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_data},
            prompt
        ])
        return response.text

    except Exception as e:
        raise RuntimeError(f"Captioning failed: {e}")

def extract_json_from_caption(caption):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", caption, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in the caption.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Error extracting JSON: {e}")

def generate_coverage_details(image_path="files/static_map.png", latitude=None, longitude=None):
    prompt = """From this image, estimate the land coverage percentages and return the result in valid JSON format. The JSON must match the following schema:
{
  "vegetation_coverage": float,
  "building_coverage": float,
  "road_coverage": float,
  "empty_land": float,
  "water_body": float
}
Only return the JSON object. Do not include any explanation or extra text. All values should be in percentage format as floats (e.g., 23.5).
"""

    try:
        caption = caption_image(image_path, prompt=prompt)
        parsed_json = extract_json_from_caption(caption)
        return {
            "status": "success",
            "message": "Coverage details extracted successfully.",
            "caption": parsed_json
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "caption": None
        }

# Example usage:
# result = generate_coverage_details()
# print(result)
