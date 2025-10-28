import os
import re
import json
import requests
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from dotenv import load_dotenv
import google.generativeai as genai

# ================== CONFIG =====================
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Color codes for visual overlay
CLASS_COLORS = {
    21: (0, 255, 0),     # greenery (trees)
    97: (0, 0, 255),     # water
    13: (128, 128, 128), # road
    11: (255, 0, 0),     # building/wall
}

# ================== STEP 1: DOWNLOAD STATIC MAP =====================
def download_static_map(latitude, longitude, zoom=19, size="640x640"):
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("Google Maps API key not found. Set GOOGLE_MAPS_API_KEY in .env")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={latitude},{longitude}&zoom={zoom}&size={size}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    )

    os.makedirs("files", exist_ok=True)
    file_path = f"files/static_map_{latitude}_{longitude}.png"

    print(f"🌍 Downloading satellite map for {latitude}, {longitude} ...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Map saved to {file_path}")
        return file_path
    else:
        raise RuntimeError(f"Failed to download map: {response.status_code}")

# ================== STEP 2: LOCAL SEGMENTATION =====================
def load_model():
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def preprocess(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def create_overlay(orig, mask):
    overlay = orig.copy()
    alpha = 0.45
    for cls, color in CLASS_COLORS.items():
        overlay[mask == cls] = (1 - alpha) * overlay[mask == cls] + alpha * np.array(color)
    return overlay.astype(np.uint8)

def segment_image(image_path):
    print("🔍 Running segmentation model locally...")
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    inp = preprocess(img)

    with torch.no_grad():
        output = model(inp)["out"][0]
    mask = output.argmax(0).cpu().numpy()

    orig = np.array(img)
    overlay = create_overlay(orig, mask)

    output_path = image_path.replace(".png", "_overlay.png")
    Image.fromarray(overlay).save(output_path)
    print(f"✅ Segmentation overlay saved: {output_path}")
    return output_path

# ================== STEP 3: GEMINI COVERAGE ANALYSIS =====================
def get_mime_type(file_path):
    extension = file_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
    }
    if extension in mime_types:
        return mime_types[extension]
    raise ValueError(f"Unsupported file format '{extension}'.")

def caption_image(image_path, prompt):
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env")

    genai.configure(api_key=GOOGLE_API_KEY)
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    mime_type = get_mime_type(image_path)
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content([
        {"mime_type": mime_type, "data": image_data},
        prompt
    ])
    return response.text

def extract_json_from_caption(caption):
    match = re.search(r"```json\s*(\{.*?\})\s*```", caption, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in model response.")
    json_str = match.group(1)
    return json.loads(json_str)

def generate_coverage_details(image_path):
    prompt = """From this image, estimate the land coverage percentages and return the result in valid JSON format. 
The JSON must match this schema:
{
  "vegetation_coverage": float,
  "building_coverage": float,
  "road_coverage": float,
  "empty_land": float,
  "water_body": float
}
Only return the JSON object. Do not include any explanation or text."""
    print("🤖 Asking Gemini to analyze land coverage...")
    caption = caption_image(image_path, prompt)
    return extract_json_from_caption(caption)

# ================== MAIN FLOW =====================
def analyze_location(latitude, longitude):
    map_path = download_static_map(latitude, longitude)
    overlay_path = segment_image(map_path)
    coverage = generate_coverage_details(overlay_path)
    print("\n🌿 Land Coverage Details:")
    print(json.dumps(coverage, indent=2))
    return {
        "map_path": map_path,
        "overlay_path": overlay_path,
        "coverage": coverage
    }

# ================== RUN EXAMPLE =====================
if __name__ == "__main__":
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))
    analyze_location(lat, lon)
