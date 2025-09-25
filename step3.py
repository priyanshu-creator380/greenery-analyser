import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def add_images_from_pexels(recommendations):
    """
    Fetch one image per plant from Pexels and add it to the recommendations list.
    
    :param recommendations: List of plant recommendation dicts
    :return: Updated recommendations with 'image' field
    """
    headers = {"Authorization": PEXELS_API_KEY}
    
    for rec in recommendations:
        plant_name = rec["name"]
        url = f"https://api.pexels.com/v1/search?query={plant_name}&per_page=1"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["photos"]:
                rec["image"] = data["photos"][0]["src"]["original"]
            else:
                rec["image"] = None
        except Exception as e:
            rec["image"] = None
            print(f"⚠️ Error fetching image for {plant_name}: {e}")

    return recommendations
