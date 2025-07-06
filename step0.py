import requests
from dotenv import load_dotenv
import os

def download_static_map(latitude, longitude, zoom=19, size="640x640"):
    """
    Downloads a static map image from Google Maps Static API.

    Args:
        latitude (float): Latitude of the map center.
        longitude (float): Longitude of the map center.
        zoom (int): Zoom level.
        size (str): Image size in 'WIDTHxHEIGHT' format.
        map_type (str): Map type ('roadmap', 'satellite', 'terrain', 'hybrid').
        file_path (str): Path to save the image.
        api_key (str): Google Maps API key. If None, loads from environment variable.

    Returns:
        dict: Result with status, message, and file_path.
    """
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        raise ValueError("Google Maps API key not found. Set it in .env or pass explicitly.")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={latitude},{longitude}&zoom={zoom}&size={size}&maptype={"satellite"}&key={api_key}"
    )
    os.makedirs("files", exist_ok=True)
    file_path = f"files/static_map-{latitude}-{longitude}.png"
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        return {
            "status": "success",
            "message": "Map image saved successfully.",
            "file_path": file_path,
            "latitude": latitude,
            "longitude": longitude,
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to download map: {response.status_code}",
            "file_path": None,
            "latitude": latitude,
            "longitude": longitude,
        }


# result = download_static_map(25.5941, 85.1376, file_path="static_map.png")
# print(result)
# This will download a static map centered at the specified latitude and longitude, save it as 'static_map.png', and print the result.