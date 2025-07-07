from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sqlite3
from sqlite3 import Error
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
from fastapi import FastAPI, HTTPException

from step0 import download_static_map
from step1 import generate_coverage_details
from step2 import generate_final_report

app = FastAPI()
app.mount("/static", StaticFiles(directory="public"), name="static")
app.mount("/files", StaticFiles(directory="files"), name="files")
FILES_DIR = os.path.dirname(__file__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get-image/{image_name}")
async def get_image(image_name: str):
    file_path = os.path.join(FILES_DIR, image_name)

    # Ensure file is inside the expected directory to prevent path traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(FILES_DIR)):
        raise HTTPException(status_code=400, detail="Invalid file path.")

    # Check if file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(file_path, media_type="image/png")


@app.get("/")
async def test():
    return {
        "status": "success",
        "message": "Welcome to the Image Greenery Report Generator API"
    }
# async def serve_index():
#     return FileResponse(os.path.join("publix", "index.html"))

@app.post("/analyze-image/")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    pass  # Placeholder for future image analysis logic

@app.post("/analyze-location/")
async def analyze_location(request: Request, city: str = Form(...),country: str = Form(...),latitude: float = Form(...), longitude: float = Form(...), zoom: int = Form(18)):
    """    Analyze a location by downloading a static map and generating a coverage report.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        zoom (int): Zoom level for the static map.
    Returns:
        dict: Result with status, message, and analysis result.
    """
    map_result = download_static_map(latitude, longitude, zoom=zoom)


    print(city, country, latitude, longitude)

    if map_result["status"] == "error":
        return {
            "status": "error",
            "message": map_result["message"],
            "file_path": None
        }
    try:
        coverage_details = generate_coverage_details(map_result["file_path"])
        if coverage_details["status"] == "error":
            return {
                "status": "error",
                "message": coverage_details["message"],
                "file_path": map_result["file_path"]
            }
        final_report = generate_final_report(coverage_details["caption"], city, country ,latitude, longitude)
        print(final_report)
        return {
            "status": "success",
            "message": "Location analyzed successfully.",
            "file_path": map_result["file_path"],
            "final_report": final_report
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "file_path": map_result["file_path"]
        }
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
