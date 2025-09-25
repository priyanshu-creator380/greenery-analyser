# Imports and setup
import os
import requests
from typing import Dict, Optional
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# ------------------------------
# Data structures
# ------------------------------
@dataclass
class WeatherData:
    temperature: float
    humidity: float
    pressure: float
    weather_description: str
    wind_speed: float
    precipitation: float
    feels_like: float
    uv_index: float
    visibility: float
    local_time: str
    timezone: str

@dataclass
class SeasonData:
    season: str
    month: int
    day_of_year: int
    is_growing_season: bool
    planting_season: str

@dataclass
class LandCoverageData:
    vegetation_coverage: float
    building_coverage: float
    road_coverage: float
    empty_land: float
    water_body: float

@dataclass
class LocationData:
    latitude: float
    longitude: float
    city: str
    country: str

# ------------------------------
# Plant Recommendation Parser
# ------------------------------
class PlantRecommendationParser(BaseOutputParser):
    def parse(self, text: str) -> Dict:
        try:
            lines = text.strip().split('\n')
            recommendations = []
            current_plant = {}

            for line in lines:
                line = line.strip()
                if line.startswith('Plant:') or line.startswith('**Plant:'):
                    if current_plant:
                        recommendations.append(current_plant)
                    current_plant = {'name': line.split(':', 1)[1].strip().replace('**', '')}
                elif line.startswith('Reason:') or line.startswith('**Reason:'):
                    current_plant['reason'] = line.split(':', 1)[1].strip().replace('**', '')
                elif line.startswith('Care:') or line.startswith('**Care:'):
                    current_plant['care'] = line.split(':', 1)[1].strip().replace('**', '')

            if current_plant:
                recommendations.append(current_plant)

            return {
                'recommendations': recommendations,
            }
        except Exception as e:
            return {
                'recommendations': [],
                'error': str(e)
            }

# ------------------------------
# Weather Service (Google Weather API)
# ------------------------------
class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://weather.googleapis.com/v1/currentConditions:lookup"

    def get_weather_data(self, latitude: float, longitude: float) -> Optional[WeatherData]:
        try:
            params = {
                "key": self.api_key,
                "location.latitude": latitude,
                "location.longitude": longitude
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            current = data.get("weatherCondition", {})
            temperature_info = data.get("temperature", {})
            feels_like_info = data.get("feelsLikeTemperature", {})
            wind_info = data.get("wind", {})
            precipitation_info = data.get("precipitation", {})
            visibility_info = data.get("visibility", {})
            pressure_info = data.get("airPressure", {})

            return WeatherData(
                temperature=temperature_info.get("degrees", 0.0),
                humidity=data.get("relativeHumidity", 0.0),
                pressure=pressure_info.get("meanSeaLevelMillibars", 0.0),
                weather_description=current.get("description", {}).get("text", ""),
                wind_speed=wind_info.get("speed", {}).get("value", 0.0) / 3.6,  # km/h to m/s
                precipitation=precipitation_info.get("qpf", {}).get("quantity", 0.0),
                feels_like=feels_like_info.get("degrees", 0.0),
                uv_index=data.get("uvIndex", 0.0),
                visibility=visibility_info.get("distance", 0.0),
                local_time=data.get("currentTime", ""),
                timezone=data.get("timeZone", {}).get("id", "")
            )
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

# ------------------------------
# Season Service
# ------------------------------
class SeasonService:
    @staticmethod
    def determine_season(date_str: str, timezone_str: str, latitude: float = 0, city: str = "", country: str = "") -> SeasonData:
        try:
            local_time = datetime.strptime(date_str[:16], '%Y-%m-%dT%H:%M')
            month = local_time.month
            day_of_year = local_time.timetuple().tm_yday
            is_southern = latitude < 0
            season = "Unknown"
            planting = "Season-appropriate planting recommended"

            is_tropical_region = country.lower() in ["india", "bangladesh", "sri lanka", "nepal", "pakistan"]
            if is_tropical_region and month in [6, 7, 8, 9]:
                season = "Rainy"
                planting = "Excellent for rice, sugarcane, tropical fruits, and water-loving plants"
            else:
                if is_southern:
                    if month in [12, 1, 2]:
                        season = "Summer"
                    elif month in [3, 4, 5]:
                        season = "Autumn"
                    elif month in [6, 7, 8]:
                        season = "Winter"
                    else:
                        season = "Spring"
                else:
                    if month in [12, 1, 2]:
                        season = "Winter"
                    elif month in [3, 4, 5]:
                        season = "Spring"
                    elif month in [6, 7, 8]:
                        season = "Summer"
                    else:
                        season = "Autumn"

                if season == "Spring":
                    planting = "Prime planting time for most plants"
                elif season == "Summer":
                    planting = "Good for heat-tolerant plants, ensure adequate watering"
                elif season == "Autumn":
                    planting = "Good for trees and shrubs, prepare for dormancy"
                else:
                    planting = "Indoor planting or dormant season preparations"

            is_growing = season in ["Spring", "Summer", "Rainy"]
            return SeasonData(season, month, day_of_year, is_growing, planting)

        except Exception as e:
            print(f"Error determining season: {e}")
            return SeasonData("Unknown", datetime.now().month, datetime.now().timetuple().tm_yday, True, "Season-appropriate planting recommended")

    @staticmethod
    def get_location_coordinates(city: str, country: str) -> tuple:
        coordinates = {
            "lucknow": (26.8467, 80.9462),
            "delhi": (28.7041, 77.1025),
            "mumbai": (19.0760, 72.8777),
            "bangalore": (12.9716, 77.5946),
            "london": (51.5074, -0.1278),
            "new york": (40.7128, -74.0060),
            "sydney": (-33.8688, 151.2093)
        }
        return coordinates.get(city.lower(), (26.8467, 80.9462))

# ------------------------------
# Plant Recommendation System
# ------------------------------
class PlantRecommendationSystem:
    def __init__(self, weatherapi_key: str, gemini_api_key: str):
        self.weather_service = WeatherService(weatherapi_key)
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        self.prompt_template = PromptTemplate(
            input_variables=[
                "temperature", "humidity", "weather_description", "wind_speed", 
                "precipitation", "feels_like", "uv_index", "visibility",
                "vegetation_coverage", "building_coverage", "road_coverage", 
                "empty_land", "water_body", "city", "country", "season", "planting_season"
            ],
            template="""
                You are an expert botanist and landscape designer. Based on the following environmental conditions, land usage patterns, and seasonal planting guidelines, generate a concise analytical report that recommends **3–5 plant species** suitable for this specific location.

Your recommendations should take into account:
- Climatic tolerance (temperature, UV index, humidity, wind, rainfall)
- Current weather condition and visibility
- Land coverage distribution (urban, semi-urban, rural mix)
- Seasonal compatibility (what thrives in the current season and planting window)
- Availability of water bodies and vegetation for support ecosystems
- Possibility of deploying different plants in different types of spaces (e.g., open fields, roadside greening, rooftop gardens, water-adjacent planting)
- keep in account what type of location it is like lucknow is a city so dont suggest to plant sugarcane or farm crops, in delhi you will suggest decorative plants and tree

Each recommendation must include:
- **Indian Name then in bracket english name like gulab (Rose)**
- A reasoned justification that connects environmental parameters to plant suitability
- Basic but practical care tips tailored to these conditions
- And in the **Care** section, include **in quotes** an estimate of the **number of plants that can be planted**, based on the available empty land percentage and general space requirement of that plant

---

**Location:** {city}, {country}

**Weather Conditions:**
- Temperature: {temperature}°C (Feels like: {feels_like}°C)
- Humidity: {humidity}%
- Weather: {weather_description}
- Wind Speed: {wind_speed} m/s
- Precipitation: {precipitation} mm
- UV Index: {uv_index}
- Visibility: {visibility} km

**Land Coverage:**
- Vegetation Coverage: {vegetation_coverage}%
- Building Coverage: {building_coverage}%
- Road Coverage: {road_coverage}%
- Empty Land Available: {empty_land}%
- Water Bodies: {water_body}%

**Season:** {season}  
**Planting Guidelines:** {planting_season}

---
Please provide your recommendations using the following format for each plant:

**Plant:** [Common Name] *(Scientific Name)*  
**Reason:** [Explain why this plant is suitable for the given weather, land, and seasonal factors]  
**Care:** [Include basic care tips: watering needs, sunlight preference, soil type, and maintenance level. At the end, include an estimate in quotes: e.g., "Approximately 50–60 plants can be planted per 1000 sq.m based on spacing requirements and current land availability."]

---
Note:
- Be context-aware — tailor each suggestion to the presence of water, percentage of built-up area, or wind exposure.
- Do not recommend invasive or non-native species unless explicitly suitable and controlled.
- Do not any thing other than the plant reason and care in the response and keep the proper format
            """
        )
        self.parser = PlantRecommendationParser()

# ------------------------------
# Final function with lat/lng support
# ------------------------------
def generate_final_report(
    coverage_details: Dict,
    city: str,
    country: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Dict:
    WEATHERAPI_KEY = os.getenv('WEATHERAPI_KEY')
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

    if not WEATHERAPI_KEY or not GEMINI_API_KEY:
        return {
            "status": "error",
            "message": "API keys not found. Please set WEATHERAPI_KEY and GOOGLE_API_KEY in your .env file.",
            "response": None,
        }

    if not coverage_details:
        return {
            "status": "error",
            "message": "Coverage details not provided.",
            "response": None,
        }

    try:
        land_coverage = LandCoverageData(
            vegetation_coverage=coverage_details.get("vegetation_coverage", 0.0),
            building_coverage=coverage_details.get("building_coverage", 0.0),
            road_coverage=coverage_details.get("road_coverage", 0.0),
            empty_land=coverage_details.get("empty_land", 0.0),
            water_body=coverage_details.get("water_body", 0.0),
        )

        total_coverage = (
            land_coverage.vegetation_coverage +
            land_coverage.building_coverage +
            land_coverage.road_coverage +
            land_coverage.empty_land +
            land_coverage.water_body
        )

        if abs(total_coverage - 100) > 1:
            return {
                "status": "error",
                "message": f"Land coverage percentages must sum to 100%. Provided sum: {total_coverage}%",
                "response": None,
            }

        if latitude is None or longitude is None:
            latitude, longitude = SeasonService.get_location_coordinates(city, country)

        plant_system = PlantRecommendationSystem(WEATHERAPI_KEY, GEMINI_API_KEY)

        weather_data = plant_system.weather_service.get_weather_data(latitude, longitude)
        if not weather_data:
            return {
                "status": "error",
                "message": "Failed to fetch weather data for the provided location.",
                "response": None,
            }

        season_data = SeasonService.determine_season(
            weather_data.local_time, weather_data.timezone, latitude, city, country
        )

        prompt = plant_system.prompt_template.format(
            temperature=weather_data.temperature,
            humidity=weather_data.humidity,
            weather_description=weather_data.weather_description,
            wind_speed=weather_data.wind_speed,
            precipitation=weather_data.precipitation,
            feels_like=weather_data.feels_like,
            uv_index=weather_data.uv_index,
            visibility=weather_data.visibility,
            vegetation_coverage=land_coverage.vegetation_coverage,
            building_coverage=land_coverage.building_coverage,
            road_coverage=land_coverage.road_coverage,
            empty_land=land_coverage.empty_land,
            water_body=land_coverage.water_body,
            city=city,
            country=country,
            season=season_data.season,
            planting_season=season_data.planting_season
        )

        response = plant_system.model.generate_content(prompt)
        parsed_result = plant_system.parser.parse(response.text)

        parsed_result['weather_data'] = weather_data.__dict__
        parsed_result['land_coverage'] = land_coverage.__dict__
        parsed_result['season'] = season_data.__dict__
        parsed_result['location'] = {
            'city': city,
            'country': country,
            'latitude': latitude,
            'longitude': longitude,
        }

        return {
            "status": "success",
            "message": "Plant recommendations generated successfully.",
            "response": parsed_result,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "response": None,
        }
