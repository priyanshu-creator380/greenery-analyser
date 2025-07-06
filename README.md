# 🌱 Satellite-Based Plant Recommendation System

This project is an intelligent system that analyzes environmental and land coverage data — including weather conditions and satellite imagery-derived land features — to recommend suitable plants for a given location. It combines satellite image interpretation, live weather data, seasonal insights, and AI-generated recommendations to support landscape planning, agriculture, and urban greening efforts.

---

## 🔍 Overview

The system performs the following key tasks:

- 📡 **Integrates WeatherAPI** for real-time temperature, humidity, wind speed, UV index, and other environmental factors.
- 🗺️ **Extracts Land Coverage Data** from satellite or drone imagery to compute:
  - Plant coverage
  - Building coverage
  - Road coverage
  - Empty land
  - Water bodies
- 🌦️ **Determines Current Season** based on location, latitude, and date to assess whether it’s a growing or dormant season.
- 🤖 **Generates Plant Recommendations** using Google's Gemini AI model, considering all of the above data.
- ✅ Provides structured results with reasons and care instructions for each recommended plant.

---

## 🧠 Technology Stack

- **Python 3.10+**
- **Google Gemini (via `google.generativeai`)**
- **LangChain** for prompt templates and parsing
- **WeatherAPI** for environmental data
- **PIL / OpenCV** *(optional, for land coverage from images)*
- **Dotenv** for API key management

---