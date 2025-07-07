import React, { useState, useEffect, useRef } from 'react';
import "./App.css"

const App = () => {
  const [currentView, setCurrentView] = useState('map');
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [errorPopup, setErrorPopup] = useState({ show: false, title: '', message: '' });
  const [staticMapImagePath, setStaticMapImagePath] = useState(null);
  
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  // Function to handle view changes and map refresh
  const handleViewChange = (newView) => {
    setCurrentView(newView);
    
    // If switching back to map view, refresh the map
    if (newView === 'map' && mapInstanceRef.current) {
      // Small delay to ensure DOM is updated
      setTimeout(() => {
        // Trigger resize event to re-render the map
        window.google.maps.event.trigger(mapInstanceRef.current, 'resize');
        
        // Optionally re-center the map
        if (selectedLocation) {
          mapInstanceRef.current.setCenter({ 
            lat: selectedLocation.lat, 
            lng: selectedLocation.lng 
          });
        } else {
          mapInstanceRef.current.setCenter({ lat: 28.6139, lng: 77.2090 });
        }
      }, 100);
    }
  };

  // Notification system
  const addNotification = (type, title, message) => {
    const id = Date.now().toString();
    const notification = { id, type, title, message };
    setNotifications(prev => [...prev, notification]);
    
    if (type !== 'loading') {
      setTimeout(() => {
        removeNotification(id);
      }, 5000);
    }
    
    return id;
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const showErrorPopup = (title, message) => {
    setErrorPopup({ show: true, title, message });
  };

  const closeErrorPopup = () => {
    setErrorPopup({ show: false, title: '', message: '' });
  };

  // Initialize Google Maps
  const initMap = () => {
    if (!window.google || !mapRef.current) return;

    const map = new window.google.maps.Map(mapRef.current, {
      center: { lat: 28.6139, lng: 77.2090 },
      zoom: 10,
      mapTypeId: 'hybrid',
      styles: [
        {
          featureType: "all",
          elementType: "geometry",
          stylers: [{ color: "#1e293b" }]
        },
        {
          featureType: "all",
          elementType: "labels.text.fill",
          stylers: [{ color: "#ffffff" }]
        },
        {
          featureType: "water",
          elementType: "geometry",
          stylers: [{ color: "#0f172a" }]
        },
        {
          featureType: "landscape",
          elementType: "geometry",
          stylers: [{ color: "#334155" }]
        },
        {
          featureType: "poi.park",
          elementType: "geometry",
          stylers: [{ color: "#10b981" }]
        }
      ],
      disableDefaultUI: false,
      zoomControl: true,
      mapTypeControl: true,
      streetViewControl: false,
      fullscreenControl: true
    });

    mapInstanceRef.current = map;

    map.addListener("click", (event) => {
      if (event.latLng) {
        handleMapClick(event.latLng);
      }
    });

    setMapLoaded(true);
    addNotification('info', 'Map Ready', 'Click anywhere on the map to analyze that location');
  };

  // Handle map click
  const handleMapClick = async (latLng) => {
    if (isLoading) return;

    const lat = latLng.lat();
    const lng = latLng.lng();

    // Remove existing marker
    if (markerRef.current) {
      markerRef.current.setMap(null);
    }

    // Add new marker
    markerRef.current = new window.google.maps.Marker({
      position: { lat, lng },
      map: mapInstanceRef.current,
      animation: window.google.maps.Animation.DROP,
      icon: {
        path: window.google.maps.SymbolPath.CIRCLE,
        scale: 10,
        fillColor: "#10b981",
        fillOpacity: 1,
        strokeColor: "#ffffff",
        strokeWeight: 2
      }
    });

    // Smooth zoom and pan to the clicked location
    const targetZoom = 15; // Zoom level for detailed view
    
    // Animate to the new position
    mapInstanceRef.current.panTo({ lat, lng });
    
    // Smooth zoom transition
    setTimeout(() => {
      mapInstanceRef.current.setZoom(targetZoom);
    }, 500); // Delay zoom slightly for smoother experience

    const loadingId = addNotification('loading', 'Analyzing Location', 'Getting weather data and plant recommendations...');
    setIsLoading(true);

    try {
      const geocoder = new window.google.maps.Geocoder();
      const geocodeResponse = await geocoder.geocode({ location: { lat, lng } });

      let city = "Unknown City";
      let country = "Unknown Country";

      if (geocodeResponse.results && geocodeResponse.results.length > 0) {
        for (const component of geocodeResponse.results[0].address_components) {
          if (component.types.includes("locality")) {
            city = component.long_name;
          }
          if (component.types.includes("country")) {
            country = component.long_name;
          }
        }
      }

      setSelectedLocation({ lat, lng, city, country });

      const formData = new FormData();
      formData.append("longitude", lng);
      formData.append("latitude", lat);
      formData.append("city", city);
      formData.append("country", country);

      const response = await fetch(`${import.meta.env.VITE_BACKEND}/analyze-location/`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (!result || result.status !== "success" || !result.final_report || !result.final_report.response) {
        throw new Error("Invalid response format from server");
      }

      const responseData = result.final_report.response;

      if (!responseData.weather_data || !responseData.land_coverage || !responseData.season || !responseData.location || !responseData.recommendations) {
        throw new Error("Missing required data in response");
      }

      setStaticMapImagePath(result.file_path || null);
      setDashboardData(responseData);

      removeNotification(loadingId);
      addNotification('success', 'Analysis Complete', `Found data for ${city}, ${country}. Opening dashboard...`);

      setTimeout(() => {
        setCurrentView('dashboard');
      }, 1500);

    } catch (error) {
      console.error("Error analyzing location:", error);
      removeNotification(loadingId);

      let errorTitle = "Analysis Failed";
      let errorMessage = "Could not analyze this location. Please try another area.";

      if (error.message.includes("HTTP error")) {
        errorTitle = "Server Error";
        errorMessage = "The analysis server is not responding. Please check if the backend service is running.";
      }

      showErrorPopup(errorTitle, errorMessage);
      addNotification('error', errorTitle, errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Load Google Maps API
  useEffect(() => {
    if (window.google) {
      initMap();
      return;
    }

    const script = document.createElement('script');
    const API_KEY = import.meta.env.VITE_API_KEY;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${API_KEY}&libraries=places&callback=initGoogleMaps`;
    script.async = true;
    script.defer = true;
    
    window.initGoogleMaps = initMap;
    
    script.onerror = () => {
      addNotification('error', 'Map Loading Error', 'Failed to load Google Maps API. Please check your API key.');
    };
    
    document.head.appendChild(script);

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
    };
  }, []);

  // Effect to handle map resize when view changes
  useEffect(() => {
    if (currentView === 'map' && mapInstanceRef.current && window.google) {
      const timer = setTimeout(() => {
        window.google.maps.event.trigger(mapInstanceRef.current, 'resize');
      }, 200);
      
      return () => clearTimeout(timer);
    }
  }, [currentView]);

  const NotificationIcon = ({ type }) => {
    switch (type) {
      case 'loading':
        return <div className="w-5 h-5 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></div>;
      case 'success':
        return <div className="text-green-500 text-xl">✓</div>;
      case 'error':
        return <div className="text-red-500 text-xl">⚠</div>;
      case 'info':
        return <div className="text-blue-500 text-xl">📍</div>;
      default:
        return null;
    }
  };

  const ProgressBar = ({ percentage, type }) => {
    const colorClasses = {
      vegetation: 'bg-gradient-to-r from-green-500 to-green-400',
      building: 'bg-gradient-to-r from-slate-500 to-slate-400',
      road: 'bg-gradient-to-r from-gray-500 to-gray-400',
      empty: 'bg-gradient-to-r from-gray-300 to-gray-200',
      water: 'bg-gradient-to-r from-blue-500 to-blue-400'
    };

    return (
      <div className="h-2 bg-white/10 rounded-full overflow-hidden border border-white/10">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${colorClasses[type]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white relative overflow-hidden">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(16,185,129,0.1)_0%,transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(52,211,153,0.1)_0%,transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_40%_40%,rgba(132,204,22,0.05)_0%,transparent_50%)]" />
      </div>

      {/* Error Popup */}
      {errorPopup.show && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white/5 backdrop-blur-xl border border-red-500/50 rounded-2xl p-8 max-w-md w-full text-center transform transition-all duration-300">
            <div className="text-6xl text-red-500 mb-4">⚠️</div>
            <h2 className="text-2xl font-bold mb-4">{errorPopup.title}</h2>
            <p className="text-slate-300 mb-6 leading-relaxed">{errorPopup.message}</p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={closeErrorPopup}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors duration-200"
              >
                Try Again
              </button>
              <button
                onClick={closeErrorPopup}
                className="px-6 py-3 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg font-semibold transition-colors duration-200"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Notifications */}
      <div className="fixed top-4 right-4 z-40 flex flex-col gap-2 max-w-sm">
        {notifications.map((notification) => (
          <div
            key={notification.id}
            className={`bg-white/5 backdrop-blur-xl border rounded-xl p-4 min-w-80 flex items-start gap-3 transform transition-all duration-500 ${
              notification.type === 'success' ? 'border-green-500/50 bg-green-500/10' :
              notification.type === 'error' ? 'border-red-500/50 bg-red-500/10' :
              notification.type === 'info' ? 'border-blue-500/50 bg-blue-500/10' :
              'border-amber-500/50 bg-amber-500/10'
            }`}
          >
            <NotificationIcon type={notification.type} />
            <div className="flex-1">
              <h4 className="font-semibold text-sm mb-1">{notification.title}</h4>
              <p className="text-xs text-slate-300">{notification.message}</p>
            </div>
            {notification.type !== 'loading' && (
              <button
                onClick={() => removeNotification(notification.id)}
                className="text-slate-400 hover:text-white transition-colors text-lg leading-none"
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      {/* Map View */}
      {currentView === 'map' && (
        <div className="absolute inset-0 transition-all duration-1000 ease-in-out">
          <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-slate-900/90 to-transparent p-6 text-center">
            <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-green-500 via-green-400 to-green-300 bg-clip-text text-transparent">
              🌱 Green Dashboard
            </h1>
            <p className="text-slate-300 text-lg">Click anywhere on the map to analyze plant conditions</p>
          </div>

          <div ref={mapRef} className="w-full h-full" />

          {!mapLoaded && (
            <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center flex-col gap-4">
              <div className="w-12 h-12 border-3 border-green-500/30 border-t-green-500 rounded-full animate-spin" />
              <p className="text-xl text-white">Loading Map...</p>
              <p className="text-sm text-slate-300">Preparing your plant analysis tool</p>
            </div>
          )}

          {mapLoaded && (
            <div className="absolute bottom-6 left-6 right-6 bg-white/5 backdrop-blur-xl border border-white/20 rounded-xl p-4 flex items-center gap-3">
              <div className="text-green-500 text-2xl">📍</div>
              <div>
                <h3 className="font-semibold mb-1">Ready to Analyze</h3>
                <p className="text-sm text-slate-300">Click on any location to get weather data and plant recommendations</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Dashboard View */}
      {currentView === 'dashboard' && dashboardData && selectedLocation && (
        <div className="absolute inset-0 transition-all duration-1000 ease-in-out overflow-y-auto p-4">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <header className="text-center mb-8">
              <div className="flex items-center justify-center gap-4 mb-4">
                <button
                  onClick={() => handleViewChange('map')}
                  className="px-4 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-sm font-medium transition-colors duration-200"
                >
                  ← Back to Map
                </button>
                <h1 className="text-5xl font-bold bg-gradient-to-r from-green-500 via-green-400 to-green-300 bg-clip-text text-transparent">
                  🌱 Green Dashboard
                </h1>
              </div>
              <p className="text-slate-300 text-lg">
                Analysis for {selectedLocation.city}, {selectedLocation.country}
              </p>
            </header>

            {/* Dashboard Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Combined Weather & Season Card */}
              <div className="lg:col-span-4 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-green-500 via-green-400 to-green-300" />
                
                {/* Weather Section */}
                <div className="mb-6 pb-6 border-b border-white/10">
                  <div className="flex items-center gap-2 mb-4 text-lg font-semibold bg-gradient-to-r from-green-500 to-green-400 bg-clip-text text-transparent">
                    🌤️ Current Weather
                  </div>
                  <div className="text-center mb-6">
                    <div className="text-5xl font-bold mb-2">{dashboardData.weather_data.temperature.toFixed(1)}°C</div>
                    <div className="text-lg text-slate-300">{dashboardData.weather_data.weather_description}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/10 rounded-lg p-3 border border-white/10">
                      <div className="font-semibold">{dashboardData.weather_data.humidity.toFixed(0)}%</div>
                      <div className="text-xs text-slate-400">Humidity</div>
                    </div>
                    <div className="bg-white/10 rounded-lg p-3 border border-white/10">
                      <div className="font-semibold">{dashboardData.weather_data.wind_speed.toFixed(1)} m/s</div>
                      <div className="text-xs text-slate-400">Wind</div>
                    </div>
                    <div className="bg-white/10 rounded-lg p-3 border border-white/10">
                      <div className="font-semibold">{dashboardData.weather_data.feels_like.toFixed(1)}°C</div>
                      <div className="text-xs text-slate-400">Feels Like</div>
                    </div>
                    <div className="bg-white/10 rounded-lg p-3 border border-white/10">
                      <div className="font-semibold">{dashboardData.weather_data.uv_index.toFixed(1)}</div>
                      <div className="text-xs text-slate-400">UV Index</div>
                    </div>
                  </div>
                </div>

                {/* Season Section */}
                <div>
                  <div className="flex items-center gap-2 mb-4 text-lg font-semibold bg-gradient-to-r from-green-500 to-green-400 bg-clip-text text-transparent">
                    ☀️ Season & Growing
                  </div>
                  <div className="text-center mb-4">
                    <div className="text-3xl font-bold mb-2">{dashboardData.season.season}</div>
                    <div className="text-lg text-slate-300">
                      {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][dashboardData.season.month - 1]}
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-center">
                      <div className={`inline-block px-4 py-2 rounded-full font-semibold border ${
                        dashboardData.season.is_growing_season 
                          ? 'bg-green-500/20 border-green-500/50 text-green-400' 
                          : 'bg-amber-500/20 border-amber-500/50 text-amber-400'
                      }`}>
                        Growing Season: {dashboardData.season.is_growing_season ? 'Yes' : 'No'}
                      </div>
                    </div>
                    <div className="text-slate-300 text-sm leading-relaxed p-3 bg-white/5 rounded-lg border border-white/10">
                      {dashboardData.season.planting_season}
                    </div>
                  </div>
                </div>
              </div>

              {/* Satellite Image Card */}
              <div className="lg:col-span-5 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-green-500 via-green-400 to-green-300" />
                <div className="aspect-square rounded-xl overflow-hidden">
                  <img
                    src={staticMapImagePath ? `http://localhost:8000/${staticMapImagePath}` : "https://images.unsplash.com/photo-1446329813274-7c9036bd9a1f?w=800&h=600&fit=crop&crop=center"}
                    alt="Satellite view"
                    className="w-full h-full object-cover brightness-90 contrast-110"
                  />
                </div>
              </div>

              {/* Location & Land Coverage Card */}
              <div className="lg:col-span-3 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-green-500 via-green-400 to-green-300" />
                
                <div className="mb-6 pb-6 border-b border-white/10">
                  <div className="flex items-center gap-2 mb-4 text-lg font-semibold bg-gradient-to-r from-green-500 to-green-400 bg-clip-text text-transparent">
                    📍 Location
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg border border-white/10">
                      <span className="text-slate-400">City:</span>
                      <span className="font-semibold">{dashboardData.location.city}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg border border-white/10">
                      <span className="text-slate-400">Country:</span>
                      <span className="font-semibold">{dashboardData.location.country}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg border border-white/10">
                      <span className="text-slate-400">Coordinates:</span>
                      <span className="font-semibold text-sm">{selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}</span>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-4 text-lg font-semibold bg-gradient-to-r from-green-500 to-green-400 bg-clip-text text-transparent">
                    🏞️ Land Coverage
                  </div>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">🌿 Vegetation</span>
                        <span className="text-sm font-semibold">{dashboardData.land_coverage.vegetation_coverage.toFixed(0)}%</span>
                      </div>
                      <ProgressBar percentage={dashboardData.land_coverage.vegetation_coverage} type="vegetation" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">🏢 Buildings</span>
                        <span className="text-sm font-semibold">{dashboardData.land_coverage.building_coverage.toFixed(0)}%</span>
                      </div>
                      <ProgressBar percentage={dashboardData.land_coverage.building_coverage} type="building" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">🛣️ Roads</span>
                        <span className="text-sm font-semibold">{dashboardData.land_coverage.road_coverage.toFixed(0)}%</span>
                      </div>
                      <ProgressBar percentage={dashboardData.land_coverage.road_coverage} type="road" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm">🌾 Empty Land</span>
                        <span className="text-sm font-semibold">{dashboardData.land_coverage.empty_land.toFixed(0)}%</span>
                      </div>
                      <ProgressBar percentage={dashboardData.land_coverage.empty_land} type="empty" />
                    </div>
                    {dashboardData.land_coverage.water_body && dashboardData.land_coverage.water_body > 0 && (
                      <div>
                        <div className="flex justify-between mb-2">
                          <span className="text-sm">💧 Water Body</span>
                          <span className="text-sm font-semibold">{dashboardData.land_coverage.water_body.toFixed(0)}%</span>
                        </div>
                        <ProgressBar percentage={dashboardData.land_coverage.water_body} type="water" />
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Plant Recommendations Card */}
              <div className="lg:col-span-12 bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-green-500 via-green-400 to-green-300" />
                <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-green-500 to-green-400 bg-clip-text text-transparent">
                  🌺 Plant Recommendations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {dashboardData.recommendations && dashboardData.recommendations.length > 0 ? (
                    dashboardData.recommendations.map((plant, index) => (
                      <div key={index} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4 hover:bg-green-500/10 hover:border-green-500/30 transition-all duration-300 hover:-translate-y-1">
                        <div className="font-semibold text-lg mb-2">{plant.name.replace(/\*([^*]+)\*/g, '$1').trim()}</div>
                        <div className="text-slate-300 mb-3 leading-relaxed">{plant.reason}</div>
                        {plant.care && (
                          <div className="bg-white/5 rounded-lg p-3 border-l-3 border-green-500">
                            <div className="font-semibold text-slate-300 mb-1">Care Instructions:</div>
                            <div className="text-sm text-slate-400 leading-relaxed">{plant.care}</div>
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="col-span-full bg-white/5 rounded-xl p-6 text-center">
                      <div className="text-lg font-semibold mb-2">No Recommendations Available</div>
                      <div className="text-slate-300">No plant recommendations were provided for this location.</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;