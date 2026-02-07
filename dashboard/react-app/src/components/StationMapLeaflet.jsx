import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  CircleMarker,
  Tooltip,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "../styles/leaflet-custom.css";
import {
  MapPin,
  Activity,
  Zap,
  TrendingUp,
  AlertCircle,
  Droplets,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import axios from "axios";

const API_BASE = "http://localhost:8000";

// Coordonnées réelles des stations au Burkina Faso
const STATION_COORDS = {
  OUG_ZOG: { lat: 12.3672, lng: -1.5339, city: "Ouagadougou", zone: "Zogona" },
  OUG_PIS: {
    lat: 12.3547,
    lng: -1.5081,
    city: "Ouagadougou",
    zone: "Paspanga",
  },
  BOBO_KUA: {
    lat: 11.177,
    lng: -4.2933,
    city: "Bobo-Dioulasso",
    zone: "Kuinima",
  },
  OUG_NAB: {
    lat: 12.3944,
    lng: -1.4897,
    city: "Ouagadougou",
    zone: "Nabitenga",
  },
  BOBO_DAR: {
    lat: 11.1858,
    lng: -4.3136,
    city: "Bobo-Dioulasso",
    zone: "Darsalamy",
  },
};

// Créer des icônes personnalisées pour chaque statut
const createCustomIcon = (status, isSelected) => {
  const colors = {
    excellent: "#10b981",
    good: "#eab308",
    warning: "#f59e0b",
    critical: "#ef4444",
  };

  const color = colors[status] || "#6b7280";
  const size = isSelected ? 40 : 30;
  const pulseClass = isSelected ? "animate-pulse" : "";

  return L.divIcon({
    className: "custom-icon",
    html: `
      <div class="relative ${pulseClass}">
        <div class="absolute inset-0 rounded-full animate-ping opacity-75" style="background-color: ${color};"></div>
        <div class="relative w-${size / 4} h-${size / 4} rounded-full border-4 border-white shadow-xl flex items-center justify-center" 
             style="background-color: ${color}; width: ${size}px; height: ${size}px;">
          <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
          </svg>
        </div>
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size],
    popupAnchor: [0, -size],
  });
};

export default function StationMapLeaflet({
  selectedStation,
  onStationSelect,
}) {
  const [stations, setStations] = useState([]);
  const [stationsData, setStationsData] = useState({});
  const [loading, setLoading] = useState(true);
  const [key, setKey] = useState(0); // Clé pour forcer le remount complet

  useEffect(() => {
    loadStations();
  }, []);

  // Forcer un nouveau mount à chaque fois que le composant est monté
  useEffect(() => {
    setKey((prev) => prev + 1);
  }, []);

  const loadStations = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stations`);
      setStations(response.data);

      // Charger les données de chaque station
      const dataPromises = response.data.map((station) =>
        axios
          .get(`${API_BASE}/station/${station.station_id}`)
          .then((res) => ({ [station.station_id]: res.data }))
          .catch(() => ({ [station.station_id]: null })),
      );

      const results = await Promise.all(dataPromises);
      const dataMap = Object.assign({}, ...results);
      setStationsData(dataMap);
      setLoading(false);
    } catch (err) {
      console.error("Erreur chargement stations:", err);
      setLoading(false);
    }
  };

  const getStationStatus = (stationId) => {
    const data = stationsData[stationId];
    if (!data || !data.stats)
      return { status: "unknown", color: "gray", label: "Inconnu" };

    const efficiency = data.stats.efficiency;
    const powerFactor = data.stats.power_factor;

    if (efficiency < 0.7 || powerFactor < 0.8) {
      return { status: "critical", color: "#ef4444", label: "Critique" };
    } else if (efficiency < 0.75 || powerFactor < 0.85) {
      return { status: "warning", color: "#f59e0b", label: "Attention" };
    } else if (efficiency < 0.8) {
      return { status: "good", color: "#eab308", label: "Bon" };
    } else {
      return { status: "excellent", color: "#10b981", label: "Excellent" };
    }
  };

  if (loading) {
    return (
      <Card className="h-[700px]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5" />
            Carte des Stations ONEA
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[600px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">⏳ Chargement de la carte...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Centre de la carte (Burkina Faso)
  const mapCenter = [12.3, -1.7];
  const mapZoom = 8;

  return (
    <Card className="h-[700px]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPin className="w-5 h-5 text-blue-600" />
          Carte Interactive des Stations ONEA
        </CardTitle>
        <p className="text-sm text-gray-600">
          Visualisation géographique des sites de production • Cliquez sur un
          marqueur pour voir les détails
        </p>
      </CardHeader>
      <CardContent>
        {/* Carte Leaflet */}
        <div className="relative h-[500px] rounded-lg overflow-hidden border-2 border-gray-200 shadow-lg">
          <MapContainer
            key={`map-${key}`} // Clé unique pour chaque instance
            center={mapCenter}
            zoom={mapZoom}
            style={{ height: "100%", width: "100%" }}
            scrollWheelZoom={true}
          >
            {/* Tuiles OpenStreetMap */}
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* Marqueurs des stations */}
            {stations.map((station) => {
              const coords = STATION_COORDS[station.station_id];
              if (!coords) return null;

              const statusInfo = getStationStatus(station.station_id);
              const data = stationsData[station.station_id];
              const isSelected = selectedStation === station.station_id;

              return (
                <Marker
                  key={station.station_id}
                  position={[coords.lat, coords.lng]}
                  icon={createCustomIcon(statusInfo.status, isSelected)}
                  eventHandlers={{
                    click: () => onStationSelect(station.station_id),
                  }}
                >
                  <Popup maxWidth={300} className="custom-popup">
                    <div className="p-2">
                      {/* En-tête */}
                      <div className="flex items-start justify-between mb-3 pb-2 border-b">
                        <div className="flex-1">
                          <h3 className="font-bold text-base text-gray-900">
                            {station.name}
                          </h3>
                          <p className="text-xs text-gray-600 mt-0.5">
                            📍 {station.location} • {coords.city}
                          </p>
                          <p className="text-xs text-gray-500 mt-0.5">
                            Zone: {coords.zone}
                          </p>
                        </div>
                        <Badge
                          style={{
                            backgroundColor: statusInfo.color,
                            color: "white",
                            fontSize: "0.7rem",
                          }}
                        >
                          {statusInfo.label}
                        </Badge>
                      </div>

                      {/* Statistiques */}
                      {data && data.stats && (
                        <div className="space-y-2 mb-3">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1.5 text-gray-700">
                              <Activity className="w-3.5 h-3.5 text-green-600" />
                              <span className="font-medium">Efficacité</span>
                            </span>
                            <span className="font-bold text-gray-900">
                              {(data.stats.efficiency * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1.5 text-gray-700">
                              <Zap className="w-3.5 h-3.5 text-yellow-600" />
                              <span className="font-medium">
                                Facteur Puissance
                              </span>
                            </span>
                            <span className="font-bold text-gray-900">
                              {data.stats.power_factor.toFixed(2)}
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1.5 text-gray-700">
                              <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
                              <span className="font-medium">
                                Consommation moy.
                              </span>
                            </span>
                            <span className="font-bold text-gray-900">
                              {data.stats.avg_consumption.toFixed(0)} kWh
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1.5 text-gray-700">
                              <Droplets className="w-3.5 h-3.5 text-cyan-600" />
                              <span className="font-medium">
                                Niveau Réservoir
                              </span>
                            </span>
                            <span className="font-bold text-gray-900">
                              {data.stats.reservoir_level?.toFixed(0) || "N/A"}%
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Bouton action */}
                      <button
                        onClick={() => onStationSelect(station.station_id)}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold py-2 px-3 rounded-md transition-colors shadow-sm"
                      >
                        📊 Voir le tableau de bord
                      </button>
                    </div>
                  </Popup>

                  {/* Tooltip au survol */}
                  <Tooltip direction="top" offset={[0, -10]} opacity={0.9}>
                    <div className="text-center">
                      <div className="font-bold text-sm">{station.name}</div>
                      <div className="text-xs">
                        {coords.city} • {coords.zone}
                      </div>
                    </div>
                  </Tooltip>
                </Marker>
              );
            })}

            {/* Cercles pour les zones urbaines */}
            <CircleMarker
              center={[12.37, -1.52]}
              radius={30}
              pathOptions={{
                color: "#3b82f6",
                fillColor: "#3b82f6",
                fillOpacity: 0.1,
                weight: 2,
                dashArray: "5, 5",
              }}
            >
              <Tooltip
                permanent
                direction="center"
                className="text-sm font-semibold"
              >
                Ouagadougou
              </Tooltip>
            </CircleMarker>

            <CircleMarker
              center={[11.18, -4.29]}
              radius={25}
              pathOptions={{
                color: "#06b6d4",
                fillColor: "#06b6d4",
                fillOpacity: 0.1,
                weight: 2,
                dashArray: "5, 5",
              }}
            >
              <Tooltip
                permanent
                direction="center"
                className="text-sm font-semibold"
              >
                Bobo-Dioulasso
              </Tooltip>
            </CircleMarker>
          </MapContainer>
        </div>

        {/* Légende */}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-xs font-semibold text-gray-700">
              État des stations:
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-xs text-gray-600">Excellent</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <span className="text-xs text-gray-600">Bon</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                <span className="text-xs text-gray-600">Attention</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span className="text-xs text-gray-600">Critique</span>
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">
            🗺️ Carte interactive • Zoom: Molette • Navigation: Glisser
          </div>
        </div>

        {/* Statistiques globales */}
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="text-xs font-semibold text-gray-700">
                Opérationnelles
              </span>
            </div>
            <p className="text-2xl font-bold text-green-700">
              {
                stations.filter((s) => {
                  const status = getStationStatus(s.station_id);
                  return (
                    status.status === "excellent" || status.status === "good"
                  );
                }).length
              }
            </p>
            <p className="text-xs text-gray-600 mt-0.5">Stations actives</p>
          </div>

          <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <AlertCircle className="w-3 h-3 text-orange-500" />
              <span className="text-xs font-semibold text-gray-700">
                Surveillance
              </span>
            </div>
            <p className="text-2xl font-bold text-orange-700">
              {
                stations.filter(
                  (s) => getStationStatus(s.station_id).status === "warning",
                ).length
              }
            </p>
            <p className="text-xs text-gray-600 mt-0.5">À surveiller</p>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-1">
              <AlertCircle className="w-3 h-3 text-red-500" />
              <span className="text-xs font-semibold text-gray-700">
                Critiques
              </span>
            </div>
            <p className="text-2xl font-bold text-red-700">
              {
                stations.filter(
                  (s) => getStationStatus(s.station_id).status === "critical",
                ).length
              }
            </p>
            <p className="text-xs text-gray-600 mt-0.5">Intervention urgente</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
