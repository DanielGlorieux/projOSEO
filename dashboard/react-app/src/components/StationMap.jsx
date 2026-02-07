import { useEffect, useState } from 'react'
import { MapPin, Activity, Zap, TrendingUp, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

// Coordonnées des stations (Burkina Faso - approximatives)
const STATION_COORDS = {
  'OUG_ZOG': { lat: 12.3672, lng: -1.5339, city: 'Ouagadougou' },
  'OUG_PIS': { lat: 12.3547, lng: -1.5081, city: 'Ouagadougou' },
  'BOBO_KUA': { lat: 11.1770, lng: -4.2933, city: 'Bobo-Dioulasso' },
  'OUG_NAB': { lat: 12.3944, lng: -1.4897, city: 'Ouagadougou' },
  'BOBO_DAR': { lat: 11.1858, lng: -4.3136, city: 'Bobo-Dioulasso' }
}

export default function StationMap({ selectedStation, onStationSelect }) {
  const [stations, setStations] = useState([])
  const [stationsData, setStationsData] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStations()
  }, [])

  const loadStations = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stations`)
      setStations(response.data)
      
      // Charger les données de chaque station
      const dataPromises = response.data.map(station =>
        axios.get(`${API_BASE}/station/${station.station_id}`)
          .then(res => ({ [station.station_id]: res.data }))
          .catch(() => ({ [station.station_id]: null }))
      )
      
      const results = await Promise.all(dataPromises)
      const dataMap = Object.assign({}, ...results)
      setStationsData(dataMap)
      setLoading(false)
    } catch (err) {
      console.error('Erreur chargement stations:', err)
      setLoading(false)
    }
  }

  const getStationStatus = (stationId) => {
    const data = stationsData[stationId]
    if (!data || !data.stats) return { status: 'unknown', color: 'gray' }
    
    const efficiency = data.stats.efficiency
    const powerFactor = data.stats.power_factor
    
    if (efficiency < 0.70 || powerFactor < 0.80) {
      return { status: 'critical', color: 'red', label: 'Critique' }
    } else if (efficiency < 0.75 || powerFactor < 0.85) {
      return { status: 'warning', color: 'orange', label: 'Attention' }
    } else if (efficiency < 0.80) {
      return { status: 'good', color: 'yellow', label: 'Bon' }
    } else {
      return { status: 'excellent', color: 'green', label: 'Excellent' }
    }
  }

  const getMarkerColor = (status) => {
    switch (status) {
      case 'critical': return '#ef4444'
      case 'warning': return '#f59e0b'
      case 'good': return '#eab308'
      case 'excellent': return '#10b981'
      default: return '#6b7280'
    }
  }

  if (loading) {
    return (
      <Card className="h-[600px]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5" />
            Carte des Stations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[500px]">
            <p className="text-gray-500">⏳ Chargement de la carte...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-[600px]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPin className="w-5 h-5" />
          Carte des Stations ONEA
        </CardTitle>
        <p className="text-sm text-gray-600">
          Visualisation géographique des stations de pompage
        </p>
      </CardHeader>
      <CardContent>
        {/* Carte simplifiée avec markers */}
        <div className="relative w-full h-[450px] bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border-2 border-blue-200 overflow-hidden">
          {/* Grille de fond */}
          <div className="absolute inset-0" style={{
            backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
                             linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)`,
            backgroundSize: '40px 40px'
          }}></div>
          
          {/* Titre Burkina Faso */}
          <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm px-4 py-2 rounded-lg shadow-lg border border-blue-200">
            <h3 className="font-bold text-lg text-blue-900">🇧🇫 Burkina Faso</h3>
            <p className="text-xs text-gray-600">Stations de pompage ONEA</p>
          </div>

          {/* Légende */}
          <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border border-blue-200">
            <p className="text-xs font-semibold mb-2">État des stations:</p>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span>Excellent</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <span>Bon</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                <span>Attention</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span>Critique</span>
              </div>
            </div>
          </div>

          {/* Markers des stations */}
          {stations.map((station) => {
            const coords = STATION_COORDS[station.station_id]
            if (!coords) return null

            // Conversion coordonnées géographiques en position sur la carte
            // Simplifié pour démo (normalisation approximative)
            const x = ((coords.lng + 5) / 4) * 100 // Normaliser longitude
            const y = ((15 - coords.lat) / 5) * 100 // Normaliser latitude

            const statusInfo = getStationStatus(station.station_id)
            const data = stationsData[station.station_id]
            const isSelected = selectedStation === station.station_id

            return (
              <div
                key={station.station_id}
                className="absolute transition-all cursor-pointer hover:z-50"
                style={{
                  left: `${x}%`,
                  top: `${y}%`,
                  transform: 'translate(-50%, -50%)'
                }}
                onClick={() => onStationSelect(station.station_id)}
              >
                {/* Marker */}
                <div className={`relative ${isSelected ? 'scale-125' : ''} transition-transform`}>
                  {/* Pulse animation pour station sélectionnée */}
                  {isSelected && (
                    <div className="absolute inset-0 animate-ping">
                      <div
                        className="w-8 h-8 rounded-full opacity-75"
                        style={{ backgroundColor: getMarkerColor(statusInfo.status) }}
                      ></div>
                    </div>
                  )}

                  {/* Marker principal */}
                  <div
                    className="relative w-8 h-8 rounded-full border-4 border-white shadow-xl flex items-center justify-center"
                    style={{ backgroundColor: getMarkerColor(statusInfo.status) }}
                  >
                    <MapPin className="w-4 h-4 text-white" fill="white" />
                  </div>

                  {/* Tooltip */}
                  <div className={`absolute left-10 top-0 ${isSelected ? 'block' : 'hidden group-hover:block'} z-50`}>
                    <div className="bg-white rounded-lg shadow-2xl border-2 border-gray-200 p-3 w-64">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <h4 className="font-bold text-sm">{station.name}</h4>
                          <p className="text-xs text-gray-600">{station.location}</p>
                          <p className="text-xs text-gray-500">{coords.city}</p>
                        </div>
                        <Badge
                          className="text-xs"
                          style={{
                            backgroundColor: getMarkerColor(statusInfo.status),
                            color: 'white'
                          }}
                        >
                          {statusInfo.label}
                        </Badge>
                      </div>

                      {data && data.stats && (
                        <div className="space-y-2 mt-3 pt-3 border-t border-gray-200">
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-gray-600">
                              <Activity className="w-3 h-3" />
                              Efficacité
                            </span>
                            <span className="font-semibold">
                              {(data.stats.efficiency * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-gray-600">
                              <Zap className="w-3 h-3" />
                              Facteur Puissance
                            </span>
                            <span className="font-semibold">
                              {data.stats.power_factor.toFixed(2)}
                            </span>
                          </div>

                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-1 text-gray-600">
                              <TrendingUp className="w-3 h-3" />
                              Consommation moy.
                            </span>
                            <span className="font-semibold">
                              {data.stats.avg_consumption.toFixed(0)} kWh
                            </span>
                          </div>
                        </div>
                      )}

                      <button
                        onClick={() => onStationSelect(station.station_id)}
                        className="mt-3 w-full bg-blue-600 hover:bg-blue-700 text-white text-xs py-1.5 rounded-md transition-colors"
                      >
                        Voir les détails
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}

          {/* Zones géographiques */}
          <div className="absolute bottom-20 left-20 bg-blue-600/10 backdrop-blur-sm px-4 py-2 rounded-full border border-blue-300">
            <p className="text-sm font-semibold text-blue-800">Ouagadougou</p>
          </div>

          <div className="absolute bottom-32 right-32 bg-cyan-600/10 backdrop-blur-sm px-4 py-2 rounded-full border border-cyan-300">
            <p className="text-sm font-semibold text-cyan-800">Bobo-Dioulasso</p>
          </div>
        </div>

        {/* Statistiques globales */}
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-xs font-semibold text-gray-700">Opérationnelles</span>
            </div>
            <p className="text-2xl font-bold text-green-700 mt-1">
              {stations.filter(s => {
                const status = getStationStatus(s.station_id)
                return status.status === 'excellent' || status.status === 'good'
              }).length}
            </p>
          </div>

          <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-3 h-3 text-orange-500" />
              <span className="text-xs font-semibold text-gray-700">Surveillance</span>
            </div>
            <p className="text-2xl font-bold text-orange-700 mt-1">
              {stations.filter(s => getStationStatus(s.station_id).status === 'warning').length}
            </p>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-3 h-3 text-red-500" />
              <span className="text-xs font-semibold text-gray-700">Critiques</span>
            </div>
            <p className="text-2xl font-bold text-red-700 mt-1">
              {stations.filter(s => getStationStatus(s.station_id).status === 'critical').length}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
