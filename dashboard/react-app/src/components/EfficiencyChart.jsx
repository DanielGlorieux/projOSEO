import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useEffect, useState } from 'react'
import axios from 'axios'

export default function EfficiencyChart({ stationId = 'OUG_ZOG' }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Charger les données horaires d'efficacité depuis le nouvel endpoint
        const response = await axios.get(`http://localhost:8000/analytics/hourly-efficiency/${stationId}`)
        const efficiencyData = response.data
        
        // Les données sont déjà formatées par l'API
        setData(efficiencyData.hourly_data)
        setStats({
          efficiency: efficiencyData.stats.efficiency.toFixed(1),
          powerFactor: efficiencyData.stats.power_factor.toFixed(2),
          reservoir: efficiencyData.stats.reservoir.toFixed(1)
        })
        
        setError(null)
      } catch (err) {
        console.error('Erreur chargement données efficacité:', err)
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [stationId])

  if (loading) {
    return (
      <div className="chart-card">
        <h2>⚙️ Performance et Efficacité</h2>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <p>⏳ Chargement des données de performance...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="chart-card">
        <h2>⚙️ Performance et Efficacité</h2>
        <div style={{ textAlign: 'center', padding: '50px', color: 'red' }}>
          <p>❌ Erreur: {error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <h2>⚙️ Performance et Efficacité</h2>
      <div style={{ marginBottom: '30px' }}>
        <h3 style={{ fontSize: '1rem', color: '#666', marginBottom: '10px' }}>Efficacité Pompes (%)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis domain={[60, 100]} />
            <Tooltip formatter={(value) => `${value}%`} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="efficiency" 
              stroke="#9467bd" 
              strokeWidth={2}
              name="Efficacité (%)"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 style={{ fontSize: '1rem', color: '#666', marginBottom: '10px' }}>Niveau Réservoir (%)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis domain={[0, 100]} />
            <Tooltip formatter={(value) => `${value}%`} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="reservoir" 
              stroke="#17becf" 
              strokeWidth={2}
              fill="#17becf"
              fillOpacity={0.3}
              name="Niveau Réservoir (%)"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
        <div style={{ padding: '15px', background: '#f3f4f6', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#667eea' }}>{stats.efficiency}%</div>
          <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px' }}>Efficacité Moyenne</div>
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '2px' }}>24h moyennes</div>
        </div>
        <div style={{ padding: '15px', background: '#f3f4f6', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#10b981' }}>{stats.powerFactor}</div>
          <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px' }}>Facteur Puissance</div>
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '2px' }}>Moyenne réelle</div>
        </div>
        <div style={{ padding: '15px', background: '#f3f4f6', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#17becf' }}>{stats.reservoir}%</div>
          <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '5px' }}>Niveau Réservoir</div>
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '2px' }}>Moyenne actuelle</div>
        </div>
      </div>
      
      <div style={{ marginTop: '10px', padding: '10px', background: '#f9fafb', borderRadius: '6px', fontSize: '0.85em', color: '#666' }}>
        <strong>📊 Source:</strong> Données CSV historiques • Moyennes horaires calculées depuis les données réelles de la station
      </div>
    </div>
  )
}
