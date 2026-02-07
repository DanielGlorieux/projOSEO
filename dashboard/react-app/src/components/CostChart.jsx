import { useEffect, useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

const API_BASE = 'http://localhost:8000'

export default function CostChart({ stationId }) {
  const [costData, setCostData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadCostData()
  }, [stationId])

  const loadCostData = async () => {
    try {
      const response = await axios.get(`${API_BASE}/analytics/costs/${stationId}`)
      setCostData(response.data)
      setLoading(false)
    } catch (error) {
      console.error('Erreur chargement coûts:', error)
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="chart-card">Chargement des données de coûts...</div>
  }

  if (!costData) {
    return <div className="chart-card">Données de coûts non disponibles</div>
  }

  const currentData = [
    { period: 'Heures Creuses\n(23h-6h)', cost: costData.current_costs.off_peak, color: '#10b981' },
    { period: 'Heures Normales\n(6h-18h)', cost: costData.current_costs.normal, color: '#f59e0b' },
    { period: 'Heures Pleines\n(18h-23h)', cost: costData.current_costs.peak, color: '#ef4444' },
    { period: 'Pénalités', cost: costData.current_costs.penalties, color: '#8b5cf6' }
  ]

  const optimizedData = [
    { period: 'Heures Creuses\n(23h-6h)', cost: costData.optimized_costs.off_peak, color: '#10b981' },
    { period: 'Heures Normales\n(6h-18h)', cost: costData.optimized_costs.normal, color: '#f59e0b' },
    { period: 'Heures Pleines\n(18h-23h)', cost: costData.optimized_costs.peak, color: '#ef4444' },
    { period: 'Pénalités', cost: costData.optimized_costs.penalties, color: '#8b5cf6' }
  ]

  return (
    <div className="chart-card">
      <h2>💰 Analyse Coûts Énergétiques (FCFA) - {costData.period_days} derniers jours</h2>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '1rem', color: '#666', marginBottom: '10px' }}>Coûts Actuels (Données Réelles)</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={currentData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="period" style={{ fontSize: '0.8rem' }} />
            <YAxis />
            <Tooltip formatter={(value) => `${value.toLocaleString()} FCFA`} />
            <Bar dataKey="cost" name="Coût">
              {currentData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 style={{ fontSize: '1rem', color: '#666', marginBottom: '10px' }}>Après Optimisation IA ✨</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={optimizedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="period" style={{ fontSize: '0.8rem' }} />
            <YAxis />
            <Tooltip formatter={(value) => `${value.toLocaleString()} FCFA`} />
            <Bar dataKey="cost" name="Coût Optimisé">
              {optimizedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: '15px', padding: '15px', background: '#d4edda', borderRadius: '8px', borderLeft: '4px solid #28a745' }}>
        <p style={{ margin: 0, fontWeight: 'bold', color: '#155724' }}>
          💰 Économie Totale: {costData.savings.total_fcfa.toLocaleString()} FCFA ({costData.savings.percent.toFixed(1)}%)
        </p>
      </div>
    </div>
  )
}
