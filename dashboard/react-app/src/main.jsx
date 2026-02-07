import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Désactiver StrictMode temporairement pour éviter les problèmes avec Leaflet
// En production, StrictMode ne s'active pas automatiquement
ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)
