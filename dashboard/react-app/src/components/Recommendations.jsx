export default function Recommendations({ actions, savings, savingsPercent }) {
  const recommendations = [
    {
      icon: '⏰',
      title: 'Programmer 45% du pompage en heures creuses',
      description: '(23h-6h) pour profiter des tarifs réduits',
      savings: '620,000 FCFA/mois'
    },
    {
      icon: '⚙️',
      title: 'Réduire de 1 pompe active aux heures normales',
      description: 'Optimisation du nombre de pompes selon la demande',
      savings: 'Gain efficacité: +6.2%'
    },
    {
      icon: '⚡',
      title: 'Corriger le facteur de puissance',
      description: 'Installation de condensateurs pour éliminer les pénalités',
      savings: '180,000 FCFA/mois'
    },
    {
      icon: '🔧',
      title: 'Maintenance préventive Pompe #3',
      description: 'Dégradation détectée par IA - intervention requise',
      savings: 'Éviter panne coûteuse'
    }
  ]

  return (
    <div className="recommendations">
      <h2>🎯 Recommandations IA Immédiates</h2>
      
      {actions && actions.length > 0 && (
        <div style={{ marginBottom: '20px', padding: '15px', background: '#fff3cd', borderRadius: '8px', borderLeft: '4px solid #ffc107' }}>
          <h4 style={{ margin: '0 0 10px 0', color: '#856404' }}>⚡ Actions en Cours</h4>
          {actions.map((action, index) => (
            <div key={index} style={{ marginBottom: '10px', color: '#856404' }}>
              <strong>{action.action}:</strong> {action.reason}
              <br />
              <small>Actuel: {action.current} → Recommandé: {action.recommended}</small>
            </div>
          ))}
        </div>
      )}

      {recommendations.map((rec, index) => (
        <div key={index} className="recommendation-item">
          <h4>{rec.icon} {rec.title}</h4>
          <p>{rec.description}</p>
          <div style={{ 
            marginTop: '10px', 
            fontWeight: 'bold', 
            color: '#28a745',
            fontSize: '0.95rem'
          }}>
            💰 Économie: {rec.savings}
          </div>
        </div>
      ))}

      <div style={{ 
        marginTop: '25px', 
        padding: '20px', 
        background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        borderRadius: '10px',
        color: 'white'
      }}>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '10px' }}>📊 Planning Optimisé Prochaines 24h</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '8px', fontSize: '0.75rem' }}>
          {Array.from({ length: 24 }, (_, h) => {
            let status = '🟡'
            let pumps = 3
            if (h >= 23 || h < 6) { status = '🟢'; pumps = 4 }
            else if (h >= 18 && h < 23) { status = '🔴'; pumps = 2 }
            
            return (
              <div key={h} style={{ 
                padding: '8px', 
                background: 'rgba(255,255,255,0.2)', 
                borderRadius: '5px',
                textAlign: 'center'
              }}>
                <div style={{ fontWeight: 'bold' }}>{h}h</div>
                <div>{status}</div>
                <div style={{ fontSize: '0.7rem' }}>{pumps}/4</div>
              </div>
            )
          })}
        </div>
        <div style={{ marginTop: '15px', fontSize: '0.85rem', textAlign: 'center' }}>
          🟢 Utilisation max | 🟡 Utilisation modérée | 🔴 Utilisation min
        </div>
      </div>
    </div>
  )
}
