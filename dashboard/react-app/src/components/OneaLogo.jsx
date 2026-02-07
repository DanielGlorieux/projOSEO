/**
 * Logo officiel ONEA (Office National de l'Eau et de l'Assainissement)
 * Utilise le logo officiel depuis les assets
 */
import React from 'react';
import oneaLogoImg from '../assets/ONEA_Logo.webp';

const OneaLogo = ({ className = "w-12 h-12", showText = false }) => {
  return (
    <div className="flex items-center gap-3">
      {/* Logo officiel ONEA */}
      <img 
        src={oneaLogoImg} 
        alt="Logo ONEA - Office National de l'Eau et de l'Assainissement" 
        className={`${className} object-contain`}
        style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
      />
      
      {/* Texte optionnel à côté du logo */}
      {showText && (
        <div className="flex flex-col">
          <span className="text-xl font-bold text-blue-700">ONEA</span>
          <span className="text-xs text-gray-600">Office National de l'Eau</span>
        </div>
      )}
    </div>
  );
};

export default OneaLogo;
