# 🚀 Guide Rapide: Hébergement Gratuit - ONEA Smart Energy Optimizer

**Date**: 7 Février 2026  
**Objectif**: Déployer l'application complète gratuitement

---

## 🎯 Solution Recommandée: Render + Vercel

### ✅ Pourquoi Cette Combinaison ?

| Service    | Usage           | Plan Gratuit | Avantages                          |
| ---------- | --------------- | ------------ | ---------------------------------- |
| **Render** | Backend FastAPI | 750h/mois    | Python natif, PostgreSQL inclus    |
| **Vercel** | Frontend React  | Illimité     | CDN global, déploiement instantané |
| **GitHub** | Code + CSV      | Illimité     | Versionning, collaboration         |

**Total: 100% GRATUIT** ✅

---

## 📦 ÉTAPE 1: Préparer le Projet

### A. Créer un Repository GitHub

```bash
cd C:\Users\danie\Desktop\projetLLMDocumentationHelperDaniel-master\hackathon_onea_2026

# Initialiser Git (si pas déjà fait)
git init

# Créer .gitignore
echo "node_modules/
__pycache__/
*.pyc
.env
.venv
venv/
dist/
build/
.DS_Store" > .gitignore

# Premier commit
git add .
git commit -m "Commit initial  - ONEA Smart Energy Optimizer"

# Créer repo sur GitHub puis:
git remote add origin https://github.com/VOTRE_USERNAME/onea-energy-optimizer.git
git branch -M main
git push -u origin main
```

### B. Préparer les Fichiers de Configuration

**1. Pour Render (Backend) - Créer `render.yaml`**

```yaml
services:
  # Backend API
  - type: web
    name: onea-backend-api
    env: python
    region: frankfurt
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.13
      - key: PORT
        value: 8000
    healthCheckPath: /health
```

**2. Créer `Procfile` (alternative)**

```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**3. Mettre à jour `requirements.txt`**

```bash
cd hackathon_onea_2026
pip freeze > requirements.txt
```

---

## 🔧 ÉTAPE 2: Déployer le Backend sur Render

### 1. Créer un Compte Render

- Aller sur https://render.com/
- Cliquer "Get Started for Free"
- S'inscrire avec GitHub (recommandé)

### 2. Créer le Service Backend

**A. Dashboard Render**

- Cliquer "New +" → "Web Service"
- Connecter votre repo GitHub
- Autoriser Render à accéder au repo

**B. Configuration**

```
Name: onea-backend-api
Region: Frankfurt (Europe)
Branch: main
Root Directory: (laisser vide OU mettre "hackathon_onea_2026")
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**C. Plan**

- Sélectionner "Free"
- 750 heures/mois (suffisant)

**D. Variables d'Environnement (si nécessaire)**

```
PYTHON_VERSION=3.10.13
PORT=8000
```

**E. Cliquer "Create Web Service"**

### 3. Attendre le Déploiement

- Durée: 5-10 minutes
- Suivre les logs en temps réel
- Une fois terminé, vous obtenez: `https://onea-backend-api-xxxx.onrender.com`

### 4. Tester l'API

```bash
# Test de santé
curl https://onea-backend-api-xxxx.onrender.com/health

# Test stations
curl https://onea-backend-api-xxxx.onrender.com/stations

# Test prévisions
curl https://onea-backend-api-xxxx.onrender.com/forecast/OUG_ZOG
```

---

## ⚡ ÉTAPE 3: Déployer le Frontend sur Vercel

### 1. Installer Vercel CLI

```bash
npm install -g vercel
```

### 2. Configurer le Frontend

**A. Mettre à jour l'URL de l'API**

Créer `dashboard/react-app/.env.production` :

```env
VITE_API_URL=https://onea-backend-api-xxxx.onrender.com
```

**B. Modifier les appels API** (si nécessaire)

Dans vos composants React :

```javascript
// Avant
const API_BASE = "http://localhost:8000";

// Après
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

**C. Tester en local**

```bash
cd dashboard/react-app
npm run build
npm run preview
```

### 3. Déployer sur Vercel

**A. Login**

```bash
vercel login
```

**B. Premier déploiement**

```bash
cd dashboard/react-app
vercel
```

Répondre aux questions :

```
? Set up and deploy "~/dashboard/react-app"? [Y/n] Y
? Which scope? Your Username
? Link to existing project? [y/N] N
? What's your project's name? onea-dashboard
? In which directory is your code located? ./
? Want to override the settings? [y/N] N
```

**C. Déploiement en production**

```bash
vercel --prod
```

### 4. Configurer les Variables d'Environnement

**Sur Vercel Dashboard** (https://vercel.com/dashboard)

- Aller dans votre projet "onea-dashboard"
- Settings → Environment Variables
- Ajouter :
  - Key: `VITE_API_URL`
  - Value: `https://onea-backend-api-xxxx.onrender.com`
  - Environments: Production, Preview, Development
- Save

**Redéployer** :

```bash
vercel --prod
```

### 5. URL Finale

Vous obtenez : `https://onea-dashboard-xxxx.vercel.app`

---

## 🗺️ ÉTAPE 4: Héberger les Données CSV

### Option 1: GitHub (Recommandé)

**Les CSV sont déjà dans votre repo !**

Accès via raw.githubusercontent.com :

```
https://raw.githubusercontent.com/VOTRE_USERNAME/onea-energy-optimizer/main/hackathon_onea_2026/data/raw/station_OUG_ZOG.csv
```

**Modifier l'API pour charger depuis GitHub** (si nécessaire) :

```python
import requests
import pandas as pd

def load_csv_from_github(station_id):
    base_url = "https://raw.githubusercontent.com/USERNAME/repo/main/hackathon_onea_2026/data/raw/"
    url = f"{base_url}station_{station_id}.csv"
    response = requests.get(url)
    return pd.read_csv(io.StringIO(response.text))
```

### Option 2: Render Disk (Alternative)

Render offre 1GB de stockage persistant gratuit :

```yaml
# Dans render.yaml, ajouter:
disk:
  name: data-disk
  mountPath: /data
  sizeGB: 1
```

Copier les CSV manuellement via le Shell de Render.

---

## ✅ ÉTAPE 5: Vérifications Post-Déploiement

### Checklist de Validation

```bash
# Backend Health Check
curl https://onea-backend-api-xxxx.onrender.com/health
# Attendu: {"status": "ok", "version": "1.0.0"}

# Stations List
curl https://onea-backend-api-xxxx.onrender.com/stations
# Attendu: [...liste des 5 stations...]

# Frontend Loading
curl -I https://onea-dashboard-xxxx.vercel.app
# Attendu: HTTP/2 200
```

### Tests dans le Navigateur

1. **Ouvrir** : https://onea-dashboard-xxxx.vercel.app
2. **Vérifier** :
   - [ ] Page se charge (< 3 secondes)
   - [ ] KPIs s'affichent
   - [ ] Carte Leaflet visible
   - [ ] Graphiques chargent
   - [ ] Pas d'erreurs console (F12)
   - [ ] Sélection de station fonctionne

---

## 🔄 ÉTAPE 6: Automatiser les Déploiements

### A. Déploiement Automatique Render

Render redéploie automatiquement à chaque push sur `main` :

```bash
git add .
git commit -m "Update backend"
git push origin main
# → Render redéploie automatiquement
```

### B. Déploiement Automatique Vercel

Vercel aussi redéploie automatiquement :

```bash
git add .
git commit -m "Update frontend"
git push origin main
# → Vercel redéploie automatiquement
```

**Webhooks** : Render + Vercel peuvent se déclencher mutuellement

---

## 📊 Récapitulatif des URLs

### Après Déploiement Complet

| Service         | URL                                                | Type         |
| --------------- | -------------------------------------------------- | ------------ |
| **Frontend**    | https://onea-dashboard-xxxx.vercel.app             | Public       |
| **Backend API** | https://onea-backend-api-xxxx.onrender.com         | Public       |
| **API Docs**    | https://onea-backend-api-xxxx.onrender.com/docs    | Public       |
| **GitHub Repo** | https://github.com/USERNAME/onea-energy-optimizer  | Public/Privé |
| **CSV Data**    | https://raw.githubusercontent.com/.../data/raw/... | Public       |

### À Partager dans le Hackathon

```markdown
## 🌐 Démos en Ligne

**Application Web** : https://onea-dashboard-xxxx.vercel.app  
**API Documentation** : https://onea-backend-api-xxxx.onrender.com/docs  
**Code Source** : https://github.com/USERNAME/onea-energy-optimizer

**Identifiants de test** : (si vous ajoutez une authentification)

- Email: demo@onea.bf
- Password: demo2026
```

---

## 🐛 Résolution de Problèmes

### Problème 1: Backend ne démarre pas

**Symptôme** : Error "Application failed to respond"

**Solutions** :

```bash
# Vérifier requirements.txt
pip freeze > requirements.txt

# Vérifier le port
# Dans main.py, ne PAS spécifier de port fixe:
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### Problème 2: Frontend charge mais API erreur

**Symptôme** : CORS error ou Failed to fetch

**Solution** :

Dans `api/main.py`, ajouter CORS :

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod, spécifier l'URL Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Problème 3: CSV non trouvés

**Symptôme** : FileNotFoundError

**Solution** :

```python
import os
from pathlib import Path

# Chemin relatif robuste
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

def load_station_data(station_id):
    file_path = DATA_DIR / f"station_{station_id}.csv"
    return pd.read_csv(file_path)
```

### Problème 4: Build échoue sur Render

**Symptôme** : "Failed to install requirements"

**Solution** :

```bash
# Simplifier requirements.txt
# Retirer les versions spécifiques si conflit
fastapi
uvicorn
pandas
numpy
scikit-learn
# etc.
```

---

## 💰 Limites des Plans Gratuits

### Render Free Tier

- ⏱️ **750 heures/mois** (suffisant pour 1 projet)
- 💤 **Sleep après 15 min d'inactivité** (réveil en ~30 sec)
- 💾 **512 MB RAM**
- 💽 **1 GB stockage disque**
- ⚠️ **Pas de custom domain sur free**

**Astuce** : Utiliser un service de ping (comme UptimeRobot) pour garder l'app éveillée

### Vercel Free Tier

- ✅ **Déploiements illimités**
- ✅ **100 GB bande passante/mois**
- ✅ **CDN global**
- ⚠️ **10 secondes max d'exécution serverless**
- ⚠️ **50 MB max par fichier**

---

## 🎁 Bonus: Monitoring Gratuit

### 1. UptimeRobot (Uptime Monitoring)

- https://uptimerobot.com/
- Ping toutes les 5 min
- Alertes email si down
- Garde l'app Render éveillée

**Configuration** :

```
Monitor Type: HTTP(s)
URL: https://onea-backend-api-xxxx.onrender.com/health
Interval: 5 minutes
```

### 2. Sentry (Error Tracking)

- https://sentry.io/
- 5,000 erreurs/mois gratuit
- Tracking frontend + backend

---

## ✅ Checklist Finale Hébergement

- [ ] Repository GitHub créé et poussé
- [ ] Backend déployé sur Render
- [ ] Backend accessible via curl
- [ ] Frontend déployé sur Vercel
- [ ] Frontend charge correctement
- [ ] API URL configurée dans frontend
- [ ] CORS configuré dans backend
- [ ] CSV accessibles (GitHub ou Render)
- [ ] Toutes les fonctionnalités testées
- [ ] URLs documentées
- [ ] Monitoring configuré (optionnel)

---

## 🚀 Commandes Utiles

```bash
# Backend - Render
render login
render services list
render logs -s onea-backend-api

# Frontend - Vercel
vercel login
vercel ls
vercel logs onea-dashboard

# Git
git status
git add .
git commit -m "Deploy to production"
git push origin main
```

---

## 📞 Support

**Render** : https://render.com/docs  
**Vercel** : https://vercel.com/docs  
**FastAPI** : https://fastapi.tiangolo.com/deployment/  
**Vite** : https://vitejs.dev/guide/static-deploy.html

---

**🎉 Félicitations ! Votre application ONEA est maintenant hébergée et accessible au monde entier !** 🌍🇧🇫
