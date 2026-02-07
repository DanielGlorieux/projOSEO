"""
Script de test pour vérifier toutes les fonctionnalités ONEA
"""
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    print_section("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"✅ Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_stations():
    print_section("TEST 2: Liste des Stations")
    try:
        response = requests.get(f"{API_BASE}/stations")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Nombre de stations: {len(data)}")
        for station in data:
            print(f"  - {station['name']} ({station['station_id']})")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_forecast():
    print_section("TEST 3: Prédiction Consommation")
    try:
        payload = {
            "station_id": "OUG_ZOG",
            "horizon_hours": 24
        }
        response = requests.post(f"{API_BASE}/forecast", json=payload)
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Station: {data['station_id']}")
        print(f"Nombre de prédictions: {len(data['predictions'])}")
        print(f"Modèle: {data['metadata'].get('model', 'N/A')}")
        print(f"MAPE: {data['metadata'].get('mape', 'N/A')}")
        print(f"Exemple prédictions (3 premières heures):")
        for i in range(min(3, len(data['predictions']))):
            print(f"  Heure {i}: {data['predictions'][i]:.2f} kWh")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_optimization():
    print_section("TEST 4: Optimisation RL")
    try:
        payload = {
            "station_id": "OUG_ZOG",
            "current_state": {}
        }
        response = requests.post(f"{API_BASE}/optimize", json=payload)
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Station: {data['station_id']}")
        print(f"Économies attendues: {data['expected_savings_fcfa']:,.0f} FCFA")
        print(f"Pourcentage d'économie: {data['expected_savings_percent']:.1f}%")
        print(f"Nombre de recommandations: {len(data['recommended_actions'])}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_analytics():
    print_section("TEST 5: Analytics Summary")
    try:
        response = requests.get(f"{API_BASE}/analytics/summary/OUG_ZOG?days=7")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Station: {data['station_id']}")
        print(f"Points de données: {data['data_points']}")
        print(f"Métriques:")
        for key, value in data['metrics'].items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_hourly_efficiency():
    print_section("TEST 6: Efficacité Horaire")
    try:
        response = requests.get(f"{API_BASE}/analytics/hourly-efficiency/OUG_ZOG")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Station: {data['station_id']}")
        print(f"Stats globales:")
        print(f"  - Efficacité moyenne: {data['stats']['efficiency']:.2f}%")
        print(f"  - Facteur puissance: {data['stats']['power_factor']:.3f}")
        print(f"  - Niveau réservoir: {data['stats']['reservoir']:.1f}%")
        print(f"Exemple données horaires (3 premières heures):")
        for i in range(min(3, len(data['hourly_data']))):
            hour_data = data['hourly_data'][i]
            print(f"  {hour_data['hour']}: Eff={hour_data['efficiency']:.1f}%, Res={hour_data['reservoir']:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_chatbot_suggestions():
    print_section("TEST 7: Chatbot - Suggestions")
    try:
        response = requests.get(f"{API_BASE}/chatbot/suggestions")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Nombre de suggestions: {data['count']}")
        for suggestion in data['suggestions']:
            print(f"  {suggestion['icon']} [{suggestion['category']}] {suggestion['question']}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_chatbot_query():
    print_section("TEST 8: Chatbot - Query")
    try:
        payload = {
            "query": "Comment réduire la consommation énergétique ?",
            "chat_history": [],
            "station_id": "OUG_ZOG"
        }
        response = requests.post(f"{API_BASE}/chatbot/query", json=payload)
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Succès: {data['success']}")
        print(f"Réponse: {data['answer'][:200]}..." if len(data['answer']) > 200 else data['answer'])
        print(f"Nombre de sources: {len(data.get('sources', []))}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_export_list():
    print_section("TEST 9: Liste des Exports")
    try:
        response = requests.get(f"{API_BASE}/export/list-exports")
        print(f"✅ Status: {response.status_code}")
        data = response.json()
        print(f"Nombre d'exports disponibles: {data['count']}")
        for export in data['exports'][:3]:  # Afficher max 3
            print(f"  - {export['filename']} ({export['size_mb']:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_export_station():
    print_section("TEST 10: Export Station Data")
    try:
        payload = {
            "station_id": "OUG_ZOG",
            "days": 7,
            "include_analytics": True
        }
        response = requests.post(f"{API_BASE}/export/station-data", json=payload)
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            filename = f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Fichier sauvegardé: {filename}")
            print(f"Taille: {len(response.content) / 1024:.2f} KB")
            return True
        else:
            print(f"Erreur: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TEST SUITE ONEA SMART ENERGY OPTIMIZER API            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Base: {API_BASE}")
    
    tests = [
        ("Health Check", test_health),
        ("Stations", test_stations),
        ("Forecast", test_forecast),
        ("Optimization", test_optimization),
        ("Analytics", test_analytics),
        ("Hourly Efficiency", test_hourly_efficiency),
        ("Chatbot Suggestions", test_chatbot_suggestions),
        ("Chatbot Query", test_chatbot_query),
        ("Export List", test_export_list),
        ("Export Station", test_export_station),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print_section("RÉSUMÉ DES TESTS")
    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed
    
    print(f"\nTotal: {total}")
    print(f"✅ Réussis: {passed}")
    print(f"❌ Échoués: {failed}")
    print(f"\nTaux de réussite: {(passed/total)*100:.1f}%")
    
    print("\nDétails:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if failed == 0:
        print("\n🎉 Tous les tests sont réussis! L'API fonctionne parfaitement.")
    else:
        print(f"\n⚠️ {failed} test(s) ont échoué. Vérifiez les logs ci-dessus.")
        print("\nVérifications suggérées:")
        print("  1. L'API est-elle démarrée sur http://localhost:8000 ?")
        print("  2. Les variables d'environnement sont-elles configurées (.env) ?")
        print("  3. Les données CSV existent-elles dans data/raw/ ?")
        print("  4. Les services RAG sont-ils disponibles (Pinecone, Gemini) ?")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
