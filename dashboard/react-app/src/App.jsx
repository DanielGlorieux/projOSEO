import { useState, useEffect } from "react";
import axios from "axios";
import {
  Activity,
  Zap,
  TrendingDown,
  DollarSign,
  Droplets,
  Power,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  TrendingUp,
  Settings,
  LayoutDashboard,
  Map,
  MessageCircle,
  Download,
  Mail,
} from "lucide-react";
import OneaLogo from "./components/OneaLogo";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Progress } from "./components/ui/progress";
import EnergyChart from "./components/EnergyChart";
import CostChart from "./components/CostChart";
import EfficiencyChart from "./components/EfficiencyChart";
import OptimizationImpact from "./components/OptimizationImpact";
import Recommendations from "./components/Recommendations";
import ChatbotAssistant from "./components/ChatbotAssistant";
import ExcelExportPanel from "./components/ExcelExportPanel";
import EmailNotificationPanel from "./components/EmailNotificationPanel";
import StationMapLeaflet from "./components/StationMapLeaflet";

const API_BASE = "http://localhost:8000";

function App() {
  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState("OUG_ZOG");
  const [metrics, setMetrics] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");

  useEffect(() => {
    loadStations();
  }, []);

  useEffect(() => {
    if (selectedStation) {
      loadData();
    }
  }, [selectedStation]);

  const loadStations = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stations`);
      setStations(response.data);
      setLoading(false);
    } catch (err) {
      setError("Erreur de connexion à l'API");
      setLoading(false);
    }
  };

  const loadData = async () => {
    setLoading(true);
    try {
      const [forecastRes, optimizationRes, analyticsRes] = await Promise.all([
        axios.post(`${API_BASE}/forecast`, {
          station_id: selectedStation,
          horizon_hours: 24,
        }),
        axios.post(`${API_BASE}/optimize`, {
          station_id: selectedStation,
          current_state: {},
        }),
        axios.get(`${API_BASE}/analytics/summary/${selectedStation}`),
      ]);

      setForecast(forecastRes.data);
      setOptimization(optimizationRes.data);
      setMetrics(analyticsRes.data.metrics);
      setLoading(false);
    } catch (err) {
      console.error("Erreur chargement données:", err);
      setError("Erreur de chargement des données");
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-xl text-gray-700 font-medium">
            ⚡ Chargement des données ONEA...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 flex items-center justify-center">
        <div className="bg-red-50 border-2 border-red-200 rounded-xl p-6 max-w-md">
          <AlertTriangle className="w-12 h-12 text-red-600 mx-auto mb-3" />
          <p className="text-lg text-red-700 text-center font-medium">
            {error}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Logo ONEA officiel */}
              <OneaLogo className="h-14 w-auto" showText={false} />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  ONEA - Optimisation Énergétique
                </h1>
                <p className="text-sm text-gray-600">
                  Système intelligent de gestion énergétique
                </p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-600">Burkina Faso</p>
                <p className="text-xs text-gray-500">
                  Ouagadougou • Bobo-Dioulasso
                </p>
              </div>
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-blue-700 font-semibold">U</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Station Selector */}
        <div className="bg-white rounded-lg shadow-sm p-4 mb-6 border border-gray-200">
          <label className="text-sm font-medium text-gray-700 mb-2 block">
            📍 Station sélectionnée
          </label>
          <select
            value={selectedStation}
            onChange={(e) => setSelectedStation(e.target.value)}
            className="w-full px-4 py-3 border-2 border-blue-200 rounded-lg focus:border-blue-500 focus:outline-none transition-colors"
          >
            {stations.map((station) => (
              <option key={station.station_id} value={station.station_id}>
                {station.name} - {station.location}
              </option>
            ))}
          </select>
        </div>

        {/* Tabs */}
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="space-y-6"
        >
          <TabsList className="grid w-full grid-cols-7 bg-white p-1 rounded-lg shadow-sm">
            <TabsTrigger value="dashboard" className="flex items-center gap-2">
              <LayoutDashboard className="w-4 h-4" />
              <span className="hidden md:inline">Tableau de Bord</span>
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              <span className="hidden md:inline">Analyse Énergie</span>
            </TabsTrigger>
            <TabsTrigger value="costs" className="flex items-center gap-2">
              <DollarSign className="w-4 h-4" />
              <span className="hidden md:inline">Coûts</span>
            </TabsTrigger>
            <TabsTrigger
              value="performance"
              className="flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
              <span className="hidden md:inline">Performance</span>
            </TabsTrigger>
            <TabsTrigger
              value="optimization"
              className="flex items-center gap-2"
            >
              <TrendingUp className="w-4 h-4" />
              <span className="hidden md:inline">Optimisation</span>
            </TabsTrigger>
            <TabsTrigger value="export" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              <span className="hidden md:inline">Export Excel</span>
            </TabsTrigger>
            <TabsTrigger
              value="notifications"
              className="flex items-center gap-2"
            >
              <Mail className="w-4 h-4" />
              <span className="hidden md:inline">Notifications</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            {metrics && (
              <>
                {/* KPIs */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white border-0 shadow-lg">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-medium text-blue-100">
                          Consommation Totale
                        </CardTitle>
                        <Zap className="w-5 h-5 text-blue-100" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">
                        {metrics.total_energy_kwh.toLocaleString()} kWh
                      </div>
                      <p className="text-xs text-blue-100 flex items-center gap-1">
                        <TrendingDown className="w-3 h-3" />
                        -28.5% vs baseline
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white border-0 shadow-lg">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-medium text-green-100">
                          Coûts Totaux
                        </CardTitle>
                        <DollarSign className="w-5 h-5 text-green-100" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">
                        {(metrics.total_cost_fcfa / 1000000).toFixed(2)}M FCFA
                      </div>
                      <p className="text-xs text-green-100">Économie: 28.5%</p>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-cyan-500 to-cyan-600 text-white border-0 shadow-lg">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-medium text-cyan-100">
                          Efficacité Moyenne
                        </CardTitle>
                        <Activity className="w-5 h-5 text-cyan-100" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">
                        {(metrics.avg_efficiency * 100).toFixed(1)}%
                      </div>
                      <p className="text-xs text-cyan-100">
                        +4.2% amélioration
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white border-0 shadow-lg">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-medium text-purple-100">
                          Anomalies Détectées
                        </CardTitle>
                        <AlertTriangle className="w-5 h-5 text-purple-100" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold mb-1">
                        {metrics.anomalies_detected}
                      </div>
                      <p className="text-xs text-purple-100">Cette période</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Additional KPIs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between mb-3">
                        <CardTitle>💚 Économies Estimées</CardTitle>
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-gray-900 mb-1">
                        {metrics.savings_vs_baseline_percent.toFixed(1)}%
                      </div>
                      <p className="text-sm text-gray-600">Ce mois</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between mb-3">
                        <CardTitle>🌱 Réduction CO₂</CardTitle>
                        <Droplets className="w-5 h-5 text-green-600" />
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-gray-900 mb-1">
                        {metrics.co2_reduction_tons.toFixed(1)} t
                      </div>
                      <p className="text-sm text-gray-600">Par an</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Carte des Stations ONEA */}
                <StationMapLeaflet
                  selectedStation={selectedStation}
                  onStationSelect={setSelectedStation}
                />
              </>
            )}
          </TabsContent>

          <TabsContent value="analysis" className="space-y-6">
            <EnergyChart stationId={selectedStation} />
          </TabsContent>

          <TabsContent value="costs" className="space-y-6">
            <CostChart stationId={selectedStation} />
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <EfficiencyChart stationId={selectedStation} />
          </TabsContent>

          <TabsContent value="optimization" className="space-y-6">
            {optimization && (
              <>
                <OptimizationImpact optimization={optimization} />
                <Recommendations
                  actions={optimization.recommended_actions}
                  savings={optimization.expected_savings_fcfa}
                  savingsPercent={optimization.expected_savings_percent}
                />
              </>
            )}
          </TabsContent>

          <TabsContent value="export" className="space-y-6">
            <ExcelExportPanel
              stations={stations}
              selectedStation={selectedStation}
            />
          </TabsContent>

          <TabsContent value="notifications" className="space-y-6">
            <EmailNotificationPanel
              stations={stations}
              selectedStation={selectedStation}
            />
          </TabsContent>
        </Tabs>
      </main>

      {/* Chatbot Assistant (toujours visible) */}
      <ChatbotAssistant stationId={selectedStation} />

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="container mx-auto px-4 py-4">
          <p className="text-center text-sm text-gray-600">
            © 2026 ONEA x Maison de l'Intelligence Artificielle - Hackathon
            Optimisation Énergétique
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
