import { useState } from "react";
import axios from "axios";
import {
  Download,
  FileSpreadsheet,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

const API_BASE = "http://localhost:8000";

const ExcelExportPanel = ({ stations, selectedStation }) => {
  const [loading, setLoading] = useState(false);
  const [exportType, setExportType] = useState("single");
  const [days, setDays] = useState(7);
  const [includeAnalytics, setIncludeAnalytics] = useState(true);
  const [message, setMessage] = useState(null);

  const handleExport = async () => {
    setLoading(true);
    setMessage(null);

    try {
      let response;

      if (exportType === "single") {
        response = await axios.post(
          `${API_BASE}/export/station-data`,
          {
            station_id: selectedStation,
            days: days,
            include_analytics: includeAnalytics,
          },
          {
            responseType: "blob",
          }
        );
      } else {
        const stationIds = stations.map((s) => s.station_id);
        response = await axios.post(
          `${API_BASE}/export/all-stations`,
          {
            station_ids: stationIds,
            days: days,
          },
          {
            responseType: "blob",
          }
        );
      }

      // Créer un lien de téléchargement
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;

      const timestamp = new Date().toISOString().split("T")[0];
      link.setAttribute(
        "download",
        exportType === "single"
          ? `ONEA_${selectedStation}_${timestamp}.xlsx`
          : `ONEA_Toutes_Stations_${timestamp}.xlsx`
      );

      document.body.appendChild(link);
      link.click();
      link.remove();

      setMessage({
        type: "success",
        text: "✅ Export Excel téléchargé avec succès!",
      });
    } catch (err) {
      console.error("Erreur export:", err);
      setMessage({
        type: "error",
        text: "❌ Erreur lors de l'export. Veuillez réessayer.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="shadow-lg">
      <CardHeader className="bg-gradient-to-r from-green-500 to-emerald-500 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileSpreadsheet className="w-6 h-6" />
            <CardTitle>Export de Données Excel</CardTitle>
          </div>
          <Badge variant="secondary" className="bg-white/20 text-white">
            Pour analyse externe
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-6 space-y-6">
        {/* Type d'export */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Type d'export
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setExportType("single")}
              className={`p-4 rounded-lg border-2 transition-all ${
                exportType === "single"
                  ? "border-green-500 bg-green-50 text-green-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <FileSpreadsheet className="w-6 h-6 mx-auto mb-2" />
              <p className="font-medium text-sm">Station Unique</p>
              <p className="text-xs text-gray-500 mt-1">
                Données détaillées + analyses
              </p>
            </button>

            <button
              onClick={() => setExportType("multi")}
              className={`p-4 rounded-lg border-2 transition-all ${
                exportType === "multi"
                  ? "border-green-500 bg-green-50 text-green-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <FileSpreadsheet className="w-6 h-6 mx-auto mb-2" />
              <p className="font-medium text-sm">Toutes les Stations</p>
              <p className="text-xs text-gray-500 mt-1">
                Synthèse comparative
              </p>
            </button>
          </div>
        </div>

        {/* Période */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Période de données (jours)
          </label>
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-green-500 focus:outline-none"
          >
            <option value={7}>7 jours (1 semaine)</option>
            <option value={14}>14 jours (2 semaines)</option>
            <option value={30}>30 jours (1 mois)</option>
            <option value={90}>90 jours (3 mois)</option>
            <option value={180}>180 jours (6 mois)</option>
            <option value={365}>365 jours (1 an)</option>
          </select>
        </div>

        {/* Options */}
        {exportType === "single" && (
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeAnalytics}
                onChange={(e) => setIncludeAnalytics(e.target.checked)}
                className="w-4 h-4 text-green-600 focus:ring-green-500 rounded"
              />
              <span className="text-sm text-gray-700">
                Inclure feuilles d'analyse et recommandations
              </span>
            </label>
          </div>
        )}

        {/* Contenu de l'export */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm font-semibold text-blue-800 mb-2">
            📊 Contenu de l'export:
          </p>
          <ul className="text-xs text-blue-700 space-y-1">
            <li>✓ Données brutes horodatées</li>
            {exportType === "single" && includeAnalytics && (
              <>
                <li>✓ Résumé statistique complet</li>
                <li>✓ Analyse horaire détaillée</li>
                <li>✓ Analyse par période tarifaire</li>
                <li>✓ Liste des anomalies détectées</li>
                <li>✓ Recommandations personnalisées</li>
              </>
            )}
            {exportType === "multi" && (
              <>
                <li>✓ Synthèse comparative entre stations</li>
                <li>✓ Données séparées par station</li>
              </>
            )}
            <li>✓ Format Excel (.xlsx) professionnel</li>
            <li>✓ Graphiques et mise en forme</li>
          </ul>
        </div>

        {/* Message */}
        {message && (
          <div
            className={`p-4 rounded-lg flex items-center gap-3 ${
              message.type === "success"
                ? "bg-green-50 text-green-800 border border-green-200"
                : "bg-red-50 text-red-800 border border-red-200"
            }`}
          >
            {message.type === "success" ? (
              <CheckCircle className="w-5 h-5" />
            ) : (
              <AlertCircle className="w-5 h-5" />
            )}
            <span className="text-sm font-medium">{message.text}</span>
          </div>
        )}

        {/* Bouton d'export */}
        <button
          onClick={handleExport}
          disabled={loading}
          className="w-full bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 disabled:from-gray-300 disabled:to-gray-400 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 shadow-lg hover:shadow-xl transition-all"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Export en cours...</span>
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              <span>Télécharger Excel</span>
            </>
          )}
        </button>

        {/* Info */}
        <div className="text-center">
          <p className="text-xs text-gray-500">
            Les fichiers Excel sont compatibles avec Microsoft Excel, Google
            Sheets et LibreOffice Calc
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExcelExportPanel;
