import { useState } from "react";
import axios from "axios";
import { Mail, Bell, Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

const API_BASE = "http://localhost:8000";

const EmailNotificationPanel = ({ stations, selectedStation }) => {
  const [loading, setLoading] = useState(false);
  const [notificationType, setNotificationType] = useState("anomaly");
  const [recipient, setRecipient] = useState("");
  const [message, setMessage] = useState(null);

  const [anomalyData, setAnomalyData] = useState({
    type: "surconsommation_specifique",
    value: "2.5 kWh/m³",
    threshold: "1.8 kWh/m³",
    deviation: "+38%",
    severity: "high",
  });

  const [maintenanceData, setMaintenanceData] = useState({
    type: "Maintenance préventive trimestrielle",
    scheduledDate: new Date().toISOString().split("T")[0],
  });

  const handleSendNotification = async () => {
    if (!recipient.trim()) {
      setMessage({
        type: "error",
        text: "❌ Veuillez saisir une adresse e-mail",
      });
      return;
    }

    // Validation e-mail basique
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(recipient)) {
      setMessage({
        type: "error",
        text: "❌ Adresse e-mail invalide",
      });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const station = stations.find((s) => s.station_id === selectedStation);

      if (notificationType === "anomaly") {
        await axios.post(`${API_BASE}/notifications/anomaly-alert`, {
          recipient: recipient,
          station_id: selectedStation,
          station_name: station?.name || selectedStation,
          anomaly_details: anomalyData,
          severity: anomalyData.severity,
        });
      } else {
        await axios.post(
          `${API_BASE}/notifications/maintenance-reminder`,
          null,
          {
            params: {
              recipient: recipient,
              station_id: selectedStation,
              station_name: station?.name || selectedStation,
              maintenance_type: maintenanceData.type,
              scheduled_date: maintenanceData.scheduledDate,
            },
          },
        );
      }

      setMessage({
        type: "success",
        text: "✅ E-mail envoyé avec succès!",
      });
      setRecipient("");
    } catch (err) {
      console.error("Erreur envoi e-mail:", err);
      setMessage({
        type: "error",
        text: "❌ Erreur lors de l'envoi. Veuillez réessayer.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="shadow-lg">
      <CardHeader className="bg-gradient-to-r from-orange-500 to-red-500 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Mail className="w-6 h-6" />
            <CardTitle>Notifications par E-mail</CardTitle>
          </div>
          <Badge variant="secondary" className="bg-white/20 text-white">
            <Bell className="w-3 h-3 mr-1" />
            Surveillance continue
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-6 space-y-6">
        {/* Type de notification */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Type de notification
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setNotificationType("anomaly")}
              className={`p-4 rounded-lg border-2 transition-all ${
                notificationType === "anomaly"
                  ? "border-orange-500 bg-orange-50 text-orange-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <AlertCircle className="w-6 h-6 mx-auto mb-2" />
              <p className="font-medium text-sm">Alerte Anomalie</p>
              <p className="text-xs text-gray-500 mt-1">
                Problème détecté sur station
              </p>
            </button>

            <button
              onClick={() => setNotificationType("maintenance")}
              className={`p-4 rounded-lg border-2 transition-all ${
                notificationType === "maintenance"
                  ? "border-orange-500 bg-orange-50 text-orange-700"
                  : "border-gray-200 hover:border-gray-300"
              }`}
            >
              <Bell className="w-6 h-6 mx-auto mb-2" />
              <p className="font-medium text-sm">Rappel Maintenance</p>
              <p className="text-xs text-gray-500 mt-1">
                Maintenance programmée
              </p>
            </button>
          </div>
        </div>

        {/* Destinataire */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Adresse e-mail du destinataire
          </label>
          <input
            type="email"
            value={recipient}
            onChange={(e) => setRecipient(e.target.value)}
            placeholder="agent@onea.bf"
            className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-orange-500 focus:outline-none"
          />
        </div>

        {/* Configuration spécifique */}
        {notificationType === "anomaly" ? (
          <div className="space-y-4">
            <p className="text-sm font-semibold text-gray-700">
              Détails de l'anomalie
            </p>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-600 mb-1">
                  Sévérité
                </label>
                <select
                  value={anomalyData.severity}
                  onChange={(e) =>
                    setAnomalyData({ ...anomalyData, severity: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                >
                  <option value="low">⚠️ Basse</option>
                  <option value="medium">⚠️ Moyenne</option>
                  <option value="high">🚨 Haute</option>
                  <option value="critical">🚨 Critique</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-gray-600 mb-1">Type</label>
                <select
                  value={anomalyData.type}
                  onChange={(e) =>
                    setAnomalyData({ ...anomalyData, type: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                >
                  <option value="surconsommation_specifique">
                    Surconsommation
                  </option>
                  <option value="facteur_puissance_bas">
                    Facteur de puissance
                  </option>
                  <option value="efficacite_degradee">
                    Efficacité dégradée
                  </option>
                </select>
              </div>
            </div>

            <div className="bg-orange-50 border border-orange-200 rounded-lg p-3">
              <p className="text-xs font-semibold text-orange-800 mb-2">
                📊 Valeurs détectées:
              </p>
              <div className="text-xs text-orange-700 space-y-1">
                <div className="flex justify-between">
                  <span>Valeur mesurée:</span>
                  <span className="font-medium">{anomalyData.value}</span>
                </div>
                <div className="flex justify-between">
                  <span>Seuil normal:</span>
                  <span className="font-medium">{anomalyData.threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span>Écart:</span>
                  <span className="font-medium text-red-600">
                    {anomalyData.deviation}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-sm font-semibold text-gray-700">
              Détails de la maintenance
            </p>

            <div>
              <label className="block text-xs text-gray-600 mb-1">
                Type de maintenance
              </label>
              <input
                type="text"
                value={maintenanceData.type}
                onChange={(e) =>
                  setMaintenanceData({
                    ...maintenanceData,
                    type: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-600 mb-1">
                Date prévue
              </label>
              <input
                type="date"
                value={maintenanceData.scheduledDate}
                onChange={(e) =>
                  setMaintenanceData({
                    ...maintenanceData,
                    scheduledDate: e.target.value,
                  })
                }
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
              />
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-xs font-semibold text-blue-800 mb-2">
                ✅ Checklist maintenance:
              </p>
              <ul className="text-xs text-blue-700 space-y-1">
                <li>• Vérification pompes et moteurs</li>
                <li>• Contrôle lubrification</li>
                <li>• Inspection électrique</li>
                <li>• Test systèmes de sécurité</li>
                <li>• Calibration capteurs</li>
              </ul>
            </div>
          </div>
        )}

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

        {/* Bouton d'envoi */}
        <button
          onClick={handleSendNotification}
          disabled={loading || !recipient.trim()}
          className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 disabled:from-gray-300 disabled:to-gray-400 text-white py-3 px-6 rounded-lg font-semibold flex items-center justify-center gap-2 shadow-lg hover:shadow-xl transition-all"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Envoi en cours...</span>
            </>
          ) : (
            <>
              <Mail className="w-5 h-5" />
              <span>Envoyer la notification</span>
            </>
          )}
        </button>

        {/* Info */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <p className="text-xs text-gray-600">
            💡 <strong>INFO:</strong> Les notifications par e-mail sont en cours
            de développement, cette fonctionalité sera disponible dans les
            prochaines versions
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default EmailNotificationPanel;
