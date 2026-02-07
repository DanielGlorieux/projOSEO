"""
Service de notifications automatiques par e-mail
Pour alertes pompes et surveillance continue
"""
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """Service de notifications par e-mail pour ONEA"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
        self.enabled = bool(self.sender_email and self.sender_password)
        
        if not self.enabled:
            logger.warning("Service e-mail non configuré. Notifications désactivées.")
    
    def send_email(
        self,
        recipient: str,
        subject: str,
        body_html: str,
        body_text: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """
        Envoyer un e-mail
        
        Args:
            recipient: Adresse e-mail du destinataire
            subject: Sujet de l'e-mail
            body_html: Corps de l'e-mail en HTML
            body_text: Corps de l'e-mail en texte brut (optionnel)
            attachments: Liste de chemins de fichiers à attacher
        """
        if not self.enabled:
            logger.info(f"[SIMULATION] E-mail à {recipient}: {subject}")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
            
            # Ajouter le corps du message
            if body_text:
                msg.attach(MIMEText(body_text, 'plain', 'utf-8'))
            msg.attach(MIMEText(body_html, 'html', 'utf-8'))
            
            # Ajouter les pièces jointes
            if attachments:
                for file_path in attachments:
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={Path(file_path).name}'
                            )
                            msg.attach(part)
            
            # Envoyer l'e-mail
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"E-mail envoyé avec succès à {recipient}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur envoi e-mail: {str(e)}")
            return False
    
    def send_anomaly_alert(
        self,
        recipient: str,
        station_id: str,
        station_name: str,
        anomaly_details: Dict,
        severity: str = "medium"
    ) -> bool:
        """
        Envoyer une alerte d'anomalie détectée
        
        Args:
            recipient: Adresse e-mail du destinataire
            station_id: ID de la station
            station_name: Nom de la station
            anomaly_details: Détails de l'anomalie
            severity: Niveau de sévérité (low, medium, high, critical)
        """
        severity_colors = {
            "low": "#FFA500",
            "medium": "#FF8C00",
            "high": "#FF4500",
            "critical": "#DC143C"
        }
        
        severity_labels = {
            "low": "⚠️ Attention",
            "medium": "⚠️ Anomalie Modérée",
            "high": "🚨 Anomalie Importante",
            "critical": "🚨 ALERTE CRITIQUE"
        }
        
        color = severity_colors.get(severity, "#FFA500")
        label = severity_labels.get(severity, "Anomalie")
        
        subject = f"{label} - Station {station_name}"
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, {color} 0%, #FF6B6B 100%); 
                           color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; }}
                .alert-box {{ background: white; border-left: 4px solid {color}; 
                             padding: 15px; margin: 15px 0; border-radius: 4px; }}
                .metric {{ background: white; padding: 10px; margin: 10px 0; 
                          border-radius: 4px; display: flex; justify-content: space-between; }}
                .footer {{ background: #333; color: white; padding: 15px; 
                          text-align: center; border-radius: 0 0 8px 8px; font-size: 12px; }}
                .button {{ background: {color}; color: white; padding: 12px 24px; 
                          text-decoration: none; border-radius: 4px; display: inline-block; 
                          margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0;">💧 ONEA - Alerte Système</h1>
                    <p style="margin: 5px 0 0 0;">{label}</p>
                </div>
                
                <div class="content">
                    <div class="alert-box">
                        <h2 style="color: {color}; margin-top: 0;">Station: {station_name}</h2>
                        <p><strong>ID Station:</strong> {station_id}</p>
                        <p><strong>Heure de détection:</strong> {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}</p>
                        <p><strong>Niveau de sévérité:</strong> <span style="color: {color};">{severity.upper()}</span></p>
                    </div>
                    
                    <h3>📊 Détails de l'anomalie</h3>
                    
                    <div class="metric">
                        <span><strong>Type:</strong></span>
                        <span>{anomaly_details.get('type', 'Non spécifié')}</span>
                    </div>
                    
                    <div class="metric">
                        <span><strong>Valeur mesurée:</strong></span>
                        <span>{anomaly_details.get('value', 'N/A')}</span>
                    </div>
                    
                    <div class="metric">
                        <span><strong>Seuil normal:</strong></span>
                        <span>{anomaly_details.get('threshold', 'N/A')}</span>
                    </div>
                    
                    <div class="metric">
                        <span><strong>Écart:</strong></span>
                        <span style="color: {color};">{anomaly_details.get('deviation', 'N/A')}</span>
                    </div>
                    
                    <h3>🔧 Actions recommandées</h3>
                    <ul>
                        <li>Vérifier immédiatement l'état des pompes de la station</li>
                        <li>Consulter le tableau de bord pour plus de détails</li>
                        <li>Contacter l'équipe de maintenance si l'anomalie persiste</li>
                        <li>Documenter l'incident dans le système</li>
                    </ul>
                    
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="http://localhost:3000" class="button">Voir le Tableau de Bord</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>© 2026 ONEA - Office National de l'Eau et de l'Assainissement</p>
                    <p>Système d'optimisation énergétique par IA</p>
                    <p style="font-size: 10px; margin-top: 10px;">
                        Cet e-mail est envoyé automatiquement. Ne pas répondre.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        body_text = f"""
ONEA - Alerte Système
{label}

Station: {station_name} ({station_id})
Heure: {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}
Sévérité: {severity.upper()}

Détails de l'anomalie:
- Type: {anomaly_details.get('type', 'Non spécifié')}
- Valeur: {anomaly_details.get('value', 'N/A')}
- Seuil: {anomaly_details.get('threshold', 'N/A')}
- Écart: {anomaly_details.get('deviation', 'N/A')}

Actions recommandées:
1. Vérifier l'état des pompes
2. Consulter le tableau de bord
3. Contacter la maintenance si nécessaire
4. Documenter l'incident

---
ONEA - Système d'optimisation énergétique par IA
        """
        
        return self.send_email(recipient, subject, body_html, body_text)
    
    def send_maintenance_reminder(
        self,
        recipient: str,
        station_id: str,
        station_name: str,
        maintenance_type: str,
        scheduled_date: str
    ) -> bool:
        """Envoyer un rappel de maintenance programmée"""
        
        subject = f"🔧 Rappel Maintenance - {station_name}"
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%); 
                           color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; }}
                .info-box {{ background: white; padding: 15px; margin: 15px 0; 
                            border-radius: 4px; border-left: 4px solid #4A90E2; }}
                .footer {{ background: #333; color: white; padding: 15px; 
                          text-align: center; border-radius: 0 0 8px 8px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0;">💧 ONEA - Rappel Maintenance</h1>
                </div>
                
                <div class="content">
                    <div class="info-box">
                        <h2 style="color: #4A90E2; margin-top: 0;">Maintenance Programmée</h2>
                        <p><strong>Station:</strong> {station_name} ({station_id})</p>
                        <p><strong>Type de maintenance:</strong> {maintenance_type}</p>
                        <p><strong>Date prévue:</strong> {scheduled_date}</p>
                    </div>
                    
                    <h3>📋 Checklist de maintenance</h3>
                    <ul>
                        <li>Vérification des pompes et moteurs</li>
                        <li>Contrôle des niveaux de lubrification</li>
                        <li>Inspection des câbles et connexions électriques</li>
                        <li>Test des systèmes de sécurité</li>
                        <li>Nettoyage des filtres</li>
                        <li>Calibration des capteurs</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>© 2026 ONEA - Office National de l'Eau et de l'Assainissement</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(recipient, subject, body_html)
    
    def send_report_summary(
        self,
        recipient: str,
        report_period: str,
        summary_data: Dict,
        attachment_path: Optional[str] = None
    ) -> bool:
        """Envoyer un résumé de rapport avec données Excel en pièce jointe"""
        
        subject = f"📊 Rapport ONEA - {report_period}"
        
        body_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #27AE60 0%, #229954 100%); 
                           color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; }}
                .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 4px; 
                               text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .footer {{ background: #333; color: white; padding: 15px; 
                          text-align: center; border-radius: 0 0 8px 8px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0;">💧 ONEA - Rapport Période</h1>
                    <p style="margin: 5px 0 0 0;">{report_period}</p>
                </div>
                
                <div class="content">
                    <h3>📊 Métriques Clés</h3>
                    
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h4 style="margin: 0; color: #666;">Consommation</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 10px 0; color: #27AE60;">
                                {summary_data.get('total_energy_kwh', 0):,.0f} kWh
                            </p>
                        </div>
                        
                        <div class="metric-card">
                            <h4 style="margin: 0; color: #666;">Coûts</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 10px 0; color: #27AE60;">
                                {summary_data.get('total_cost_fcfa', 0):,.0f} FCFA
                            </p>
                        </div>
                        
                        <div class="metric-card">
                            <h4 style="margin: 0; color: #666;">Efficacité</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 10px 0; color: #27AE60;">
                                {summary_data.get('avg_efficiency', 0) * 100:.1f}%
                            </p>
                        </div>
                        
                        <div class="metric-card">
                            <h4 style="margin: 0; color: #666;">Économies</h4>
                            <p style="font-size: 24px; font-weight: bold; margin: 10px 0; color: #27AE60;">
                                {summary_data.get('savings_percent', 0):.1f}%
                            </p>
                        </div>
                    </div>
                    
                    <p style="margin-top: 20px; padding: 15px; background: white; border-radius: 4px;">
                        📎 <strong>Pièce jointe:</strong> Données détaillées au format Excel
                    </p>
                </div>
                
                <div class="footer">
                    <p>© 2026 ONEA - Office National de l'Eau et de l'Assainissement</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        attachments = [attachment_path] if attachment_path else None
        return self.send_email(recipient, subject, body_html, attachments=attachments)


# Instance globale du service
email_service = EmailNotificationService()
