import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  MessageCircle,
  Send,
  Loader2,
  X,
  Minimize2,
  Maximize2,
  Sparkles,
  BookOpen,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

const API_BASE = "http://localhost:8000";

const ChatbotAssistant = ({ stationId = null }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (isOpen && suggestions.length === 0) {
      loadSuggestions();
    }
  }, [isOpen]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadSuggestions = async () => {
    try {
      const response = await axios.get(`${API_BASE}/chatbot/suggestions`);
      setSuggestions(response.data.suggestions);
    } catch (err) {
      console.error("Erreur chargement suggestions:", err);
    }
  };

  const sendMessage = async (messageText = null) => {
    const textToSend = messageText || inputMessage;
    if (!textToSend.trim() || isLoading) return;

    const userMessage = {
      role: "user",
      content: textToSend,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);
    setError(null);

    try {
      const chatHistory = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await axios.post(`${API_BASE}/chatbot/query`, {
        query: textToSend,
        chat_history: chatHistory,
        station_id: stationId,
      });

      const botMessage = {
        role: "assistant",
        content: response.data.answer,
        sources: response.data.sources || [],
        timestamp: response.data.timestamp,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Erreur chatbot:", err);
      setError(
        "Désolé, une erreur s'est produite. Veuillez réessayer dans un instant.",
      );

      const errorMessage = {
        role: "assistant",
        content:
          "Je rencontre des difficultés techniques. Veuillez réessayer dans quelques instants.",
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (question) => {
    sendMessage(question);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-br from-blue-600 to-cyan-500 text-white rounded-full shadow-2xl hover:shadow-blue-500/50 hover:scale-110 transition-all duration-300 flex items-center justify-center z-50 group"
      >
        <MessageCircle className="w-7 h-7 group-hover:scale-110 transition-transform" />
        <span className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse"></span>
      </button>
    );
  }

  return (
    <div
      className={`fixed ${
        isMinimized ? "bottom-6 right-6 w-80" : "bottom-6 right-6 w-96"
      } transition-all duration-300 z-50`}
    >
      <Card
        className={`shadow-2xl border-2 border-blue-200 ${
          isMinimized ? "h-16" : "h-[600px]"
        } flex flex-col`}
      >
        {/* Header */}
        <CardHeader className="bg-gradient-to-r from-blue-600 to-cyan-500 text-white p-4 rounded-t-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <CardTitle className="text-lg font-bold">
                  Assistant IA ONEA
                </CardTitle>
                <p className="text-xs text-blue-100">
                  Aide aux agents de l'eau
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="hover:bg-white/20 p-2 rounded-lg transition-colors"
              >
                {isMinimized ? (
                  <Maximize2 className="w-4 h-4" />
                ) : (
                  <Minimize2 className="w-4 h-4" />
                )}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-white/20 p-2 rounded-lg transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </CardHeader>

        {!isMinimized && (
          <>
            {/* Messages Area */}
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
              {messages.length === 0 ? (
                <div className="text-center py-8">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-cyan-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <BookOpen className="w-8 h-8 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">
                    Bonjour! Comment puis-je vous aider?
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    Je peux répondre à vos questions sur:
                  </p>
                  <div className="space-y-2 text-left max-w-xs mx-auto">
                    <div className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-blue-600">⚡</span>
                      <span>Optimisation énergétique</span>
                    </div>
                    <div className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-blue-600">🔧</span>
                      <span>Maintenance des pompes</span>
                    </div>
                    <div className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-blue-600">📊</span>
                      <span>Analyse des performances</span>
                    </div>
                    <div className="flex items-start gap-2 text-sm text-gray-700">
                      <span className="text-blue-600">💰</span>
                      <span>Réduction des coûts</span>
                    </div>
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${
                      msg.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                        msg.role === "user"
                          ? "bg-blue-600 text-white"
                          : msg.isError
                            ? "bg-red-50 text-red-800 border border-red-200"
                            : "bg-white text-gray-800 shadow-sm border border-gray-200"
                      }`}
                    >
                      <div className="text-sm whitespace-pre-wrap">
                        {msg.content}
                      </div>
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <p className="text-xs font-semibold text-gray-600 mb-2">
                            📚 Sources:
                          </p>
                          {msg.sources.map((source, sIdx) => (
                            <div
                              key={sIdx}
                              className="text-xs text-gray-600 mb-1"
                            >
                              {source.score > 0.8 ? "🔥" : "📄"} {source.source}{" "}
                              (Score: {source.score.toFixed(2)})
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="text-xs text-gray-400 mt-2">
                        {new Date(msg.timestamp).toLocaleTimeString("fr-FR")}
                      </div>
                    </div>
                  </div>
                ))
              )}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-200">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                      <span className="text-sm text-gray-600">
                        Je réfléchis...
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="flex justify-center">
                  <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-2 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-600" />
                    <span className="text-sm text-red-800">{error}</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </CardContent>

            {/* Suggestions */}
            {messages.length === 0 && suggestions.length > 0 && (
              <div className="px-4 py-2 bg-white border-t border-gray-200">
                <p className="text-xs font-semibold text-gray-600 mb-2">
                  💡 Questions suggérées:
                </p>
                <div className="flex flex-wrap gap-2">
                  {suggestions.slice(0, 3).map((suggestion) => (
                    <button
                      key={suggestion.id}
                      onClick={() => handleSuggestionClick(suggestion.question)}
                      className="text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 px-3 py-1.5 rounded-full transition-colors border border-blue-200"
                    >
                      {suggestion.icon} {suggestion.category}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input Area */}
            <div className="p-4 bg-white border-t border-gray-200 rounded-b-lg">
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  className="text-xs text-gray-500 hover:text-gray-700 mb-2"
                >
                  🗑️ Effacer la conversation
                </button>
              )}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Posez votre question..."
                  className="flex-1 px-4 py-2 border-2 border-gray-300 rounded-full focus:border-blue-500 focus:outline-none text-sm"
                  disabled={isLoading}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={!inputMessage.trim() || isLoading}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white p-2 rounded-full transition-colors shadow-lg hover:shadow-blue-500/50"
                >
                  {isLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>
          </>
        )}
      </Card>
    </div>
  );
};

export default ChatbotAssistant;
