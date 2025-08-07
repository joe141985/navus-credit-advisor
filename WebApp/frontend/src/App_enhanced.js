import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, CreditCard, User, DollarSign, MapPin, Settings, Sparkles, Clock } from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://navus-api.onrender.com';

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: "Hi! I'm NAVUS, your Canadian credit card advisor. I can help you find the perfect credit card based on your needs, income, and spending habits. What would you like to know?",
      suggestions: []
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [featuredCards, setFeaturedCards] = useState([]);
  const [presetQuestions, setPresetQuestions] = useState({});
  const [selectedCategory, setSelectedCategory] = useState('general');
  const [showPersonaModal, setShowPersonaModal] = useState(false);
  const [userProfile, setUserProfile] = useState({
    persona: '',
    income: '',
    location: ''
  });
  const [apiHealth, setApiHealth] = useState({ status: 'unknown', model_type: 'unknown' });
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // Load health status
        const healthResponse = await axios.get(`${API_BASE_URL}/health`);
        setApiHealth(healthResponse.data);

        // Load featured cards
        const cardsResponse = await axios.get(`${API_BASE_URL}/cards`);
        setFeaturedCards(cardsResponse.data.featured_cards || []);

        // Load preset questions
        const questionsResponse = await axios.get(`${API_BASE_URL}/preset-questions`);
        setPresetQuestions(questionsResponse.data.preset_questions || {});

      } catch (error) {
        console.error('Error loading initial data:', error);
      }
    };
    
    loadInitialData();
  }, []);

  const sendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: userMessage,
        conversation_history: [],
        user_profile: userProfile
      });

      // Add assistant response with suggestions
      setMessages(prev => [
        ...prev, 
        { 
          type: 'assistant', 
          content: response.data.response,
          suggestions: response.data.suggested_questions || [],
          processing_time: response.data.processing_time
        }
      ]);

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev, 
        { 
          type: 'assistant', 
          content: "I'm sorry, I encountered an error. Please make sure the backend server is running and try again.",
          suggestions: []
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuestionClick = (question) => {
    setInputMessage(question);
  };

  const handlePersonaSave = () => {
    setShowPersonaModal(false);
    // Add a message indicating persona was set
    if (userProfile.persona) {
      setMessages(prev => [
        ...prev,
        {
          type: 'system',
          content: `✅ Profile updated: ${userProfile.persona}${userProfile.income ? `, Income: $${userProfile.income}` : ''}${userProfile.location ? `, Location: ${userProfile.location}` : ''}`
        }
      ]);
    }
  };

  const PersonaModal = () => (
    <div className="modal-overlay" onClick={() => setShowPersonaModal(false)}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h3>👤 Set Your Profile</h3>
        <p>Help NAVUS provide more personalized recommendations:</p>
        
        <div className="form-group">
          <label>I am a:</label>
          <select 
            value={userProfile.persona} 
            onChange={e => setUserProfile({...userProfile, persona: e.target.value})}
          >
            <option value="">Select...</option>
            <option value="student">Student</option>
            <option value="frequent_traveler">Frequent Traveler</option>
            <option value="cashback_focused">Cashback Focused</option>
            <option value="premium_seeker">Premium Benefits Seeker</option>
            <option value="credit_builder">Building Credit</option>
            <option value="business_owner">Business Owner</option>
          </select>
        </div>

        <div className="form-group">
          <label>Annual Income (optional):</label>
          <input 
            type="number" 
            placeholder="e.g. 50000"
            value={userProfile.income}
            onChange={e => setUserProfile({...userProfile, income: e.target.value})}
          />
        </div>

        <div className="form-group">
          <label>Province (optional):</label>
          <select 
            value={userProfile.location} 
            onChange={e => setUserProfile({...userProfile, location: e.target.value})}
          >
            <option value="">Select...</option>
            <option value="BC">British Columbia</option>
            <option value="AB">Alberta</option>
            <option value="ON">Ontario</option>
            <option value="QC">Quebec</option>
            <option value="NS">Nova Scotia</option>
            <option value="NB">New Brunswick</option>
            <option value="MB">Manitoba</option>
            <option value="SK">Saskatchewan</option>
            <option value="PE">Prince Edward Island</option>
            <option value="NL">Newfoundland and Labrador</option>
          </select>
        </div>

        <div className="modal-buttons">
          <button onClick={() => setShowPersonaModal(false)} className="btn-secondary">
            Cancel
          </button>
          <button onClick={handlePersonaSave} className="btn-primary">
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <CreditCard className="logo-icon" />
          <div className="header-text">
            <h1>NAVUS</h1>
            <p>Your Canadian Credit Card Advisor</p>
          </div>
          <div className="header-actions">
            <div className="model-status">
              <Sparkles size={16} />
              <span className={`status-indicator ${apiHealth.model_type}`}>
                {apiHealth.model_type === 'finetuned' ? 'AI Enhanced' : 'Standard'}
              </span>
            </div>
            <button 
              className="persona-button"
              onClick={() => setShowPersonaModal(true)}
            >
              <User size={16} />
              Profile
            </button>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar */}
        <div className="sidebar">
          {/* Persona Display */}
          {userProfile.persona && (
            <div className="user-profile-display">
              <h4>👤 Your Profile</h4>
              <div className="profile-info">
                <span className="profile-tag">{userProfile.persona.replace('_', ' ')}</span>
                {userProfile.income && <span className="profile-detail">${userProfile.income}/year</span>}
                {userProfile.location && <span className="profile-detail">{userProfile.location}</span>}
              </div>
            </div>
          )}

          {/* Category Tabs */}
          <div className="category-tabs">
            {Object.keys(presetQuestions).map(category => (
              <button
                key={category}
                className={`category-tab ${selectedCategory === category ? 'active' : ''}`}
                onClick={() => setSelectedCategory(category)}
              >
                {category.charAt(0).toUpperCase() + category.slice(1)}
              </button>
            ))}
          </div>

          {/* Preset Questions */}
          <div className="preset-questions">
            <h3>💡 Try asking:</h3>
            {(presetQuestions[selectedCategory] || []).map((question, index) => (
              <button
                key={index}
                className="preset-question"
                onClick={() => handleQuestionClick(question)}
                disabled={isLoading}
              >
                {question}
              </button>
            ))}
          </div>

          {/* Featured Cards */}
          {featuredCards.length > 0 && (
            <div className="featured-cards">
              <h3>🏆 Featured Cards:</h3>
              {featuredCards.slice(0, 4).map((card, index) => (
                <div key={index} className="featured-card">
                  <div className="card-name">{card.name}</div>
                  <div className="card-details">
                    <span className="card-issuer">{card.issuer}</span>
                    <span className="card-fee">${card.annual_fee}/year</span>
                  </div>
                  <div className="card-category">{card.category}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat Interface */}
        <div className="chat-container">
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? (
                    <User size={20} />
                  ) : message.type === 'system' ? (
                    <Settings size={20} />
                  ) : (
                    <CreditCard size={20} />
                  )}
                </div>
                <div className="message-content">
                  <div className="message-text">{message.content}</div>
                  
                  {/* Processing time display */}
                  {message.processing_time && (
                    <div className="message-meta">
                      <Clock size={12} />
                      <span>{message.processing_time.toFixed(2)}s</span>
                    </div>
                  )}
                  
                  {/* Suggested questions */}
                  {message.suggestions && message.suggestions.length > 0 && (
                    <div className="message-suggestions">
                      <div className="suggestions-label">💡 You might also ask:</div>
                      {message.suggestions.map((suggestion, idx) => (
                        <button
                          key={idx}
                          className="suggestion-chip"
                          onClick={() => handleQuestionClick(suggestion)}
                          disabled={isLoading}
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message assistant">
                <div className="message-avatar">
                  <CreditCard size={20} />
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={sendMessage} className="input-form">
            <div className="input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask me about Canadian credit cards..."
                disabled={isLoading}
                className="message-input"
              />
              <button
                type="submit"
                disabled={isLoading || !inputMessage.trim()}
                className="send-button"
              >
                <Send size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Persona Modal */}
      {showPersonaModal && <PersonaModal />}
    </div>
  );
}

export default App;