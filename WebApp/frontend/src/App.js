import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, CreditCard, MapPin, User, DollarSign, LogOut } from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://web-production-685ca.up.railway.app';

// Configure axios to always send credentials
axios.defaults.withCredentials = true;

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'assistant',
      content: "Hi! I'm NAVUS, your Canadian credit card advisor. I can help you find the perfect credit card based on your needs, income, and spending habits. What would you like to know?"
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sampleCards, setSampleCards] = useState([]);
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authLoading, setAuthLoading] = useState(true);
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Check authentication status and handle auth success
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        console.log('Checking auth status...');
        console.log('API Base URL:', API_BASE_URL);
        console.log('Request URL:', `${API_BASE_URL}/auth/status`);
        
        const response = await axios.get(`${API_BASE_URL}/auth/status`);
        
        console.log('Auth status response:', response.data);
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (response.data.authenticated) {
          console.log('User is authenticated:', response.data.user);
          setIsAuthenticated(true);
          setUser(response.data.user);
        } else {
          console.log('User is not authenticated, showing login button');
          setIsAuthenticated(false);
          setUser(null);
        }
      } catch (error) {
        console.error('Auth status check failed:', error);
        console.error('Error details:', error.response?.data);
        console.error('Error status:', error.response?.status);
        setIsAuthenticated(false);
        setUser(null);
      } finally {
        setAuthLoading(false);
      }
    };

    // Check for auth success in URL
    const urlParams = new URLSearchParams(window.location.search);
    const authSuccess = urlParams.get('auth');
    const token = urlParams.get('token');
    
    if (authSuccess === 'success' && token) {
      console.log('Auth success detected in URL, token:', token);
      // Clear the URL parameters
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    checkAuthStatus();
  }, []);

  // Load sample cards on component mount
  useEffect(() => {
    const loadSampleCards = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/cards`);
        setSampleCards(response.data.sample_cards || []);
      } catch (error) {
        console.error('Error loading sample cards:', error);
      }
    };
    
    loadSampleCards();
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
        conversation_history: []
      });

      // Add assistant response
      setMessages(prev => [
        ...prev, 
        { type: 'assistant', content: response.data.response }
      ]);

    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev, 
        { 
          type: 'assistant', 
          content: "I'm sorry, I encountered an error. Please make sure the backend server is running and try again." 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleQuestion = (question) => {
    setInputMessage(question);
  };

  // Authentication functions
  const handleGoogleLogin = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/google`);
      if (response.data.authorization_url) {
        window.location.href = response.data.authorization_url;
      }
    } catch (error) {
      console.error('Google login failed:', error);
      alert('Failed to initiate Google login. Please try again.');
    }
  };

  const handleTwitchLogin = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/twitch`);
      if (response.data.authorization_url) {
        window.location.href = response.data.authorization_url;
      }
    } catch (error) {
      console.error('Twitch login failed:', error);
      alert('Failed to initiate Twitch login. Please try again.');
    }
  };

  const handleSignOut = async () => {
    try {
      await axios.post(`${API_BASE_URL}/auth/logout`);
      setIsAuthenticated(false);
      setUser(null);
    } catch (error) {
      console.error('Sign out failed:', error);
    }
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <CreditCard className="logo-icon" />
          <h1>NAVUS</h1>
          <p>Your Canadian Credit Card Advisor</p>
        </div>
        
        {/* Authentication UI */}
        <div className="auth-section">
          {authLoading ? (
            <div className="auth-loading">Loading...</div>
          ) : isAuthenticated ? (
            <div className="user-menu">
              <div className="user-info">
                {user?.profile_picture && (
                  <img 
                    src={user.profile_picture} 
                    alt="Profile" 
                    className="profile-picture"
                  />
                )}
                <span className="user-name">{user?.name || user?.email}</span>
              </div>
            </div>
          ) : (
            <div className="login-buttons">
              <button 
                onClick={handleGoogleLogin}
                className="login-btn google-btn"
              >
                <svg width="18" height="18" viewBox="0 0 48 48">
                  <path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
                  <path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
                  <path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>
                  <path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
                </svg>
                Google
              </button>
              <button 
                onClick={handleTwitchLogin}
                className="login-btn twitch-btn"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#9146FF">
                  <path d="M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714Z"/>
                </svg>
                Twitch
              </button>
            </div>
          )}
        </div>
      </header>

      <div className="main-container">
        {/* Sample Questions Sidebar */}
        <div className="sidebar">
          <h3>💡 Try asking:</h3>
          <div className="sample-questions">
            {[
              "What's the best no-fee travel card?",
              "I'm a student looking for my first credit card",
              "Best cashback card for groceries?", 
              "Compare premium travel cards",
              "Credit cards for building credit history"
            ].map((question, index) => (
              <button
                key={index}
                className="sample-question"
                onClick={() => handleSampleQuestion(question)}
                disabled={isLoading}
              >
                {question}
              </button>
            ))}
          </div>

          {/* Sample Cards */}
          {sampleCards.length > 0 && (
            <div className="sample-cards">
              <h3>🏆 Featured Cards:</h3>
              {sampleCards.map((card, index) => (
                <div key={index} className="sample-card">
                  <div className="card-name">{card.name}</div>
                  <div className="card-details">
                    <span className="card-issuer">{card.issuer}</span>
                    <span className="card-fee">${card.annual_fee}/year</span>
                  </div>
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
                  {message.type === 'user' ? <User size={20} /> : <CreditCard size={20} />}
                </div>
                <div className="message-content">
                  {message.content}
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
    </div>
  );
}

export default App;