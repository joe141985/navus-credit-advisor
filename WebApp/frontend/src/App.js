import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, CreditCard, MapPin, User, DollarSign } from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

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

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <CreditCard className="logo-icon" />
          <h1>NAVUS</h1>
          <p>Your Canadian Credit Card Advisor</p>
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