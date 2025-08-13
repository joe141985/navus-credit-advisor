import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import uuid
import json

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./navus_auth.db")
# Handle Railway/Heroku database URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password_hash = Column(String, nullable=True)  # Nullable for OAuth users
    google_id = Column(String, unique=True, index=True, nullable=True)  # Google OAuth ID
    twitch_id = Column(String, unique=True, index=True, nullable=True)  # Twitch OAuth ID
    twitch_login = Column(String, nullable=True)  # Twitch username
    profile_picture = Column(String, nullable=True)  # Profile picture URL
    oauth_provider = Column(String, nullable=True)  # 'google', 'twitch', etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    expires_at = Column(DateTime)
    is_persistent = Column(Boolean, default=False)  # "Keep me logged in" flag
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    """
    Represents a conversation/chat session for a user.
    Each conversation has its own context and summary.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=True, index=True)  # Nullable for anonymous
    title = Column(String, default="New Conversation")  # Auto-generated from first message
    summary = Column(Text, nullable=True)  # Condensed conversation context
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    # Index for faster queries
    __table_args__ = (Index('idx_user_active', 'user_id', 'is_active'),)

class ChatMessage(Base):
    """
    Individual messages within a conversation.
    """
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey('conversations.conversation_id'), index=True)
    role = Column(String, index=True)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_metadata = Column(Text, nullable=True)  # JSON for chart_data, processing_time, etc.
    token_count = Column(Integer, nullable=True)  # For token usage tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Index for faster queries
    __table_args__ = (Index('idx_conversation_created', 'conversation_id', 'created_at'),)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Creating database tables...")
    create_tables()
    print("Database tables created successfully!")