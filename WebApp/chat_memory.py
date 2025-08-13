"""
Chat Memory Management Module
Handles conversation history, storage, and summarization for NAVUS.
"""

from sqlalchemy.orm import Session
from database import Conversation, ChatMessage, User
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

# Configuration constants
MAX_MESSAGES_PER_CONVERSATION = 100  # Start summarizing after this many messages
MAX_CONTEXT_TOKENS = 4000  # Approximate token limit for context
SUMMARY_TOKEN_TARGET = 500  # Target tokens for conversation summary


def get_or_create_conversation(db: Session, user_id: Optional[str] = None, conversation_id: Optional[str] = None) -> Conversation:
    """
    Get existing conversation or create a new one.
    
    Args:
        db: Database session
        user_id: User ID (None for anonymous users)
        conversation_id: Specific conversation to retrieve
    
    Returns:
        Conversation object
    """
    if conversation_id:
        # Get specific conversation
        conversation = db.query(Conversation).filter(
            Conversation.conversation_id == conversation_id,
            Conversation.is_active == True
        ).first()
        
        if conversation:
            return conversation
    
    # Get most recent active conversation for user
    if user_id:
        recent_conversation = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.is_active == True
        ).order_by(Conversation.updated_at.desc()).first()
        
        if recent_conversation:
            return recent_conversation
    
    # Create new conversation
    new_conversation = Conversation(user_id=user_id)
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    
    logger.info(f"Created new conversation: {new_conversation.conversation_id} for user: {user_id}")
    return new_conversation


def save_message(db: Session, conversation: Conversation, role: str, content: str, 
                metadata: Optional[Dict] = None) -> ChatMessage:
    """
    Save a chat message to the database.
    
    Args:
        db: Database session
        conversation: Conversation object
        role: 'user' or 'assistant'
        content: Message content
        metadata: Optional metadata (chart_data, processing_time, etc.)
    
    Returns:
        ChatMessage object
    """
    message = ChatMessage(
        conversation_id=conversation.conversation_id,
        role=role,
        content=content,
        message_metadata=json.dumps(metadata) if metadata else None,
        token_count=estimate_tokens(content)
    )
    
    db.add(message)
    
    # Update conversation
    conversation.message_count += 1
    conversation.updated_at = datetime.utcnow()
    
    # Auto-generate title from first user message
    if conversation.message_count == 1 and role == 'user':
        conversation.title = generate_conversation_title(content)
    
    db.commit()
    db.refresh(message)
    
    # Check if conversation needs summarization
    if conversation.message_count > MAX_MESSAGES_PER_CONVERSATION:
        asyncio.create_task(summarize_conversation_if_needed(db, conversation))
    
    return message


def get_conversation_history(db: Session, conversation_id: str, limit: int = 50) -> List[Dict]:
    """
    Get conversation history for context injection.
    
    Args:
        db: Database session
        conversation_id: Conversation ID
        limit: Maximum number of recent messages to retrieve
    
    Returns:
        List of message dictionaries
    """
    messages = db.query(ChatMessage).filter(
        ChatMessage.conversation_id == conversation_id
    ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
    
    # Reverse to get chronological order
    messages.reverse()
    
    history = []
    for message in messages:
        msg_dict = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.created_at.isoformat()
        }
        
        # Include metadata if present
        if message.message_metadata:
            try:
                msg_dict["metadata"] = json.loads(message.message_metadata)
            except json.JSONDecodeError:
                pass
        
        history.append(msg_dict)
    
    return history


def get_conversation_context(db: Session, conversation: Conversation) -> Tuple[str, List[Dict]]:
    """
    Get conversation context for AI prompt injection.
    Combines summary and recent messages to stay within token limits.
    
    Args:
        db: Database session
        conversation: Conversation object
    
    Returns:
        Tuple of (context_string, recent_messages_list)
    """
    context_parts = []
    
    # Add conversation summary if available
    if conversation.summary:
        context_parts.append(f"Previous conversation summary: {conversation.summary}")
    
    # Get recent messages
    recent_messages = get_conversation_history(db, conversation.conversation_id, limit=20)
    
    # Estimate total tokens and trim if necessary
    total_tokens = estimate_tokens("\n".join(context_parts))
    
    # Add messages while staying under token limit
    included_messages = []
    for message in reversed(recent_messages):  # Start with most recent
        message_tokens = estimate_tokens(message['content'])
        if total_tokens + message_tokens < MAX_CONTEXT_TOKENS:
            included_messages.insert(0, message)  # Insert at beginning
            total_tokens += message_tokens
        else:
            break
    
    # Create context string
    if context_parts:
        context_string = "\n".join(context_parts)
    else:
        context_string = ""
    
    return context_string, included_messages


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count.
    Generally 1 token ≈ 4 characters for English text.
    """
    return max(1, len(text) // 4)


def generate_conversation_title(first_message: str) -> str:
    """
    Generate a conversation title from the first user message.
    """
    # Clean and truncate the message
    title = first_message.strip()
    
    # Remove common question words and clean up
    title = title.replace("What's", "").replace("What is", "").replace("How do", "").replace("Can you", "")
    title = title.strip()
    
    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."
    
    # Default title if empty
    if not title:
        title = "Credit Card Advice"
    
    return title.title()


async def summarize_conversation_if_needed(db: Session, conversation: Conversation):
    """
    Summarize conversation if it's getting too long.
    This would typically call an AI service to create a summary.
    """
    try:
        if conversation.message_count > MAX_MESSAGES_PER_CONVERSATION and not conversation.summary:
            # Get all messages for summarization
            messages = db.query(ChatMessage).filter(
                ChatMessage.conversation_id == conversation.conversation_id
            ).order_by(ChatMessage.created_at.asc()).all()
            
            # Create a simple summary (in production, you'd use AI for this)
            user_questions = []
            assistant_topics = []
            
            for message in messages:
                if message.role == 'user':
                    user_questions.append(message.content[:100])
                elif message.role == 'assistant':
                    # Extract key topics (simplified)
                    if 'credit card' in message.content.lower():
                        assistant_topics.append("credit card recommendations")
                    if 'fee' in message.content.lower():
                        assistant_topics.append("fee analysis")
                    if 'reward' in message.content.lower():
                        assistant_topics.append("rewards discussion")
            
            # Create summary
            summary_parts = []
            if user_questions:
                summary_parts.append(f"User asked about: {', '.join(user_questions[:3])}")
            if assistant_topics:
                unique_topics = list(set(assistant_topics))
                summary_parts.append(f"Discussed: {', '.join(unique_topics[:5])}")
            
            conversation.summary = ". ".join(summary_parts) if summary_parts else "General credit card discussion"
            db.commit()
            
            logger.info(f"Created summary for conversation {conversation.conversation_id}")
            
    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")


def clear_conversation_history(db: Session, conversation_id: str, user_id: Optional[str] = None):
    """
    Clear conversation history for privacy.
    
    Args:
        db: Database session
        conversation_id: Conversation to clear
        user_id: User ID for authorization (optional for anonymous)
    """
    conversation = db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).first()
    
    if not conversation:
        raise ValueError("Conversation not found")
    
    # Verify ownership for logged-in users
    if user_id and conversation.user_id != user_id:
        raise ValueError("Unauthorized access")
    
    # Mark conversation as inactive and clear sensitive data
    conversation.is_active = False
    conversation.summary = None
    
    # Delete all messages
    db.query(ChatMessage).filter(
        ChatMessage.conversation_id == conversation_id
    ).delete()
    
    db.commit()
    logger.info(f"Cleared conversation history: {conversation_id}")


def get_user_conversations(db: Session, user_id: str, limit: int = 10) -> List[Dict]:
    """
    Get list of user's conversations for sidebar/history display.
    
    Args:
        db: Database session
        user_id: User ID
        limit: Maximum conversations to return
    
    Returns:
        List of conversation summaries
    """
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.is_active == True
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()
    
    result = []
    for conv in conversations:
        result.append({
            "conversation_id": conv.conversation_id,
            "title": conv.title,
            "message_count": conv.message_count,
            "updated_at": conv.updated_at.isoformat(),
            "created_at": conv.created_at.isoformat()
        })
    
    return result