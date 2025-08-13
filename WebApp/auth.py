import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from database import User, Session as DBSession
import secrets
import uuid

# JWT Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"  # Should be in environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
PERSISTENT_TOKEN_EXPIRE_DAYS = 30

class AuthError(Exception):
    pass

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired")
    except jwt.JWTError:
        raise AuthError("Invalid token")

def create_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

def create_user(db: Session, email: str, name: str, password: str = None) -> User:
    """Create a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise AuthError("Email already registered")
    
    # Create new user
    password_hash = hash_password(password) if password else None
    user = User(
        user_id=str(uuid.uuid4()),
        email=email,
        name=name,
        password_hash=password_hash
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_or_get_google_user(db: Session, google_user_info: dict) -> User:
    """Create or retrieve a Google OAuth user"""
    # Check if user exists by Google ID
    user = db.query(User).filter(User.google_id == google_user_info['google_id']).first()
    if user:
        # Update user info in case it changed
        user.name = google_user_info['name']
        user.profile_picture = google_user_info.get('picture')
        db.commit()
        return user
    
    # Check if user exists by email
    user = db.query(User).filter(User.email == google_user_info['email']).first()
    if user:
        # Link existing account with Google
        user.google_id = google_user_info['google_id']
        user.profile_picture = google_user_info.get('picture')
        user.oauth_provider = 'google'
        db.commit()
        return user
    
    # Create new Google user
    user = User(
        user_id=str(uuid.uuid4()),
        email=google_user_info['email'],
        name=google_user_info['name'],
        google_id=google_user_info['google_id'],
        profile_picture=google_user_info.get('picture'),
        oauth_provider='google',
        password_hash=None  # No password for OAuth users
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_or_get_twitch_user(db: Session, twitch_user_info: dict) -> User:
    """Create or retrieve a Twitch OAuth user"""
    # Check if user exists by Twitch ID
    user = db.query(User).filter(User.twitch_id == twitch_user_info['twitch_id']).first()
    if user:
        # Update user info in case it changed
        user.name = twitch_user_info['display_name']
        user.twitch_login = twitch_user_info['login']
        user.profile_picture = twitch_user_info.get('profile_image_url')
        db.commit()
        return user
    
    # Check if user exists by email (if email is provided)
    if twitch_user_info.get('email'):
        user = db.query(User).filter(User.email == twitch_user_info['email']).first()
        if user:
            # Link existing account with Twitch
            user.twitch_id = twitch_user_info['twitch_id']
            user.twitch_login = twitch_user_info['login']
            user.profile_picture = twitch_user_info.get('profile_image_url')
            user.oauth_provider = 'twitch'
            db.commit()
            return user
    
    # Create new Twitch user
    # Use display_name@twitch.local if no email provided
    email = twitch_user_info.get('email') or f"{twitch_user_info['login']}@twitch.local"
    
    user = User(
        user_id=str(uuid.uuid4()),
        email=email,
        name=twitch_user_info['display_name'],
        twitch_id=twitch_user_info['twitch_id'],
        twitch_login=twitch_user_info['login'],
        profile_picture=twitch_user_info.get('profile_image_url'),
        oauth_provider='twitch',
        password_hash=None  # No password for OAuth users
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_session(db: Session, user_id: str, is_persistent: bool = False) -> str:
    """Create a new session for a user"""
    session_token = create_session_token()
    
    if is_persistent:
        expires_at = datetime.utcnow() + timedelta(days=PERSISTENT_TOKEN_EXPIRE_DAYS)
    else:
        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES * 2)
    
    session = DBSession(
        session_token=session_token,
        user_id=user_id,
        expires_at=expires_at,
        is_persistent=is_persistent
    )
    
    db.add(session)
    db.commit()
    return session_token

def get_user_by_session(db: Session, session_token: str) -> Optional[User]:
    """Get user by session token"""
    session = db.query(DBSession).filter(
        DBSession.session_token == session_token,
        DBSession.expires_at > datetime.utcnow()
    ).first()
    
    if not session:
        return None
    
    user = db.query(User).filter(User.user_id == session.user_id).first()
    return user

def validate_password_strength(password: str) -> dict:
    """Validate password strength and return feedback"""
    issues = []
    score = 0
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    else:
        score += 1
    
    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")
    else:
        score += 1
    
    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")
    else:
        score += 1
    
    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one number")
    else:
        score += 1
    
    if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in password):
        issues.append("Password must contain at least one special character")
    else:
        score += 1
    
    return {
        "score": score,
        "max_score": 5,
        "is_strong": score >= 4,
        "issues": issues
    }

def logout_session(db: Session, session_token: str) -> bool:
    """Logout a user by invalidating their session"""
    session = db.query(DBSession).filter(DBSession.session_token == session_token).first()
    if session:
        db.delete(session)
        db.commit()
        return True
    return False