import json
import os
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class GoogleOAuth:
    def __init__(self):
        self.credentials_file = "google_credentials.json"
        self.client_config = None
        self.redirect_uri = None
        
        # Load Google OAuth configuration
        self.load_config()
    
    def load_config(self):
        """Load Google OAuth configuration from credentials file"""
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'r') as f:
                    credentials = json.load(f)
                
                self.client_config = credentials['web']
                
                # Determine redirect URI based on environment
                hostname = os.getenv('HOSTNAME', 'localhost')
                if hostname == 'localhost':
                    self.redirect_uri = "http://localhost:8003/auth/google/callback"
                else:
                    self.redirect_uri = "https://web-production-685ca.up.railway.app/auth/google/callback"
                
                logger.info(f"Google OAuth configured with redirect URI: {self.redirect_uri}")
                return True
            else:
                logger.warning("Google credentials file not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Google OAuth config: {e}")
            return False
    
    def get_authorization_url(self) -> Optional[str]:
        """Generate Google OAuth authorization URL"""
        try:
            if not self.client_config:
                return None
            
            flow = Flow.from_client_config(
                client_config={'web': self.client_config},
                scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
            )
            flow.redirect_uri = self.redirect_uri
            
            authorization_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='select_account'  # Force account selection
            )
            
            return authorization_url
            
        except Exception as e:
            logger.error(f"Failed to generate Google authorization URL: {e}")
            return None
    
    def exchange_code_for_token(self, authorization_code: str) -> Optional[Dict]:
        """Exchange authorization code for user information"""
        try:
            if not self.client_config:
                return None
            
            flow = Flow.from_client_config(
                client_config={'web': self.client_config},
                scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']
            )
            flow.redirect_uri = self.redirect_uri
            
            # Exchange authorization code for tokens
            flow.fetch_token(code=authorization_code)
            
            # Verify and decode the ID token with clock skew tolerance
            credentials = flow.credentials
            id_info = id_token.verify_oauth2_token(
                credentials.id_token,
                Request(),
                self.client_config['client_id'],
                clock_skew_in_seconds=60  # Allow 60 seconds clock skew
            )
            
            # Extract user information
            user_info = {
                'google_id': id_info.get('sub'),
                'email': id_info.get('email'),
                'name': id_info.get('name'),
                'picture': id_info.get('picture'),
                'email_verified': id_info.get('email_verified', False)
            }
            
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to exchange Google code for token: {e}")
            return None
    
    def is_configured(self) -> bool:
        """Check if Google OAuth is properly configured"""
        return self.client_config is not None

# Global instance
google_oauth = GoogleOAuth()