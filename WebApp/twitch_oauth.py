import os
import json
import secrets
import base64
import hashlib
from typing import Optional, Dict
import httpx
import logging

logger = logging.getLogger(__name__)

class TwitchOAuth:
    def __init__(self):
        # Twitch OAuth configuration
        self.client_id = os.getenv("TWITCH_CLIENT_ID", "")
        self.client_secret = os.getenv("TWITCH_CLIENT_SECRET", "")
        self.redirect_uri = None
        
        # Twitch OAuth URLs
        self.auth_url = "https://id.twitch.tv/oauth2/authorize"
        self.token_url = "https://id.twitch.tv/oauth2/token"
        self.user_url = "https://api.twitch.tv/helix/users"
        self.validate_url = "https://id.twitch.tv/oauth2/validate"
        
        # Determine redirect URI based on environment
        self.set_redirect_uri()
    
    def set_redirect_uri(self):
        """Set redirect URI based on environment"""
        hostname = os.getenv('HOSTNAME', 'localhost')
        if hostname == 'localhost':
            self.redirect_uri = "http://localhost:8003/auth/twitch/callback"
        else:
            self.redirect_uri = "https://web-production-685ca.up.railway.app/auth/twitch/callback"
        
        logger.info(f"Twitch OAuth configured with redirect URI: {self.redirect_uri}")
    
    def generate_state(self) -> str:
        """Generate a secure state parameter for OAuth"""
        return secrets.token_urlsafe(32)
    
    def get_authorization_url(self, state: str) -> Optional[str]:
        """Generate Twitch OAuth authorization URL"""
        try:
            if not self.client_id:
                logger.error("Twitch Client ID not configured")
                return None
            
            # Twitch OAuth scopes
            scopes = ["user:read:email", "openid"]
            
            # Build authorization URL
            params = {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": " ".join(scopes),
                "state": state
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            authorization_url = f"{self.auth_url}?{query_string}"
            
            logger.info("Generated Twitch authorization URL")
            return authorization_url
            
        except Exception as e:
            logger.error(f"Failed to generate Twitch authorization URL: {e}")
            return None
    
    async def exchange_code_for_token(self, authorization_code: str) -> Optional[str]:
        """Exchange authorization code for access token"""
        try:
            if not self.client_id or not self.client_secret:
                logger.error("Twitch OAuth credentials not configured")
                return None
            
            # Token exchange data
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": authorization_code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.token_url, data=data)
                
                if response.status_code == 200:
                    token_data = response.json()
                    return token_data.get("access_token")
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to exchange Twitch code for token: {e}")
            return None
    
    async def get_user_info(self, access_token: str) -> Optional[Dict]:
        """Get user information from Twitch API"""
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Client-Id": self.client_id
            }
            
            async with httpx.AsyncClient() as client:
                # Get user information
                response = await client.get(self.user_url, headers=headers)
                
                if response.status_code == 200:
                    user_data = response.json()
                    
                    if user_data.get("data"):
                        user = user_data["data"][0]
                        
                        user_info = {
                            "twitch_id": user.get("id"),
                            "login": user.get("login"),
                            "display_name": user.get("display_name"),
                            "email": user.get("email"),
                            "profile_image_url": user.get("profile_image_url"),
                            "created_at": user.get("created_at")
                        }
                        
                        logger.info(f"Retrieved Twitch user info for: {user_info.get('login')}")
                        return user_info
                    else:
                        logger.error("No user data returned from Twitch API")
                        return None
                else:
                    logger.error(f"Failed to get user info: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get Twitch user info: {e}")
            return None
    
    async def validate_token(self, access_token: str) -> bool:
        """Validate Twitch access token"""
        try:
            headers = {"Authorization": f"OAuth {access_token}"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(self.validate_url, headers=headers)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Failed to validate Twitch token: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if Twitch OAuth is properly configured"""
        return bool(self.client_id and self.client_secret)

# Global instance
twitch_oauth = TwitchOAuth()