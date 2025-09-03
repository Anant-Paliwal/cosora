"""
Comprehensive Authentication API with JWT, MFA, Sessions, and Security
Based on modern authentication best practices
"""
from fastapi import FastAPI, HTTPException, Depends, Request, Response, Cookie, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta
import os
import time
from dotenv import load_dotenv

# Import our authentication services
from services.auth.passwords import PasswordService
from services.auth.tokens import TokenService
from services.auth.sessions import SessionService
from services.auth.mfa import MFAService
from services.common.rate_limit import RateLimitService
from services.common.audit import AuditService

# Load environment variables
load_dotenv()

app = FastAPI(title="Cosora IDE Authentication API", version="2.0.0")

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router)

# Initialize services
password_service = PasswordService()
token_service = TokenService()
session_service = SessionService()
mfa_service = MFAService()
rate_limit_service = RateLimitService()
audit_service = AuditService()
security = HTTPBearer()

# Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
REQUIRE_EMAIL_VERIFICATION = os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Pydantic models
class UserLogin(BaseModel):
    """User login model"""
    email_or_username: str
    password: str

class UserRegister(BaseModel):
    """User registration model"""
    email: EmailStr
    username: str
    password: str
    display_name: Optional[str] = None

    password: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class MFAVerify(BaseModel):
    code: str
    session_challenge_id: str

class MagicLinkRequest(BaseModel):
    email: EmailStr

# Helper functions
def get_client_ip(request: Request) -> str:
    """Extract client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

def get_device_fingerprint(request: Request) -> str:
    """Generate device fingerprint"""
    user_agent = request.headers.get("User-Agent", "")
    accept_lang = request.headers.get("Accept-Language", "")
    accept_encoding = request.headers.get("Accept-Encoding", "")
    
    fingerprint_data = f"{user_agent}:{accept_lang}:{accept_encoding}"
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]

def init_auth_database():
    """Initialize the SQLite database with required tables if they don't exist"""
    conn = sqlite3.connect('ide_platform.db')
    cursor = conn.cursor()
    
    try:
        # Check if users table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='users'
        """)
        
        if not cursor.fetchone():
            # Read schema from file
            with open('database_schema.sql', 'r') as f:
                schema = f.read()
            
            # Execute schema
            cursor.executescript(schema)
            conn.commit()
            print("Database tables created successfully")
        else:
            print("Database tables already exist, skipping creation")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

# Initialize database if not already done
init_auth_database()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    payload = token_service.verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    conn = sqlite3.connect('ide_platform.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, email, display_name, email_verified, disabled
            FROM users WHERE id = ?
        """, (payload["user_id"],))
        
        user = cursor.fetchone()
        if not user or user[4]:  # disabled
            raise HTTPException(status_code=401, detail="User not found or disabled")
        
        return {
            "id": user[0],
            "email": user[1],
            "display_name": user[2],
            "email_verified": user[3]
        }
    finally:
        conn.close()

# API Endpoints

@api_router.post("/auth/register")
async def register(user_data: UserRegister, request: Request):
    """Register new user with email verification"""
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    # Rate limiting
    rate_key = f"register:{ip_address}"
    allowed, blocked_until = rate_limit_service.check_rate_limit(rate_key, max_attempts=3)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Too many registration attempts. Try again after {blocked_until}")
    
    # Validate password strength
    valid, message = password_service.validate_password_strength(user_data.password)
    if not valid:
        rate_limit_service.record_attempt(rate_key)
        raise HTTPException(status_code=400, detail=message)
    
    conn = sqlite3.connect('ide_platform.db')
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email.lower(),))
        if cursor.fetchone():
            rate_limit_service.record_attempt(rate_key)
            raise HTTPException(status_code=409, detail="Email already registered")
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        password_hash = password_service.hash_password(user_data.password)
        
        cursor.execute("""
            INSERT INTO users (id, email, password_hash, display_name)
            VALUES (?, ?, ?, ?)
        """, (user_id, user_data.email.lower(), password_hash, user_data.display_name))
        
        conn.commit()
        
        # Log registration
        audit_service.log_registration(user_id, user_data.email, ip_address, user_agent)
        
        # Create session
        device_fingerprint = get_device_fingerprint(request)
        session_data = session_service.create_session(
            user_id, device_fingerprint, user_agent, ip_address
        )
        
        # Create access token
        access_token = token_service.create_access_token(user_id, user_data.email)
        
        # Set refresh token as httpOnly cookie
        response = Response()
        response.set_cookie(
            key="refresh_token",
            value=session_data["refresh_token"],
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=int(token_service.refresh_token_expire.total_seconds())
        )
        
        return {
            "success": True,
            "message": "Registration successful",
            "access_token": access_token,
            "user": {
                "id": user_id,
                "email": user_data.email,
                "display_name": user_data.display_name,
                "email_verified": False
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        conn.close()

@api_router.post("/auth/login")
async def login(login_data: UserLogin, request: Request, response: Response):
    """Login with email/password, support MFA"""
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    # Rate limiting
    rate_key = f"login:{ip_address}"
    allowed, blocked_until = rate_limit_service.check_rate_limit(rate_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Too many login attempts. Try again after {blocked_until}")
    
    conn = sqlite3.connect('ide_platform.db')
    cursor = conn.cursor()
    
    try:
        # Get user
        cursor.execute("""
            SELECT id, email, password_hash, display_name, email_verified, disabled
            FROM users WHERE email = ?
        """, (login_data.email.lower(),))
        
        user = cursor.fetchone()
        if not user or user[5]:  # not found or disabled
            rate_limit_service.record_attempt(rate_key)
            audit_service.log_login_attempt(login_data.email, False, ip_address, user_agent)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not password_service.verify_password(login_data.password, user[2]):
            rate_limit_service.record_attempt(rate_key)
            audit_service.log_login_attempt(login_data.email, False, ip_address, user_agent, user[0])
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_id, email, _, display_name, email_verified, _ = user
        
        # Check if MFA is required
        if mfa_service.is_mfa_enabled(user_id):
            # Create MFA challenge token
            challenge_token = token_service.create_mfa_challenge_token(user_id)
            return {
                "mfa_required": True,
                "session_challenge_id": challenge_token,
                "message": "MFA verification required"
            }
        
        # Create session and tokens
        device_fingerprint = get_device_fingerprint(request)
        session_data = session_service.create_session(
            user_id, device_fingerprint, user_agent, ip_address
        )
        
        access_token = token_service.create_access_token(user_id, email)
        
        # Set refresh token cookie
        response.set_cookie(
            key="refresh_token",
            value=session_data["refresh_token"],
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=int(token_service.refresh_token_expire.total_seconds())
        )
        
        # Reset rate limit on successful login
        rate_limit_service.reset_rate_limit(rate_key)
        
        # Log successful login
        audit_service.log_login_attempt(email, True, ip_address, user_agent, user_id)
        audit_service.log_session_created(user_id, session_data["session_id"], ip_address, user_agent)
        
        return {
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "user": {
                "id": user_id,
                "email": email,
                "display_name": display_name,
                "email_verified": email_verified
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        conn.close()

@api_router.post("/auth/mfa/verify")
async def verify_mfa(mfa_data: MFAVerify, request: Request, response: Response):
    """Verify MFA code and complete login"""
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    # Verify challenge token
    user_id = token_service.verify_mfa_challenge_token(mfa_data.session_challenge_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA challenge")
    
    # Verify MFA code
    valid, code_type = mfa_service.verify_mfa_code(user_id, mfa_data.code)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid MFA code")
    
    # Get user info
    conn = sqlite3.connect('ide_platform.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT email, display_name, email_verified
            FROM users WHERE id = ?
        """, (user_id,))
        
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        email, display_name, email_verified = user
        
        # Create session and tokens
        device_fingerprint = get_device_fingerprint(request)
        session_data = session_service.create_session(
            user_id, device_fingerprint, user_agent, ip_address
        )
        
        access_token = token_service.create_access_token(user_id, email)
        
        # Set refresh token cookie
        response.set_cookie(
            key="refresh_token",
            value=session_data["refresh_token"],
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=int(token_service.refresh_token_expire.total_seconds())
        )
        
        # Log successful MFA login
        audit_service.log_login_attempt(email, True, ip_address, user_agent, user_id)
        audit_service.log_session_created(user_id, session_data["session_id"], ip_address, user_agent)
        
        return {
            "success": True,
            "message": "MFA verification successful",
            "access_token": access_token,
            "user": {
                "id": user_id,
                "email": email,
                "display_name": display_name,
                "email_verified": email_verified
            }
        }
        
    finally:
        conn.close()

@api_router.post("/auth/refresh")
async def refresh_token(request: Request, response: Response, 
                       refresh_token: Optional[str] = Cookie(None)):
    """Refresh access token using refresh token"""
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh token required")
    
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    device_fingerprint = get_device_fingerprint(request)
    
    # Rotate refresh token
    session_data = session_service.rotate_refresh_token(
        refresh_token, device_fingerprint, user_agent, ip_address
    )
    
    if not session_data:
        response.delete_cookie("refresh_token")
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    # Create new access token
    access_token = token_service.create_access_token(
        session_data["user_id"], session_data["email"]
    )
    
    # Set new refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=session_data["refresh_token"],
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=int(token_service.refresh_token_expire.total_seconds())
    )
    
    # Log token refresh
    audit_service.log_token_refresh(
        session_data["user_id"], session_data["session_id"], ip_address, user_agent
    )
    
    return {
        "success": True,
        "access_token": access_token
    }

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, 
                refresh_token: Optional[str] = Cookie(None),
                current_user: dict = Depends(get_current_user)):
    """Logout and revoke session"""
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    if refresh_token:
        # Find and revoke session
        session_info = session_service.validate_refresh_token(refresh_token)
        if session_info:
            session_service.revoke_session(session_info["session_id"])
            audit_service.log_session_revoked(
                current_user["id"], session_info["session_id"], ip_address, user_agent
            )
    
    # Clear refresh token cookie
    response.delete_cookie("refresh_token")
    
    return {"success": True, "message": "Logged out successfully"}

@api_router.get("/auth/sessions")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    """Get user's active sessions"""
    sessions = session_service.get_user_sessions(current_user["id"])
    return {"sessions": sessions}

@api_router.post("/auth/sessions/revoke")
async def revoke_session(session_id: str, request: Request,
                        current_user: dict = Depends(get_current_user)):
    """Revoke specific session"""
    ip_address = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    success = session_service.revoke_session(session_id)
    if success:
        audit_service.log_session_revoked(
            current_user["id"], session_id, ip_address, user_agent
        )
        return {"success": True, "message": "Session revoked"}
    
    raise HTTPException(status_code=404, detail="Session not found")

@api_router.post("/auth/mfa/setup")
async def setup_mfa(current_user: dict = Depends(get_current_user)):
    """Setup TOTP MFA for user"""
    mfa_data = mfa_service.generate_totp_secret(current_user["email"])
    recovery_codes = mfa_service.generate_recovery_codes()
    
    return {
        "secret": mfa_data["secret"],
        "qr_code": mfa_data["qr_code"],
        "manual_entry_key": mfa_data["manual_entry_key"],
        "recovery_codes": recovery_codes
    }

@api_router.post("/auth/mfa/enable")
async def enable_mfa(code: str, secret: str, recovery_codes: List[str],
                    request: Request, current_user: dict = Depends(get_current_user)):
    """Enable MFA after verifying TOTP code"""
    if not mfa_service.verify_totp_code(secret, code):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")
    
    success = mfa_service.setup_mfa(current_user["id"], secret, recovery_codes)
    if success:
        ip_address = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        audit_service.log_mfa_setup(current_user["id"], ip_address, user_agent)
        return {"success": True, "message": "MFA enabled successfully"}
    
    raise HTTPException(status_code=500, detail="Failed to enable MFA")

@app.get("/")
async def root():
    return {"message": "Cosora IDE Authentication API v2.0", "status": "running"}

@api_router.get("/health")
async def health_check():
    return {"status": "ok", "service": "auth"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
