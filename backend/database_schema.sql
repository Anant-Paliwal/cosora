-- Comprehensive Authentication Database Schema
-- Based on modern security best practices

-- Enable UUID extension (PostgreSQL)
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with email verification
CREATE TABLE users (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    disabled BOOLEAN DEFAULT FALSE
);

-- Device sessions for refresh token management
CREATE TABLE auth_sessions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_fingerprint TEXT,
    user_agent TEXT,
    ip_address TEXT,
    refresh_token_hash TEXT NOT NULL,
    refresh_expires_at TIMESTAMP NOT NULL,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Multi-factor authentication
CREATE TABLE user_mfa (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    totp_secret TEXT, -- encrypted at rest
    recovery_codes TEXT, -- JSON array of hashed codes
    enabled BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One-time codes for email verification, password reset, magic links
CREATE TABLE auth_one_time_codes (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    purpose TEXT CHECK (purpose IN ('verify_email', 'reset_password', 'login', 'mfa_challenge')) NOT NULL,
    code_hash TEXT NOT NULL,
    consumed BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Authentication audit log
CREATE TABLE auth_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    event TEXT NOT NULL,
    meta TEXT, -- JSON string
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rate limiting table
CREATE TABLE rate_limits (
    id TEXT PRIMARY KEY,
    attempts INTEGER DEFAULT 0,
    window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    blocked_until TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX idx_auth_sessions_refresh_hash ON auth_sessions(refresh_token_hash);
CREATE INDEX idx_auth_one_time_codes_user_id ON auth_one_time_codes(user_id);
CREATE INDEX idx_auth_one_time_codes_purpose ON auth_one_time_codes(purpose);
CREATE INDEX idx_auth_audit_user_id ON auth_audit(user_id);
CREATE INDEX idx_auth_audit_event ON auth_audit(event);
CREATE INDEX idx_rate_limits_window ON rate_limits(window_start);
