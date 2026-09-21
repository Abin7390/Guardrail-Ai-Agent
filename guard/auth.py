"""JWT (HS256) bearer auth for the Guardrail API.

Secret and expiry come from env (``GUARD_JWT_SECRET`` /
``GUARD_JWT_EXPIRE_MINUTES``) and are read per call so tests can patch them.
The DB row is the source of truth for roles: ``require_admin`` re-checks the
current DB role, not the token claim.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from guard.db import User, get_session

ALGORITHM = "HS256"
DEFAULT_SECRET_KEY = "guardrail-poc-dev-secret-change-me"
DEFAULT_EXPIRE_MINUTES = 1440


def get_secret_key() -> str:
    return os.environ.get("GUARD_JWT_SECRET", DEFAULT_SECRET_KEY)


def get_expire_minutes() -> int:
    return int(os.environ.get("GUARD_JWT_EXPIRE_MINUTES", str(DEFAULT_EXPIRE_MINUTES)))


def create_access_token(username: str, role: str, expires_minutes: int | None = None) -> str:
    now = datetime.now(timezone.utc)
    if expires_minutes is None:
        expires_minutes = get_expire_minutes()
    claims = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(claims, get_secret_key(), algorithm=ALGORITHM)


bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    session: Session = Depends(get_session),
) -> User:
    if credentials is None:
        raise _unauthorized("Not authenticated")
    try:
        claims = jwt.decode(credentials.credentials, get_secret_key(), algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise _unauthorized("Invalid or expired token") from exc
    username = claims.get("sub")
    if not isinstance(username, str):
        raise _unauthorized("Invalid or expired token")
    user = session.scalar(select(User).where(User.username == username))
    if user is None:
        raise _unauthorized("Invalid or expired token")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
