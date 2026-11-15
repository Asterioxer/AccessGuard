from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory sliding window rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
