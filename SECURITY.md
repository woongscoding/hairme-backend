# Security Implementation Guide

## Overview

This document describes the security measures implemented in the HairMe Backend API to protect against common vulnerabilities and attacks.

## Security Features

### 1. File Upload Size Limitation (10MB)

**Location:** `main.py:54-67`

**Purpose:** Prevent DoS attacks and resource exhaustion

**Implementation:**
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
    return await call_next(request)
```

**Benefits:**
- Prevents large file uploads that could crash the server
- Protects against memory exhaustion
- Reduces Lambda timeout risks

---

### 2. Rate Limiting

**Location:**
- `main.py:28-40` (Limiter initialization)
- `api/endpoints/analyze.py:438,619,775`
- `api/endpoints/feedback.py:24,168`

**Purpose:** Prevent DoS attacks and API abuse

**Implementation:**
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/analyze")
@limiter.limit("10/minute")
async def analyze_face(request: Request, file: UploadFile = File(...)):
    # ...
```

**Rate Limits by Endpoint:**
- `/api/analyze` - 10 requests/minute
- `/api/v2/analyze-hybrid` - 10 requests/minute
- `/api/v2/feedback` - 20 requests/minute
- `/api/feedback` - 20 requests/minute
- `/api/stats/feedback` - 30 requests/minute

**Benefits:**
- Prevents API abuse and DoS attacks
- Controls Gemini API costs (paid per request)
- Protects server resources

---

### 3. Admin API Authentication

**Location:**
- `core/auth.py` (Authentication module)
- `routers/admin.py` (Protected endpoints)

**Purpose:** Secure sensitive admin endpoints

**Implementation:**

**Step 1:** Set admin API key in environment:
```bash
# Generate a secure random key
openssl rand -hex 32

# Add to .env
ADMIN_API_KEY=your_generated_key_here
```

**Step 2:** Use X-API-Key header in requests:
```bash
curl -H "X-API-Key: your_generated_key_here" \
  https://api.hairme.app/api/admin/feedback-stats
```

**Protected Endpoints:**
- `GET /api/admin/feedback-stats` - Feedback statistics
- `GET /api/admin/feedback-distribution` - Distribution by face shape/skin tone
- `GET /api/admin/top-hairstyles` - Top liked/disliked hairstyles
- `GET /api/admin/retrain-status` - ML model retraining status

**Authentication Flow:**
```python
from core.auth import verify_admin_api_key

@router.get("/admin/feedback-stats")
async def get_feedback_stats(api_key: str = Depends(verify_admin_api_key)):
    # Only executes if API key is valid
    # ...
```

**Benefits:**
- Prevents unauthorized access to sensitive data
- Protects admin-only operations
- Easy to rotate credentials

---

### 4. CORS Configuration

**Location:** `config/settings.py:29-35`, `main.py:44-50`

**Purpose:** Control which domains can access the API

**Implementation:**
```python
# config/settings.py
ALLOWED_ORIGINS: str = "http://localhost:3000"

@property
def allowed_origins_list(self) -> List[str]:
    return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    # ...
)
```

**Configuration:**
```bash
# .env - Multiple domains supported (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,https://hairme.app,https://www.hairme.app
```

**Benefits:**
- Prevents unauthorized cross-origin requests
- Easy to add/remove domains via environment variable
- Supports multiple production domains

---

### 5. API Key Validation on Startup

**Location:** `main.py:82-85`

**Purpose:** Ensure critical configuration is set before accepting requests

**Implementation:**
```python
@app.on_event("startup")
async def startup_event():
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")
    # ...
```

**Benefits:**
- Fails fast if misconfigured
- Prevents runtime errors from missing API keys
- Clear error messages for developers

---

## Security Checklist

### Before Deployment

- [ ] Set `ADMIN_API_KEY` environment variable
- [ ] Configure `ALLOWED_ORIGINS` with production domains
- [ ] Verify `GEMINI_API_KEY` is set
- [ ] Test rate limiting is working
- [ ] Test file size limitation (try uploading 11MB file)
- [ ] Test admin endpoints require API key
- [ ] Set `DEBUG=false` in production
- [ ] Review CloudWatch logs for security events

### After Deployment

- [ ] Monitor rate limit violations in logs
- [ ] Monitor 403 errors (unauthorized admin access attempts)
- [ ] Monitor 413 errors (file too large attempts)
- [ ] Set up alerts for unusual traffic patterns
- [ ] Rotate `ADMIN_API_KEY` quarterly

---

## API Usage Examples

### Analyze Face (Public Endpoint)
```bash
curl -X POST "https://api.hairme.app/api/analyze" \
  -F "file=@photo.jpg"
```

### Get Admin Statistics (Protected)
```bash
curl -H "X-API-Key: your_admin_key" \
  "https://api.hairme.app/api/admin/feedback-stats"
```

### Rate Limit Response (429 Too Many Requests)
```json
{
  "error": "Rate limit exceeded: 10 per 1 minute"
}
```

### File Too Large (413 Payload Too Large)
```json
{
  "detail": "File too large. Maximum size is 10MB"
}
```

### Unauthorized Admin Access (403 Forbidden)
```json
{
  "detail": "Invalid API Key"
}
```

---

## Cost Protection

### Gemini API Cost Control

Rate limiting prevents cost overruns from Gemini API:
- **Without rate limit:** Unlimited requests → $$$
- **With 10/min limit:** Max 14,400 requests/day = controlled costs

**Cost Calculation:**
```
10 requests/min × 60 min × 24 hours = 14,400 requests/day
14,400 × $0.001/request = $14.40/day max
```

### AWS Lambda Cost Control

File size limit prevents Lambda timeout and memory errors:
- **Without limit:** 50MB image → 30s processing → timeout
- **With 10MB limit:** Reasonable processing time, no timeouts

---

## Troubleshooting

### Rate Limit Issues

**Problem:** Legitimate users hitting rate limits

**Solution:** Increase limits for specific endpoints
```python
@limiter.limit("20/minute")  # Increased from 10
```

### Admin API Access

**Problem:** 403 Forbidden on admin endpoints

**Solutions:**
1. Check `ADMIN_API_KEY` is set in environment
2. Verify `X-API-Key` header is included in request
3. Ensure API key matches exactly (no extra spaces)

### CORS Errors

**Problem:** "CORS policy" errors in browser console

**Solution:** Add frontend domain to `ALLOWED_ORIGINS`
```bash
ALLOWED_ORIGINS=http://localhost:3000,https://hairme.app,https://new-domain.com
```

---

## Future Enhancements

### Recommended Improvements

1. **JWT Authentication** - Token-based auth for user sessions
2. **API Key Scopes** - Different keys for different permission levels
3. **IP Whitelisting** - Restrict admin endpoints to specific IPs
4. **Request Signing** - HMAC signatures for request integrity
5. **Web Application Firewall** - AWS WAF for additional protection

### AWS API Gateway Integration

For production, consider using AWS API Gateway:
- Built-in rate limiting
- API key management
- Request validation
- DDoS protection
- Usage plans and throttling

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [slowapi Documentation](https://github.com/laurentS/slowapi)
- [AWS WAF](https://aws.amazon.com/waf/)

---

**Last Updated:** 2025-11-16
**Version:** 20.2.0
