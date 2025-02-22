"""
Launcher script for the DoubleSparse API server.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=52309,
        reload=True,
        ssl_keyfile=None,  # Add SSL if needed
        ssl_certfile=None,  # Add SSL if needed
        proxy_headers=True,
        forwarded_allow_ips="*",
    )