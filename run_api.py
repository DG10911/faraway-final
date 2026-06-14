"""
RailGuard-FSL++ API Server

Start the backend API:
    python run_api.py
"""

if __name__ == "__main__":
    import uvicorn
    print("Starting RailGuard-FSL++ API Server...")
    print("Dashboard: http://localhost:3000")
    print("API Docs:  http://localhost:8000/docs")
    # reload disabled: a live reload wipes the in-memory model state (initialized
    # flag, prototypes) mid-demo. Keep it off so the system stays "ready".
    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=8000, reload=False)
