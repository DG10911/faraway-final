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
    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["backend"])
