"""
FastAPI application for StockInsightAI
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import logging
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StockInsightAI",
    description="Indian Stock Market Analysis Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Import volume tracker after app is created to avoid circular imports
from volume_tracker import VolumeTracker

# Initialize volume tracker
volume_tracker = VolumeTracker()

# Models
class StockRequest(BaseModel):
    symbol: str
    days: int = 30

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/volume/top")
async def get_top_volume_stocks():
    """Get today's top volume stocks (most bought and most sold)"""
    logger.info("Fetching top volume stocks...")
    
    try:
        # Get the volume data
        result = volume_tracker.get_top_volume_stocks()
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error in volume tracker')
            logger.error(f"Error in volume tracker: {error_msg}")
            return JSONResponse(
                status_code=400,
                content={
                    'success': False,
                    'error': error_msg,
                    'data': {
                        'most_bought': [],
                        'most_sold': [],
                        'timestamp': datetime.now().isoformat()
                    }
                }
            )
            
        # Structure the response
        response_data = {
            'most_bought': result.get('most_bought', []),
            'most_sold': result.get('most_sold', []),
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        }
        
        logger.info(f"Returning data for {len(response_data['most_bought'])} bought and {len(response_data['most_sold'])} sold stocks")
        
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        logger.error(f"Error in /api/volume/top: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': f'Failed to fetch volume data: {str(e)}',
                'data': {
                    'most_bought': [],
                    'most_sold': [],
                    'timestamp': datetime.now().isoformat()
                }
            }
        )

@app.get("/api/volume/stock/{symbol}")
async def get_stock_volume_data(symbol: str):
    """Get volume data for a specific stock"""
    try:
        data = volume_tracker.get_stock_volume_data(symbol.upper())
        if data:
            return {"success": True, "data": data}
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"No data found for {symbol}"}
            )
    except Exception as e:
        logger.error(f"Error getting volume data for {symbol}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
