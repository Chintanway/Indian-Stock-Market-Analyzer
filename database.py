import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# SQLite database configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'stockinsight.db')}")

# Configure SQLite engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=True  # Enable SQL query logging for debugging
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StockAnalysis(Base):
    """Store stock analysis results"""
    __tablename__ = "stock_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    current_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    signal = Column(String(20), nullable=False)  # BUY, SELL, CAUTION, NEUTRAL
    confidence = Column(String(20), nullable=False)  # High, Medium, Low
    
    # Technical indicators
    rsi = Column(Float)
    ema_5 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    sma_20 = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    
    # Model performance
    train_accuracy = Column(Float)
    test_accuracy = Column(Float)
    prediction_change = Column(Float)
    
    # Metadata
    data_timestamp = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
class UserSearch(Base):
    """Track user search patterns"""
    __tablename__ = "user_searches"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    search_count = Column(Integer, default=1)
    last_searched = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPerformance(Base):
    """Track model performance over time"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float)  # To be updated later
    accuracy_percentage = Column(Float)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    verification_date = Column(DateTime)
    is_verified = Column(Boolean, default=False)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_stock_analysis(analysis_data):
    """Save stock analysis to database"""
    db = SessionLocal()
    try:
        # Check if analysis for this symbol exists today
        today = datetime.utcnow().date()
        existing = db.query(StockAnalysis).filter(
            StockAnalysis.symbol == analysis_data['symbol'],
            StockAnalysis.created_at >= today
        ).first()
        
        if existing:
            # Update existing record
            for key, value in analysis_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
        else:
            # Create new record
            analysis = StockAnalysis(**analysis_data)
            db.add(analysis)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving analysis: {e}")
        return False
    finally:
        db.close()

def update_user_search(symbol):
    """Track user search patterns with better error handling"""
    db = SessionLocal()
    try:
        # Try to find existing search record
        search = db.query(UserSearch).filter(UserSearch.symbol == symbol).first()
        if search:
            search.search_count = search.search_count + 1
            search.last_searched = datetime.utcnow()
        else:
            search = UserSearch(symbol=symbol)
            db.add(search)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating search: {e}")
        # Try to create new session if connection failed
        try:
            db.close()
            db = SessionLocal()
            search = UserSearch(symbol=symbol, search_count=1)
            db.add(search)
            db.commit()
            return True
        except Exception as retry_error:
            print(f"Retry failed: {retry_error}")
            return False
    finally:
        try:
            db.close()
        except:
            pass

def get_popular_stocks(limit=10):
    """Get most searched stocks"""
    db = SessionLocal()
    try:
        popular = db.query(UserSearch).order_by(
            UserSearch.search_count.desc()
        ).limit(limit).all()
        return [{"symbol": stock.symbol, "count": stock.search_count} for stock in popular]
    except Exception as e:
        print(f"Error getting popular stocks: {e}")
        return []
    finally:
        db.close()

def get_analysis_history(symbol, limit=10):
    """Get analysis history for a symbol"""
    db = SessionLocal()
    try:
        history = db.query(StockAnalysis).filter(
            StockAnalysis.symbol == symbol
        ).order_by(StockAnalysis.created_at.desc()).limit(limit).all()
        
        return [{
            "date": analysis.created_at.strftime("%Y-%m-%d %H:%M"),
            "current_price": analysis.current_price,
            "predicted_price": analysis.predicted_price,
            "signal": analysis.signal,
            "prediction_change": analysis.prediction_change
        } for analysis in history]
    except Exception as e:
        print(f"Error getting analysis history: {e}")
        return []
    finally:
        db.close()

def save_model_performance(symbol, predicted_price):
    """Save model prediction for later verification"""
    db = SessionLocal()
    try:
        performance = ModelPerformance(
            symbol=symbol,
            predicted_price=predicted_price
        )
        db.add(performance)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving model performance: {e}")
        return False
    finally:
        db.close()

def get_model_accuracy_stats():
    """Get overall model accuracy statistics"""
    db = SessionLocal()
    try:
        verified = db.query(ModelPerformance).filter(
            ModelPerformance.is_verified == True
        ).all()
        
        if not verified:
            return {"total_predictions": 0, "average_accuracy": 0}
        
        total_accuracy = sum(p.accuracy_percentage for p in verified if p.accuracy_percentage is not None)
        count = len([p for p in verified if p.accuracy_percentage is not None])
        
        return {
            "total_predictions": len(verified),
            "average_accuracy": total_accuracy / count if count > 0 else 0
        }
    except Exception as e:
        print(f"Error getting accuracy stats: {e}")
        return {"total_predictions": 0, "average_accuracy": 0}
    finally:
        db.close()