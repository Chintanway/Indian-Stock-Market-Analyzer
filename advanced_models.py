"""
Advanced AI Models for Stock Prediction
- XGBoost with hyperparameter tuning
- Feature importance analysis
- Advanced technical indicators
- Prediction intervals
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importances_ = None
        self.model_path = "advanced_stock_model.joblib"
        self.scaler_path = "advanced_scaler.joblib"
        
    def calculate_indicators(self, data):
        """Enhanced technical indicators calculation"""
        df = data.copy()
        
        # Ensure numeric data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Price action features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log1p(df['Returns'])
        df['Range'] = df['High'] - df['Low']
        df['Body'] = (df['Close'] - df['Open']).abs()
        df['Body/Range'] = df['Body'] / df['Range'].replace(0, 0.001)
        
        # Volatility indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Momentum indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Stoch_RSI'] = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)['STOCHRSIk_14_14_3_3']
        df['ROC'] = ta.roc(df['Close'], length=5)
        df['MACD_Line'] = ta.macd(df['Close'])['MACD_12_26_9']
        df['MACD_Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
        df['MACD_Hist'] = ta.macd(df['Close'])['MACDh_12_26_9']
        
        # Volume indicators
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
        df['VPT'] = ta.pvi(df['Close'], df['Volume']) + ta.nvi(df['Close'], df['Volume'])
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = ta.sma(df['Close'], length=period)
            df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
        
        # Price to Moving Averages ratios
        for period in [20, 50, 200]:
            df[f'Close_SMA{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
            df[f'Close_EMA{period}_Ratio'] = df['Close'] / df[f'EMA_{period}']
        
        # Pattern recognition
        df['CDL_DOJI'] = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])
        df['CDL_ENGULFING'] = ta.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
        
        # Drop any remaining NaN values
        df = df.dropna()
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for the model"""
        # Calculate all indicators
        df = self.calculate_indicators(data)
        
        # Define feature columns (excluding target and date columns)
        feature_columns = [col for col in df.columns if col not in ['Date', 'Target'] and not col.startswith('CDL_')]
        
        # Create target (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Remove rows with NaN values
        clean_df = df.dropna()
        
        if len(clean_df) < 30:
            raise ValueError("Insufficient data for prediction")
        
        X = clean_df[feature_columns]
        y = clean_df['Target']
        
        return X, y, clean_df
    
    def train_advanced_model(self, X, y):
        """Train XGBoost model with hyperparameter tuning"""
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Initialize XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        # Grid search with time series cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        print("Training XGBoost model with hyperparameter tuning...")
        grid_search.fit(X_scaled, y)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.feature_importances_ = self.model.feature_importances_
        
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Plot feature importance
        self.plot_feature_importance(X.columns)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'feature_importances': dict(zip(X.columns, self.feature_importances_))
        }
    
    def plot_feature_importance(self, feature_names, top_n=20):
        """Plot feature importance"""
        if self.feature_importances_ is None:
            return
            
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('feature_importance.png')
        plt.close()
    
    def predict(self, data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        X, _, _ = self.prepare_features(data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction intervals using bootstrapping
        n_bootstraps = 1000
        bootstrap_preds = np.zeros((n_bootstraps, len(X)))
        
        for i in range(n_bootstraps):
            # Sample with replacement
            sample_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X_scaled[sample_idx]
            y_sample = predictions[sample_idx]
            
            # Train a simple model on the bootstrap sample
            model = xgb.XGBRegressor(n_estimators=50, random_state=i)
            model.fit(X_sample, y_sample)
            
            # Store predictions
            bootstrap_preds[i] = model.predict(X_scaled)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_interval': list(zip(lower_bound, upper_bound)),
            'volatility': np.std(bootstrap_preds, axis=0)
        }
    
    def predict_next_day(self, symbol, period='1y'):
        """Predict the next day's price for a given stock
        
        Args:
            symbol (str): Stock symbol (with or without .NS suffix)
            period (str): Data period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            dict: Prediction details including price, confidence interval, and metrics
        """
        import yfinance as yf
        from datetime import datetime, timedelta
        
        try:
            # Clean and validate symbol
            symbol = str(symbol).upper().strip()
            
            # Add .NS suffix for NSE if not present and not already a yfinance symbol
            if not any(symbol.endswith(ext) for ext in ['.NS', '.BO', '.SR', '.BS']):
                symbol += '.NS'  # Default to NSE
            
            # Check if symbol exists with a direct API call
            ticker = yf.Ticker(symbol)
            
            # Try to get info to verify symbol exists
            try:
                info = ticker.info
                if not info or 'regularMarketPrice' not in info:
                    raise ValueError(f"No data found for {symbol}")
                
                # Get company name or use symbol as fallback
                company_name = info.get('shortName', symbol)
                sector = info.get('sector', 'N/A')
                current_price = info.get('regularMarketPrice', 0)
                
            except Exception as e:
                raise ValueError(f"Could not fetch data for {symbol}: {str(e)}")
            
            # Fetch historical data with error handling
            try:
                # Try with auto_adjust first
                data = ticker.history(period=period, auto_adjust=True)
                
                # If no data, try without auto_adjust
                if data.empty:
                    data = ticker.history(period=period, auto_adjust=False)
                
                # If still no data, try with a different period
                if data.empty and period != '2y':
                    data = ticker.history(period='2y', auto_adjust=True)
                
                if data.empty:
                    raise ValueError(f"No historical data available for {symbol}")
                    
            except Exception as e:
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")
            
            # Ensure we have enough data
            min_days_required = 60
            if len(data) < min_days_required:
                raise ValueError(f"Insufficient data points ({len(data)} < {min_days_required})")
            
            # Prepare features for the last day
            try:
                X, y, _ = self.prepare_features(data)
            except Exception as e:
                raise ValueError(f"Error preparing features: {str(e)}")
            
            # If model doesn't exist, train it
            if self.model is None:
                print(f"Training model for {symbol}...")
                try:
                    training_result = self.train_advanced_model(X, y)
                    print(f"Model trained with score: {training_result['best_score']:.4f}")
                except Exception as e:
                    raise ValueError(f"Error training model: {str(e)}")
            
            # Get the most recent data point
            latest_features = X.iloc[[-1]]
            
            # Make prediction
            try:
                prediction = self.model.predict(self.scaler.transform(latest_features))[0]
                
                # Get prediction interval
                preds = self.predict(data)
                lower = preds['lower_bound'][-1]
                upper = preds['upper_bound'][-1]
                
                # Calculate potential return
                potential_return = ((prediction - current_price) / current_price) * 100
                
                # Generate signal
                signal = "NEUTRAL"
                if potential_return > 5:
                    signal = "STRONG BUY"
                elif potential_return > 2:
                    signal = "BUY"
                elif potential_return < -5:
                    signal = "STRONG SELL"
                elif potential_return < -2:
                    signal = "SELL"
                
                return {
                    'symbol': symbol,
                    'name': company_name,
                    'last_price': round(float(current_price), 2),
                    'predicted_price': round(float(prediction), 2),
                    'confidence_interval': [round(float(lower), 2), round(float(upper), 2)],
                    'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'volatility': round(float(np.std([lower, upper]) / prediction * 100), 2),
                    'potential_return': round(potential_return, 2),
                    'signal': signal,
                    'exchange': 'NSE' if symbol.endswith('.NS') else 'BSE' if symbol.endswith('.BO') else 'Unknown',
                    'sector': sector,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'success'
                }
                
            except Exception as e:
                raise ValueError(f"Error making prediction: {str(e)}")
            
        except Exception as e:
            # Return error information
            return {
                'error': f"Error predicting for {symbol}",
                'details': str(e),
                'status': 'error',
                'symbol': symbol,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }