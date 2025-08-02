"""
Advanced AI Models for Stock Prediction
- Facebook Prophet for time series forecasting
- LSTM Neural Networks for deep learning predictions
- ARIMA for classical time series analysis
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try importing advanced models with fallbacks
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("ARIMA not available")

# Skip TensorFlow/LSTM for now due to numpy compatibility issues
LSTM_AVAILABLE = False
print("LSTM temporarily disabled due to numpy compatibility")

class AdvancedStockPredictor:
    def __init__(self):
        self.prophet_model = None
        self.lstm_model = None
        self.arima_model = None
        self.scaler = None
        
    def prepare_prophet_data(self, data):
        """Prepare data for Facebook Prophet"""
        df = data.copy()
        df = df.reset_index()
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Convert to datetime and remove timezone to fix Prophet error
        df['ds'] = pd.to_datetime(df['ds'])
        if hasattr(df['ds'].dtype, 'tz') and df['ds'].dtype.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)
        
        return df[['ds', 'y']]
    
    def train_prophet_model(self, data):
        """Train Facebook Prophet model"""
        try:
            if not PROPHET_AVAILABLE:
                return False
                
            prophet_data = self.prepare_prophet_data(data)
            
            # Configure Prophet with Indian market characteristics
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,  # Flexibility in trend changes
                seasonality_prior_scale=10.0,  # Seasonality strength
                holidays_prior_scale=10.0,     # Holiday effects
                daily_seasonality=True,        # Daily patterns
                weekly_seasonality=True,       # Weekly patterns
                yearly_seasonality=True,       # Yearly patterns
                interval_width=0.80           # Confidence intervals
            )
            
            # Add custom seasonalities for Indian market
            self.prophet_model.add_seasonality(
                name='monthly', period=30.5, fourier_order=5
            )
            
            self.prophet_model.fit(prophet_data)
            return True
            
        except Exception as e:
            print(f"Prophet training error: {e}")
            return False
    
    def predict_with_prophet(self, periods=1):
        """Make predictions using Prophet"""
        try:
            if not self.prophet_model:
                return None
                
            future = self.prophet_model.make_future_dataframe(periods=periods)
            forecast = self.prophet_model.predict(future)
            
            return {
                'predicted_price': float(forecast['yhat'].iloc[-1]),
                'lower_bound': float(forecast['yhat_lower'].iloc[-1]),
                'upper_bound': float(forecast['yhat_upper'].iloc[-1]),
                'trend': float(forecast['trend'].iloc[-1])
            }
            
        except Exception as e:
            print(f"Prophet prediction error: {e}")
            return None
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        from sklearn.preprocessing import MinMaxScaler
        
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i-lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, data):
        """Train LSTM model - temporarily disabled"""
        return False
    
    def predict_with_lstm(self, data):
        """Make predictions using LSTM - temporarily disabled"""
        return None
    
    def train_arima_model(self, data, order=(5,1,0)):
        """Train ARIMA model"""
        try:
            if not ARIMA_AVAILABLE:
                return False
                
            prices = data['Close'].values
            
            # Fit ARIMA model
            self.arima_model = ARIMA(prices, order=order)
            self.arima_fitted = self.arima_model.fit()
            
            return True
            
        except Exception as e:
            print(f"ARIMA training error: {e}")
            return False
    
    def predict_with_arima(self, steps=1):
        """Make predictions using ARIMA"""
        try:
            if not self.arima_fitted:
                return None
                
            forecast = self.arima_fitted.forecast(steps=steps)
            return float(forecast[0]) if steps == 1 else forecast.tolist()
            
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            return None
    
    def ensemble_prediction(self, data, model_weights=None):
        """
        Combine predictions from multiple models using weighted averaging.
        
        Args:
            data: DataFrame containing stock price data
            model_weights: Optional dictionary specifying weights for each model.
                         If None, uses default weights: {'prophet': 0.4, 'arima': 0.3, 'lstm': 0.3}
                         
        Returns:
            dict: Contains ensemble prediction, individual model predictions, weights,
                 confidence score, and any error messages.
        """
        if model_weights is None:
            model_weights = {
                'prophet': 0.4,  # 40% weight
                'arima': 0.3,    # 30% weight
                'lstm': 0.3      # 30% weight (if available)
            }
            
        predictions = {}
        errors = {}
        
        try:
            # Prophet prediction
            try:
                if self.train_prophet_model(data):
                    prophet_pred = self.predict_with_prophet()
                    if prophet_pred and 'predicted_price' in prophet_pred:
                        predictions['prophet'] = float(prophet_pred['predicted_price'])
            except Exception as e:
                errors['prophet'] = str(e)
            
            # ARIMA prediction
            try:
                if self.train_arima_model(data):
                    arima_pred = self.predict_with_arima()
                    if arima_pred is not None:
                        predictions['arima'] = float(arima_pred) if not isinstance(arima_pred, (list, np.ndarray)) else float(arima_pred[0])
            except Exception as e:
                errors['arima'] = str(e)
            
            # LSTM prediction (disabled by default, can be enabled by uncommenting and implementing)
            # try:
            #     if LSTM_AVAILABLE and self.train_lstm_model(data):
            #         lstm_pred = self.predict_with_lstm(data)
            #         if lstm_pred is not None:
            #             predictions['lstm'] = float(lstm_pred) if not isinstance(lstm_pred, (list, np.ndarray)) else float(lstm_pred[0])
            # except Exception as e:
            #     errors['lstm'] = str(e)
            
            # Calculate weighted ensemble
            if not predictions:
                return {
                    'error': 'No models produced valid predictions',
                    'errors': errors
                }
            
            # Filter weights to only include models that produced predictions
            valid_weights = {k: v for k, v in model_weights.items() if k in predictions}
            total_weight = sum(valid_weights.values())
            
            # If no valid weights, use equal weights
            if total_weight <= 0:
                valid_weights = {model: 1.0/len(predictions) for model in predictions}
                total_weight = 1.0
            
            # Calculate weighted average
            ensemble_price = sum(
                predictions[model] * (valid_weights[model] / total_weight)
                for model in predictions if model in valid_weights
            )
            
            # Calculate confidence based on model agreement and prediction variance
            if len(predictions) > 1:
                pred_values = np.array(list(predictions.values()))
                pred_std = np.std(pred_values) / np.mean(np.abs(pred_values)) if np.mean(np.abs(pred_values)) > 0 else 0
                confidence = max(0, 1 - pred_std) * (len(predictions) / len(model_weights))
            else:
                confidence = 0.5  # Lower confidence for single model
            
            return {
                'ensemble_prediction': float(ensemble_price),
                'individual_predictions': predictions,
                'model_weights': valid_weights,
                'confidence': min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                'errors': errors if errors else None
            }
            
        except Exception as e:
            return {
                'error': f'Error in ensemble prediction: {str(e)}',
                'errors': errors
            }
    
    def get_model_performance(self, data):
        """Get performance metrics for all models"""
        performance = {}
        
        if len(data) < 100:
            return {'error': 'Insufficient data for performance evaluation'}
        
        # Use last 20% of data for testing
        split_idx = int(0.8 * len(data))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        actual_prices = test_data['Close'].values
        
        # Test each model
        for model_name in ['prophet', 'lstm', 'arima']:
            try:
                if model_name == 'prophet' and self.train_prophet_model(train_data):
                    predictions = []
                    for i in range(len(test_data)):
                        pred = self.predict_with_prophet(1)
                        if pred:
                            predictions.append(pred['predicted_price'])
                
                elif model_name == 'lstm' and self.train_lstm_model(train_data):
                    pred = self.predict_with_lstm(train_data)
                    if pred:
                        predictions = [pred] * len(test_data)  # Single prediction
                
                elif model_name == 'arima' and self.train_arima_model(train_data):
                    predictions = []
                    for i in range(len(test_data)):
                        pred = self.predict_with_arima(1)
                        if pred:
                            predictions.append(pred)
                
                if 'predictions' in locals() and predictions:
                    # Calculate metrics
                    mse = np.mean((actual_prices[:len(predictions)] - predictions) ** 2)
                    mae = np.mean(np.abs(actual_prices[:len(predictions)] - predictions))
                    mape = np.mean(np.abs((actual_prices[:len(predictions)] - predictions) / actual_prices[:len(predictions)])) * 100
                    
                    performance[model_name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'mape': float(mape),
                        'accuracy': max(0, 100 - mape)
                    }
                
            except Exception as e:
                performance[model_name] = {'error': str(e)}
        
        return performance