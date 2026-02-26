"""
Market Data Simulator
Generates realistic OHLCV data with regime changes, volatility clustering, and microstructural noise.
Uses geometric Brownian motion with jumps and regime switching for realistic simulation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime configuration for simulation"""
    name: str
    volatility: float
    drift: float
    jump_prob: float
    jump_size: float
    volume_multiplier: float

class MarketDataSimulator:
    """Generates realistic market data with multiple regimes"""
    
    # Market regimes configuration
    REGIMES = {
        'bull': MarketRegime('bull', 0.15, 0.0002, 0.05, 0.02, 1.2),
        'bear': MarketRegime('bear', 0.25, -0.0003, 0.08, -0.03, 0.8),
        'range': MarketRegime('range', 0.10, 0.0000, 0.03, 0.01, 1.0),
        'high_vol': MarketRegime('high_vol', 0.40, 0.0000, 0.12, 0.05, 1.5)
    }
    
    def __init__(self, initial_price: float = 100.0, seed: Optional[int] = None):
        """
        Initialize simulator with initial price
        
        Args:
            initial_price: Starting price for simulation
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.regime_history = []
        
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to: {seed}")
            
        logger.info(f"MarketDataSimulator initialized with initial price: ${initial_price:.2f}")
    
    def _generate_regime_sequence(self, n_periods: int) -> List[str]:
        """Generate realistic regime sequence with persistence"""
        regimes = list(self.REGIMES.keys())
        current_regime = np.random.choice(regimes)
        sequence = [current_regime]
        
        for _ in range(1, n_periods):
            # Regime tends to persist (80% stay, 20% change)
            if np.random.random() < 0.8:
                sequence.append(current_regime)
            else:
                # Weighted probability for regime transition
                if current_regime == 'bull':
                    probs = [0.1, 0.3, 0.4, 0.2]  # More likely to go to range
                elif current_regime == 'bear':
                    probs = [0.3, 0.1, 0.4, 0.2]  # More likely to go to range
                elif current_regime == 'range':
                    probs = [0.3, 0.3, 0.2, 0.2]  # Equal to bull/bear
                else:  # high_vol
                    probs = [0.25, 0.25, 0.25, 0.25]
                    
                current_regime = np.random.choice(regimes, p=probs)
                sequence.append(current_regime)
        
        self.regime_history = sequence
        return sequence
    
    def generate_ohlcv(
        self, 
        n_periods: int = 1000,
        interval_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Generate OHLCV DataFrame with realistic market dynamics
        
        Args:
            n_periods: Number of candles to generate
            interval_minutes: Time interval in minutes
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, regime
        """
        logger.info(f"Generating {n_periods} periods of {interval_minutes}-minute OHLCV data")
        
        # Generate regime sequence
        regimes = self._generate_regime_sequence(n_periods)
        
        # Initialize arrays
        opens = np.zeros(n_periods)
        highs = np.zeros(n_periods)
        lows = np.zeros(n_periods)
        closes = np.zeros(n_periods)
        volumes = np.zeros(n_periods)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(minutes=n_periods * interval_minutes)
        timestamps = [start_time + timedelta(minutes=i * interval_minutes) 
                     for i in range(n_periods)]
        
        # Generate price series
        price = self.initial_price
        
        for i in range(n_periods):
            regime = self.REGIMES[regimes[i]]
            
            # Base return with drift and volatility
            dt = interval_minutes / (24 * 60 * 252)  # Annualized
            mu = regime.drift
            sigma = regime.volatility
            
            # Geometric Brownian Motion component
            normal_return = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            
            # Jump component
            if np.random.random() < regime.jump_prob:
                jump = np.random.normal(regime.jump_size, abs(regime.jump_size) * 0.5)
                normal_return += jump
            
            # Apply return
            price *= np.exp(normal_return)
            
            # Generate OHLC from price with microstructural noise
            open_price = price
            
            # Intra-period volatility
            intra_vol = sigma * np.sqrt(dt) * 2  # Higher volatility within period
            
            # Generate high and low
            period_range = np.abs(np.random.normal(0, intra_vol, 4))
            high_price = open_price * (1 + np.max(period_range))
            low_price = open_price * (1 - np.min(period_range))
            
            # Close price (can differ from open)
            close_price = open_price * (1 + np.random.normal(0, intra_vol * 0.5))
            
            # Ensure high >= low and high/low are reasonable relative to open/close
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Volume with regime multiplier and correlation with volatility
            base_volume = 1000 + np.random.exponential(500)
            volume = base_volume * regime.volume_multiplier * (1 + abs(normal_return) * 10)
            volume = max(100, volume)  # Minimum volume
            
            # Store values
            opens[i] = open_price
            highs[i] = high_price
            lows[i] = low_price
            closes[i] = close_price
            volumes[i] = volume
            
            # Update price for next period's open
            price = close_price
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low':