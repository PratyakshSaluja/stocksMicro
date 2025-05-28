from typing import List, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MutualFundScore:
    scheme_name: str
    total_score: float
    returns_score: float
    nav_score: float
    risk_score: float
    consistency_score: float

def calculate_returns_score(returns: List[float]) -> float:
    """Calculate returns score based on available return data"""
    try:
        if not returns:
            return 0.5
        # Give more weight to more recent returns
        weighted_returns = sum(r * w for r, w in zip(returns, [0.4, 0.3, 0.2, 0.1]))
        # Normalize to [0, 1] range
        return min(max(weighted_returns / 30, 0), 1)  # Cap at 30% returns
    except Exception as e:
        logger.error(f"Error calculating returns score: {e}")
        return 0.5

def calculate_mutual_fund_scores(funds: List[Dict]) -> List[MutualFundScore]:
    """Calculate composite scores for mutual funds based on multiple factors"""
    if not funds:
        return []
    
    # Calculate NAV ranges for normalization
    navs = [float(fund['data'][0]['nav']) for fund in funds if fund['data']]
    min_nav, max_nav = min(navs), max(navs) if navs else (0, 1)
    
    scores = []
    for fund in funds:
        try:
            # Extract recent NAV values and calculate returns
            nav_data = fund['data']
            if not nav_data:
                continue
                
            current_nav = float(nav_data[0]['nav'])
            nav_score = current_nav / max_nav if max_nav > 0 else 0.5
            
            # Calculate returns for different periods
            returns = []
            if len(nav_data) >= 30:  # 1 month return
                returns.append((float(nav_data[0]['nav']) - float(nav_data[29]['nav'])) / float(nav_data[29]['nav']) * 100)
            if len(nav_data) >= 90:  # 3 month return
                returns.append((float(nav_data[0]['nav']) - float(nav_data[89]['nav'])) / float(nav_data[89]['nav']) * 100)
            if len(nav_data) >= 180:  # 6 month return
                returns.append((float(nav_data[0]['nav']) - float(nav_data[179]['nav'])) / float(nav_data[179]['nav']) * 100)
            if len(nav_data) >= 365:  # 1 year return
                returns.append((float(nav_data[0]['nav']) - float(nav_data[364]['nav'])) / float(nav_data[364]['nav']) * 100)
            
            # Calculate component scores
            returns_score = calculate_returns_score(returns)
            
            # Risk score based on NAV volatility
            nav_values = [float(data['nav']) for data in nav_data[:90]]  # Last 90 days
            risk_score = 1 - (max(nav_values) - min(nav_values)) / max(nav_values) if nav_values else 0.5
            
            # Consistency score based on continuous growth
            positive_days = sum(1 for i in range(len(nav_data)-1) if float(nav_data[i]['nav']) >= float(nav_data[i+1]['nav']))
            consistency_score = positive_days / (len(nav_data)-1) if len(nav_data) > 1 else 0.5
            
            # Calculate total score with weights
            total_score = (
                returns_score * 0.40 +      # 40% weight to returns
                nav_score * 0.15 +          # 15% weight to NAV
                risk_score * 0.25 +         # 25% weight to risk
                consistency_score * 0.20     # 20% weight to consistency
            )
            
            scores.append(MutualFundScore(
                scheme_name=fund['meta']['scheme_name'],
                total_score=total_score,
                returns_score=returns_score,
                nav_score=nav_score,
                risk_score=risk_score,
                consistency_score=consistency_score
            ))
            
        except Exception as e:
            logger.error(f"Error calculating score for {fund['meta'].get('scheme_name', 'Unknown')}: {e}")
            continue
    
    return sorted(scores, key=lambda x: x.total_score, reverse=True)

def get_fund_metrics(fund: Dict) -> Dict:
    """Extract important metrics for a mutual fund"""
    try:
        nav_data = fund['data']
        if not nav_data:
            return None
        
        metrics = {
            'scheme_name': fund['meta']['scheme_name'],
            'scheme_category': fund['meta']['scheme_category'],
            'fund_house': fund['meta']['fund_house'],
            'latest_nav': float(nav_data[0]['nav'])
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error processing fund {fund['meta'].get('scheme_name', 'Unknown')}: {e}")
        return None