"""
Simplified Rural Credit Scoring Model
- Minimal dependencies
- Fast training
- Basic API for scoring
"""

import pickle
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Generate synthetic data
def generate_rural_data(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    
    # Generate features
    dbt_score = rng.uniform(0.5, 1.0, n_samples)
    bill_score = rng.uniform(0.4, 1.0, n_samples)
    digital_score = rng.uniform(0.3, 1.0, n_samples)
    cash_score = rng.uniform(0.6, 1.0, n_samples)
    geo_score = rng.uniform(0.7, 1.0, n_samples)
    income = rng.normal(5000, 2000, n_samples)
    aeps_tx = rng.poisson(8, n_samples)
    
    # Create risk score
    risk_score = (
        0.3 * dbt_score + 
        0.2 * bill_score - 
        0.1 * cash_score + 
        0.4 * geo_score - 
        0.25 * digital_score
    )
    
    # Create target variable
    p_default = 1 / (1 + np.exp(-risk_score))
    default = rng.binomial(1, p_default, n_samples)
    
    return pd.DataFrame({
        "dbt_score": dbt_score,
        "bill_score": bill_score,
        "digital_score": digital_score,
        "cash_score": cash_score,
        "geo_score": geo_score,
        "income": income,
        "aeps_tx": aeps_tx,
        "default": default
    })

class CreditModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = [
            "dbt_score", "bill_score", "digital_score",
            "cash_score", "geo_score", "income", "aeps_tx"
        ]
    
    def train(self, df):
        X = df[self.features]
        y = df["default"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        test_probs = self.predict_proba(X_test)
        auc = roc_auc_score(y_test, test_probs)
        logger.info(f"Test AUC: {auc:.4f}")
        
    def predict_proba(self, X): 
        return self.model.predict_proba(X)[:, 1]
    
    def credit_score(self, X): 
        p = self.predict_proba(X)
        return (300 + (1 - p) * 550).round().astype(int)

# FastAPI setup
app = FastAPI(title="Simple Credit Scoring API")

class ScoringRequest(BaseModel):
    dbt_score: float
    bill_score: float
    digital_score: float
    cash_score: float
    geo_score: float
    income: float
    aeps_tx: float

class ScoringResponse(BaseModel):
    credit_score: int
    default_prob: float
    risk_band: str

model = CreditModel()

@app.on_event("startup")
async def load_model():
    logger.info("Generating training data")
    df = generate_rural_data(10000)
    logger.info("Training model")
    model.train(df)
    logger.info("Model training complete")

@app.post("/score", response_model=ScoringResponse)
async def score_applicant(request: ScoringRequest):
    # Create input DataFrame
    row = pd.DataFrame([request.dict()])
    
    # Predict
    prob = model.predict_proba(row)[0]
    score = model.credit_score(row)[0]
    
    band = ("A" if score >= 750 else
            "B" if score >= 700 else
            "C" if score >= 650 else
            "D" if score >= 600 else "E")
    
    return {
        "credit_score": score,
        "default_prob": prob,
        "risk_band": band
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)