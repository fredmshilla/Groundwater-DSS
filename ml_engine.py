import os
import json
import psycopg2
import pandas as pd
import select
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor

# Load Environment Variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "groundwater_dss")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")

BASE_DIR = Path(__file__).parent
WEIGHTS_PATH = BASE_DIR / "models" / "feature_weights.json"

def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

def train_model():
    """Fetches data from PostGIS and trains the Random Forest model."""
    print("Fetching live data from PostGIS for training...")
    try:
        conn = get_db_connection()
        query = """
            SELECT yield_m3h, soil_score, slope_score, rainfall_score, 
                   lulc_score, geology_score, aquifer_score 
            FROM boreholes 
            WHERE yield_m3h IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Drop rows with missing features
        df = df.dropna()

        if len(df) < 5:
            print("Not enough data to train ML model (need at least 5 records).")
            return

        features = ['soil_score', 'slope_score', 'rainfall_score', 'lulc_score', 'geology_score', 'aquifer_score']
        X = df[features]
        y = df['yield_m3h']

        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Extract and save Feature Weights
        weights = dict(zip(features, model.feature_importances_))
        
        # Ensure models directory exists
        WEIGHTS_PATH.parent.mkdir(exist_ok=True)
        with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f:
            json.dump({"yield_weights": weights}, f, indent=4)
            
        print("✅ ML Model retrained successfully. New weights saved.")

    except Exception as e:
        print(f"Error training model: {e}")

def listen_to_qfield_updates():
    """Listens for PostGIS notifications (triggered by QField syncs)."""
    print("🎧 ML Engine is listening for QField field updates...")
    conn = get_db_connection()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("LISTEN borehole_update;")

    try:
        while True:
            if select.select([conn], [], [], 5) == ([], [], []):
                pass
            else:
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    print(f"🔔 Update detected from QField: {notify.payload}")
                    train_model() # Trigger Auto-Retrain
    except KeyboardInterrupt:
        print("Listener stopped.")
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run a manual training cycle')
    parser.add_argument('--listen', action='store_true', help='Start the QField database listener')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.listen:
        listen_to_qfield_updates()
    else:
        print("Please specify --train or --listen")