import os
import pandas as pd
import psycopg2
import traceback
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "AI_Final_Clean_2.csv"

def migrate_data():
    print("Starting Data Migration to PostGIS...")
    try:
        if not CSV_PATH.exists():
            print(f"❌ Error: Cannot find CSV file at {CSV_PATH}")
            return

        # 1. Load CSV and safely convert all NaN/missing values to standard None
        df = pd.read_csv(CSV_PATH)
        df = df.where(pd.notnull(df), None)
        print(f"Loaded {len(df)} boreholes from CSV.")

        # 2. Connect to PostGIS Database
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "groundwater_dss"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASS", "postgres"),
            host=os.getenv("DB_HOST", "localhost")
        )
        cursor = conn.cursor()

        # 3. Convert DataFrame to a list of pure dictionaries (Bulletproof extraction)
        records = df.to_dict('records')
        success_count = 0

        for index, row in enumerate(records):
            # Extract data using safe dictionary checks
            b_id = str(row.get('Borehole_ID') or f'BH_{index}')
            sub_county = str(row.get('Sub_County') or 'Unknown')
            
            yield_val = row.get('Yield (m3/hr)')
            swl_val = row.get('Static Water Level (m)')
            depth_val = row.get('Total_Depth')
            
            soil = row.get('SOIL_1')
            slope = row.get('SLOPE_1')
            rain = row.get('RAINFALL_1')
            lulc = row.get('LULC_1')
            geo = row.get('GEOLOGY_1')
            aq = row.get('AQUIFER_1')
            
            # Handle X/Y vs Longitude/Latitude column names
            lon = row.get('X') if row.get('X') is not None else row.get('Longitude')
            lat = row.get('Y') if row.get('Y') is not None else row.get('Latitude')

            # Skip if there are no coordinates
            if lon is None or lat is None:
                continue

            # 4. The SQL Insert Command
            insert_query = """
                INSERT INTO boreholes (
                    borehole_id, sub_county, yield_m3h, swl_m, total_depth_m,
                    soil_score, slope_score, rainfall_score, lulc_score, geology_score, aquifer_score,
                    geom
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                ) ON CONFLICT (borehole_id) DO NOTHING;
            """
            
            cursor.execute(insert_query, (
                b_id, sub_county, yield_val, swl_val, depth_val,
                soil, slope, rain, lulc, geo, aq,
                float(lon), float(lat)
            ))
            success_count += 1

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Migration Complete! {success_count} boreholes securely moved to PostGIS.")

    except Exception as e:
        print("❌ Error during migration. See details below:")
        # This will print the exact line number if it fails again
        traceback.print_exc() 

if __name__ == "__main__":
    migrate_data()