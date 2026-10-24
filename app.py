import os
import warnings

# --- 🚨 SILENCE TERMINAL WARNINGS 🚨 ---
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.pop('PROJ_LIB', None)
os.environ.pop('PROJ_DATA', None)

import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from groq import Groq
from geopy.geocoders import ArcGIS 
from streamlit_geolocation import streamlit_geolocation
from branca.element import Template, MacroElement

import matplotlib as mpl
import matplotlib.pyplot as plt
import io
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="Groundwater DSS", layout="wide", initial_sidebar_state="collapsed")

# --- ROBUST STATE MANAGEMENT (Memory) ---
if "active_lat" not in st.session_state:
    st.session_state.active_lat = None
    st.session_state.active_lon = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [0.5143, 35.2697]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 10
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None
if "last_processed_gps" not in st.session_state:
    st.session_state.last_processed_gps = None
if "basemap" not in st.session_state:
    st.session_state.basemap = "Google Maps"
if "show_yield" not in st.session_state:
    st.session_state.show_yield = False
if "show_swl" not in st.session_state:
    st.session_state.show_swl = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am HydroBot. I can analyze specific map locations **OR** perform data analysis on the entire Borehole Registry. Ask me for pump sizing, yield forecasts, or data cleaning insights!"}
    ]

# --- NATIVE UI CSS (WITH POINTER FIX) ---
st.markdown("""
    <style>
    div[data-baseweb="select"] > div { cursor: pointer !important; }
    div[data-baseweb="select"] * { cursor: pointer !important; }
    [data-testid="stMetric"] { background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #00A8E8; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    [data-testid="stMetricValue"] { color: #00A8E8 !important; font-weight: bold; }
    iframe { border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #E1E8ED; }
    </style>
""", unsafe_allow_html=True)

# --- PATHS & ENV ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
RASTER_PATH = os.path.join(BASE_DIR, "data", "Uasin_Gishu_AHP.tif")
CSV_PATH = os.path.join(BASE_DIR, "data", "AI_Final_Clean_2.csv")

def load_env_manually(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env_manually(ENV_PATH)
W1, W2 = 0.6, 0.4

# --- DATA LOGIC WITH AGGRESSIVE ID HUNTER ---
@st.cache_data(ttl=60)
def get_live_data():
    try:
        if not os.path.exists(CSV_PATH):
            st.error(f"Dataset missing! Please ensure AI_Final_Clean_2.csv is inside the 'data' folder on GitHub.")
            return pd.DataFrame()
            
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.lower().str.strip()
        
        rename_dict = {}
        for col in df.columns:
            if ('yield' in col) and ('yield_m3h' not in rename_dict.values()):
                rename_dict[col] = 'yield_m3h'
            elif ('swl' in col or 'static' in col) and ('swl_m' not in rename_dict.values()):
                rename_dict[col] = 'swl_m'
            elif ('depth' in col or 'dept' in col) and ('total_depth_m' not in rename_dict.values()):
                rename_dict[col] = 'total_depth_m'
            elif col in ['y', 'lat', 'latitude'] and ('lat' not in rename_dict.values()):
                rename_dict[col] = 'lat'
            elif col in ['x', 'lon', 'longitude', 'lng'] and ('lon' not in rename_dict.values()):
                rename_dict[col] = 'lon'
            elif ('borehole' in col or col == 'id' or 'name' in col) and ('borehole_id' not in rename_dict.values()):
                rename_dict[col] = 'borehole_id'
                
        df = df.rename(columns=rename_dict)
        
        for numeric_col in ['lat', 'lon', 'yield_m3h', 'swl_m', 'total_depth_m']:
            if numeric_col in df.columns:
                df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')
                
        if 'borehole_id' in df.columns:
            df['borehole_id'] = df['borehole_id'].astype(str)
                
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

# --- SPATIAL LOGIC (WITH BULLETPROOF DOUBLE-GEOFENCE) ---
def get_ahp_suitability(lat, lon):
    if not os.path.exists(RASTER_PATH): return 0.0
    try:
        with rasterio.open(RASTER_PATH) as src:
            point_lon, point_lat = transform(CRS.from_epsg(4326), src.crs, [lon], [lat])
            vals = src.sample([(point_lon[0], point_lat[0])])
            for val in vals:
                return float(val[0]) if val[0] > 0 else 0.0
    except: return 0.0

@st.cache_data(ttl=60)
def get_visible_boreholes(df):
    if df.empty or not os.path.exists(RASTER_PATH): return df
    try:
        with rasterio.open(RASTER_PATH) as src:
            check_df = df.dropna(subset=['lat', 'lon']).copy()
            if check_df.empty: return check_df

            wgs_bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
            min_lon, min_lat, max_lon, max_lat = wgs_bounds

            inside_box = check_df[
                (check_df['lon'] >= min_lon) & (check_df['lon'] <= max_lon) &
                (check_df['lat'] >= min_lat) & (check_df['lat'] <= max_lat)
            ].copy()

            if inside_box.empty: return inside_box

            lons = inside_box['lon'].tolist()
            lats = inside_box['lat'].tolist()
            
            point_lons, point_lats = transform(CRS.from_epsg(4326), src.crs, lons, lats)
            coords = list(zip(point_lons, point_lats))
            
            is_inside = []
            for val in src.sample(coords):
                if np.isnan(val[0]):
                    is_inside.append(False)
                else:
                    is_inside.append(val[0] > 0)
                
            return inside_box[pd.Series(is_inside, index=inside_box.index)]
    except Exception as e:
        return df.iloc[0:0] 

def run_idw_prediction(lat, lon, df):
    if df.empty: return 0.0, 0.0
    df['dist'] = df.apply(lambda r: distance.euclidean((lat, lon), (r['lat'], r['lon'])), axis=1)
    neighbors = df.nsmallest(5, 'dist')
    weights = 1.0 / (neighbors['dist']**2 + 1e-12)
    
    def interpolate(col):
        if col not in neighbors.columns: return 0.0
        valid_mask = pd.notna(neighbors[col])
        if len(neighbors.loc[valid_mask, col]) == 0: return 0.0
        return round(np.dot(weights[valid_mask], neighbors.loc[valid_mask, col]) / np.sum(weights[valid_mask]), 2)

    return interpolate('yield_m3h'), interpolate('swl_m')

# --- FAST-IDW ENGINE WITH STATISTICAL OUTLIER CLIPPING ---
@st.cache_data(ttl=300)
def generate_continuous_overlay(df_live, target_col, colormap_name):
    if df_live.empty or not os.path.exists(RASTER_PATH) or target_col not in df_live.columns:
        return None, None, None, None

    with rasterio.open(RASTER_PATH) as src:
        WIDTH, HEIGHT = 150, 150  
        ahp_mask = src.read(1, out_shape=(HEIGHT, WIDTH))
        
        cols, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        orig_cols = cols * (src.width / WIDTH)
        orig_rows = rows * (src.height / HEIGHT)
        
        grid_x = src.transform.c + src.transform.a * orig_cols + src.transform.b * orig_rows
        grid_y = src.transform.f + src.transform.d * orig_cols + src.transform.e * orig_rows
        grid_pts = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        valid_df = df_live.dropna(subset=['lat', 'lon', target_col])
        if valid_df.empty:
            return None, None, None, None
            
        known_lons, known_lats = valid_df['lon'].values, valid_df['lat'].values
        
        try:
            pts_crs_x, pts_crs_y = transform(CRS.from_epsg(4326), src.crs, known_lons.tolist(), known_lats.tolist())
            known_pts = np.c_[pts_crs_x, pts_crs_y]
        except Exception:
            known_pts = np.c_[known_lons, known_lats]

        known_vals = valid_df[target_col].values

        dists = cdist(grid_pts, known_pts)
        dists[dists == 0] = 1e-12
        weights = 1.0 / (dists ** 2)
        
        idw_flat = np.sum(weights * known_vals, axis=1) / np.sum(weights, axis=1)
        idw_grid = idw_flat.reshape(HEIGHT, WIDTH)

        idw_grid[ahp_mask <= 0] = np.nan

        raw_min, raw_max = np.nanmin(idw_grid), np.nanmax(idw_grid)
        stat_min = np.nanpercentile(idw_grid, 5)
        stat_max = np.nanpercentile(idw_grid, 95)
        
        if stat_min == stat_max:
            stat_min, stat_max = raw_min, raw_max

        cmap = mpl.colormaps[colormap_name].copy()
        cmap.set_bad(alpha=0.0) 
        
        norm = mpl.colors.Normalize(vmin=stat_min, vmax=stat_max)
        rgba_img = cmap(norm(idw_grid))

        buf = io.BytesIO()
        plt.imsave(buf, rgba_img, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_url = f"data:image/png;base64,{img_data}"
        
        try:
            wgs_bounds = transform_bounds(src.crs, CRS.from_epsg(4326), src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
