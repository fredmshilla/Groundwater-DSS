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

            # 1. Protect Rasterio from crashing by pre-filtering with a strict Bounding Box
            wgs_bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
            min_lon, min_lat, max_lon, max_lat = wgs_bounds

            inside_box = check_df[
                (check_df['lon'] >= min_lon) & (check_df['lon'] <= max_lon) &
                (check_df['lat'] >= min_lat) & (check_df['lat'] <= max_lat)
            ].copy()

            if inside_box.empty: return inside_box

            # 2. Strict Boundary Filter using the specific AHP county shape
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
        # If it completely fails, return an empty dataframe rather than dumping all boreholes on the map
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
            wgs_bounds = transform_bounds(src.crs, CRS.from_epsg(4326), src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            img_bounds = [[wgs_bounds[1], wgs_bounds[0]], [wgs_bounds[3], wgs_bounds[2]]]
        except Exception:
            img_bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

        return img_url, img_bounds, float(stat_min), float(stat_max)


# --- MAIN UI DASHBOARD ---
def main():
    st.title(" Groundwater Decision Support System — Uasin Gishu County")
    
    # --- DRAG AND DROP SIDEBAR COMPONENT ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Field Data")
    st.sidebar.write("Drag and drop a CSV file containing new boreholes to temporarily add them to the map.")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    new_boreholes_df = None
    
    if uploaded_file is not None:
        try:
            new_boreholes_df = pd.read_csv(uploaded_file)
            new_boreholes_df.columns = new_boreholes_df.columns.str.lower().str.strip()
            
            # Check for coordinates
            has_lat = any(col in new_boreholes_df.columns for col in ['lat', 'latitude', 'y'])
            has_lon = any(col in new_boreholes_df.columns for col in ['lon', 'longitude', 'lng', 'x'])
            
            if has_lat and has_lon:
                rename_dict = {}
                for col in new_boreholes_df.columns:
                    if col in ['lat', 'latitude', 'y']: rename_dict[col] = 'lat'
                    elif col in ['lon', 'longitude', 'lng', 'x']: rename_dict[col] = 'lon'
                    elif 'yield' in col: rename_dict[col] = 'yield_m3h'
                    elif 'swl' in col or 'static' in col: rename_dict[col] = 'swl_m'
                    elif 'depth' in col or 'dept' in col: rename_dict[col] = 'total_depth_m'
                
                new_boreholes_df = new_boreholes_df.rename(columns=rename_dict)
                st.sidebar.success(f"✅ Successfully loaded {len(new_boreholes_df)} new boreholes!")
            else:
                st.sidebar.error("❌ The CSV must contain Latitude and Longitude columns.")
                new_boreholes_df = None
                
        except Exception as e:
            st.sidebar.error("❌ Error reading the file. Make sure it is a valid CSV.")

    df_full = get_live_data()
    df_visible = get_visible_boreholes(df_full)
    
    tab1, tab2 = st.tabs(["DSS Map & Virtual Borehole", "Live Borehole Registry"])

    with tab1:
        col_map, col_analysis = st.columns([3.5, 1.5])
        
        with col_map:
            c1, c2, c3 = st.columns([1.5, 1, 1])
            with c1:
                st.selectbox("Basemap", ["Google Maps", "Google Satellite", "OpenStreetMap"], key="basemap", label_visibility="collapsed")
            with c2:
                st.toggle("Show Yield Surface", key="show_yield")
            with c3:
                st.toggle("Show SWL Surface", key="show_swl")

            if st.session_state.basemap == "Google Satellite":
                active_tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
                active_attr = 'Google'
            elif st.session_state.basemap == "OpenStreetMap":
                active_tiles = 'OpenStreetMap'
                active_attr = 'OpenStreetMap'
            else:
                active_tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
                active_attr = 'Google'

            m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles=active_tiles, attr=active_attr)

            # 1. Plot existing database points
            if not df_visible.empty:
                for _, row in df_visible.dropna(subset=['lat', 'lon']).iterrows():
                    b_id = row.get('borehole_id', 'Unknown')
                    y_val = row.get('yield_m3h', 'N/A')
                    s_val = row.get('swl_m', 'N/A')
                    d_val = row.get('total_depth_m', 'N/A')
                    
                    popup_html = f"""
                    <div style='min-width: 150px; font-family: sans-serif; color: #1A202C;'>
                        <b>ID:</b> {b_id}<br>
                        <b>Yield:</b> {y_val} m³/hr<br>
                        <b>SWL:</b> {s_val} m<br>
                        <b>Depth:</b> {d_val} m
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], 
                        radius=4, 
                        color='#003B73', 
                        fill_color='#00A8E8', 
                        fill=True, 
                        fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=250)
                    ).add_to(m)
            
            # 2. Plot NEW uploaded points as red markers
            if new_boreholes_df is not None:
                for _, row in new_boreholes_df.dropna(subset=['lat', 'lon']).iterrows():
                    y_val = row.get('yield_m3h', 'N/A')
                    s_val = row.get('swl_m', 'N/A')
                    d_val = row.get('total_depth_m', 'N/A')
                    
                    popup_html = f"""
                    <div style='min-width: 150px; font-family: sans-serif; color: #1A202C;'>
                        <b style='color: #d9534f;'>NEW FIELD DATA</b><br>
                        <b>Yield:</b> {y_val} m³/hr<br>
                        <b>SWL:</b> {s_val} m<br>
                        <b>Depth:</b> {d_val} m
                    </div>
                    """
                    
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        popup=folium.Popup(popup_html, max_width=250),
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)
                
            if st.session_state.show_yield:
                if 'yield_m3h' not in df_full.columns:
                    st.error(f"⚠️ App cannot find Yield data. Available columns are: {list(df_full.columns)}")
                else:
                    y_img, y_bounds, y_min, y_max = generate_continuous_overlay(df_full, 'yield_m3h', 'Spectral')
                    if y_img:
                        folium.raster_layers.ImageOverlay(image=y_img, bounds=y_bounds, opacity=0.75).add_to(m)
                        
                        macro_y = MacroElement()
                        macro_y._template = Template(f"""
                        {{% macro html(this, kwargs) %}}
                        <div style="position: absolute; z-index: 9999; bottom: 30px; right: 30px; background-color: rgba(255, 255, 255, 0.95); padding: 12px; border-radius: 8px; border: 1px solid #ccc; color: #1A202C; font-size: 11px; font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                            <div style="font-weight: bold; margin-bottom: 6px; font-size: 12px;">Yield<br><span style="font-weight: normal; font-size: 10px;">(m³/hr)</span></div>
                            <div style="display: flex; align-items: stretch;">
                                <div style="background: linear-gradient(to top, #d53e4f, #fdae61, #e6f598, #66c2a5, #3288bd); width: 14px; height: 120px; border: 1px solid #999; margin-right: 8px; border-radius: 2px;"></div>
                                <div style="display: flex; flex-direction: column; justify-content: space-between; height: 120px; padding: 1px 0;">
                                    <div style="margin: 0; padding: 0; line-height: 1;">{y_max:.1f}+ <span style="color:#555;">(High)</span></div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(y_min+0.75*(y_max-y_min)):.1f}</div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(y_max+y_min)/2:.1f} <span style="color:#555;">(Med)</span></div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(y_min+0.25*(y_max-y_min)):.1f}</div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{y_min:.1f} <span style="color:#555;">(Low)</span></div>
                                </div>
                            </div>
                        </div>
                        {{% endmacro %}}
                        """)
                        m.get_root().add_child(macro_y)

            if st.session_state.show_swl:
                if 'swl_m' not in df_full.columns:
                    st.error(f"⚠️ App cannot find SWL data. Available columns are: {list(df_full.columns)}")
                else:
                    s_img, s_bounds, s_min, s_max = generate_continuous_overlay(df_full, 'swl_m', 'jet')
                    if s_img:
                        folium.raster_layers.ImageOverlay(image=s_img, bounds=s_bounds, opacity=0.75).add_to(m)
                        
                        macro_s = MacroElement()
                        macro_s._template = Template(f"""
                        {{% macro html(this, kwargs) %}}
                        <div style="position: absolute; z-index: 9999; bottom: 30px; right: 160px; background-color: rgba(255, 255, 255, 0.95); padding: 12px; border-radius: 8px; border: 1px solid #ccc; color: #1A202C; font-size: 11px; font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                            <div style="font-weight: bold; margin-bottom: 6px; font-size: 12px;">SWL<br><span style="font-weight: normal; font-size: 10px;">(meters)</span></div>
                            <div style="display: flex; align-items: stretch;">
                                <div style="background: linear-gradient(to top, #000080, #00ffff, #ffff00, #ff0000); width: 14px; height: 120px; border: 1px solid #999; margin-right: 8px; border-radius: 2px;"></div>
                                <div style="display: flex; flex-direction: column; justify-content: space-between; height: 120px; padding: 1px 0;">
                                    <div style="margin: 0; padding: 0; line-height: 1;">{s_max:.1f}+ <span style="color:#555;">(Deep)</span></div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(s_min+0.75*(s_max-s_min)):.1f}</div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(s_max+s_min)/2:.1f}</div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{(s_min+0.25*(s_max-s_min)):.1f}</div>
                                    <div style="margin: 0; padding: 0; line-height: 1;">{s_min:.1f} <span style="color:#555;">(Shallow)</span></div>
                                </div>
                            </div>
                        </div>
                        {{% endmacro %}}
                        """)
                        m.get_root().add_child(macro_s)

            if st.session_state.active_lat:
                folium.Marker(
                    [st.session_state.active_lat, st.session_state.active_lon],
                    icon=folium.Icon(color='red', icon='info-sign'),
                    popup="Active Target"
                ).add_to(m)

            map_response = st_folium(m, height=650, use_container_width=True, returned_objects=["last_clicked"])

            if map_response:
                current_click = map_response.get('last_clicked')
                if current_click and current_click != st.session_state.last_processed_click:
                    st.session_state.last_processed_click = current_click
                    st.session_state.active_lat = current_click['lat']
                    st.session_state.active_lon = current_click['lng']
                    st.rerun()

        with col_analysis:
            st.subheader("Virtual Site Analysis")
            st.write("Real-time interpolation of groundwater parameters.")
            st.divider()

            if st.session_state.active_lat:
                lat = st.session_state.active_lat
                lon = st.session_state.active_lon
                
                ahp_val = round(get_ahp_suitability(lat, lon), 2)
                
                if ahp_val == 0:
                    st.error(" Target is outside the Uasin Gishu boundary. No data available.")
                else:
                    e_yield, e_swl = run_idw_prediction(lat, lon, df_full)
                    norm_yield = min(e_yield, 5.0) 
                    gwp_score = round((ahp_val * W1) + (norm_yield * W2), 2)
                    
                    st.caption(f" Coordinates: {round(lat,5)}, {round(lon,5)}")
                    
                    st.metric("Expected SWL", f"{e_swl} m")
                    st.metric("Expected Yield", f"{e_yield} m³/hr")
                    st.metric("GIS Suitability (AHP)", f"{ahp_val} / 5")
                    
                    st.divider()
                    st.subheader(f"Overall GWP Index: {gwp_score} / 5")
                    
                    if gwp_score > 3.5: st.success(" Highly Favorable for Drilling")
                    elif gwp_score > 2.0: st.warning(" Moderate Risk Level")
                    else: st.error(" High Dry-Hole Risk")
            else:
                st.info("Awaiting location input. Click the map or use GPS to generate data.")

        # --- BOTTOM SECTION: HYDROBOT UI ---
        st.divider()
        st.subheader("💬 HydroBot: AI Data & Site Analyst")
        
        current_api_key = os.environ.get("GROQ_API_KEY", "")
        if not current_api_key:
            st.warning("⚠️ Please paste your free Groq API Key to enable the AI Consultant.")
            user_key = st.text_input("🔑 Groq API Key:", type="password")
            if user_key:
                os.environ["GROQ_API_KEY"] = user_key
                st.rerun()
                
        if current_api_key:
            client = Groq(api_key=current_api_key)
            col_gps, col_chat = st.columns([1, 4])
            
            with col_gps:
                st.write("**Access My Location:**")
                loc = streamlit_geolocation()
                
                if loc and loc.get('latitude'):
                    gps_coords = {'lat': loc['latitude'], 'lng': loc['longitude']}
                    
                    if gps_coords != st.session_state.last_processed_gps:
                        st.session_state.last_processed_gps = gps_coords
                        st.session_state.active_lat = gps_coords['lat']
                        st.session_state.active_lon = gps_coords['lng']
                        st.session_state.map_center = [gps_coords['lat'], gps_coords['lng']]
                        st.session_state.map_zoom = 16 
                        
                        st.session_state.last_processed_click = None 
                        
                        st.session_state.messages.append({"role": "user", "content": "📍 *User generated data via GPS*"})
                        st.session_state.messages.append({"role": "assistant", "content": "I have successfully calculated the parameters for your GPS location. Review the Virtual Site Analysis panel. How can I help you analyze these results?"})
                        st.rerun()

            with col_chat:
                chat_container = st.container(height=400)
                
                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                
                prompt = st.chat_input("Ask about the map site, or ask 'Are there missing values in our registry?'")
                
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    with st.spinner("Analyzing data and generating insights..."):
                        try:
                            location_name = "an unspecified location"
                            if st.session_state.active_lat:
                                try:
                                    geolocator = ArcGIS(user_agent="uoe_groundwater_dss")
                                    location_data = geolocator.reverse(f"{st.session_state.active_lat}, {st.session_state.active_lon}")
                                    if location_data:
                                        location_name = location_data.address.split(',')[0]
                                except Exception as geo_error:
                                    print(f"Geocoding failed: {geo_error}") 
                            
                            if st.session_state.active_lat:
                                lat, lon = st.session_state.active_lat, st.session_state.active_lon
                                ahp_val = round(get_ahp_suitability(lat, lon), 2)
                                e_yield, e_swl = run_idw_prediction(lat, lon, df_full)
                                
                                site_context = f"""[ACTIVE MAP SITE: Coordinates {lat}, {lon}. Town/Location Name: {location_name}. Dashboard calculated: Static Water Level = {e_swl}m, Yield = {e_yield}m3/hr, AHP = {ahp_val}/5. Use this IF asking about a specific site. If asked for nearby stores or centers, prioritize the Town/Location Name provided.]"""
                            else:
                                site_context = "[NO NO ACTIVE MAP SITE.]"

                            db_context = "[NO REGISTRY DATA AVAILABLE.]"
                            if not df_full.empty:
                                total_rows = len(df_full)
                                cols = ", ".join(df_full.columns)
                                missing_vals = df_full.isnull().sum().to_dict()
                                
                                avg_yield = df_full['yield_m3h'].mean() if 'yield_m3h' in df_full.columns else 0
                                avg_swl = df_full['swl_m'].mean() if 'swl_m' in df_full.columns else 0
                                
                                db_context = f"""[GLOBAL BOREHOLE REGISTRY STATS: Total Boreholes: {total_rows}. Columns available: {cols}. County Averages -> Yield: {avg_yield:.2f} m3/hr | SWL: {avg_swl:.2f} m. Missing Data Count by Column: {missing_vals}. Use this data IF the user asks you to analyze the registry, clean data, or find statistical trends.]"""

                            system_instruction = """
You are HydroBot, an elite Senior Hydrogeologist, Civil Engineer, and Advanced Data Analyst. 
You act as the primary AI Consultant for a Decision Support System mapping Groundwater Potential in Uasin Gishu County, Kenya.

[CORE IDENTITY & TONE]
1. You are brilliant, highly analytical, and deeply knowledgeable about groundwater hydrology, GIS spatial analysis, and drilling engineering.
2. Your tone is academic, professional, and helpful—like a highly respected engineering professor. 
3. You do not just give answers; you explain the *engineering reasoning* behind them step-by-step.

[LOCAL CONTEXT: UASIN GISHU COUNTY]
- You know that Uasin Gishu primarily features volcanic rocks (phonolites, basalts, tuffs). 
- Groundwater here is typically structurally controlled (found in faults, fractures, and weathered contacts).
- The general altitude is high (around 2000m+), affecting recharge rates and pumping head requirements.

[BEHAVIORAL RULES]
- GENERAL AI CAPABILITY: If the user asks a general question (e.g., "Write an email," "Explain quantum physics," "Help me format a document"), act as a helpful, world-class AI assistant and answer them brilliantly. You are not limited to *only* talking about water.
- HYDROGEOLOGY QUERIES: If the user asks about water, drilling, pumps, or maps, switch into "Senior Engineer" mode. Use technical terms correctly (Transmissivity, Storativity, Drawdown, Yield, SWL) but explain them simply if asked.
- DATA ANALYSIS: When given BOREHOLE REGISTRY STATS, act like a data scientist. Point out anomalies, suggest reasons for missing data, and provide statistical insights.
- SITE ANALYSIS: When given ACTIVE MAP SITE data, act like a consultant advising a client. Tell them if the site is financially viable for drilling based on the Expected Yield. Provide pump sizing estimates if requested.
- GOOGLE MAPS LINKS: If you suggest a supplier or physical location, provide a link exactly like this: [Store Name](https://www.google.com/maps/search/?api=1&query=Store+Name+City+Kenya). Replace spaces with (+).

NEVER output raw Python code unless explicitly asked by the user. Do not break character.
"""

                            api_messages = [{"role": "system", "content": f"{system_instruction}\n\n{site_context}\n{db_context}"}]
                            
                            for m in st.session_state.messages[-4:]:
                                api_messages.append({"role": m["role"], "content": m["content"]})

                            chat_completion = client.chat.completions.create(
                                messages=api_messages,
                                model="llama-3.3-70b-versatile", 
                            )
                            
                            bot_reply = chat_completion.choices[0].message.content
                            
                            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                            st.rerun()

                        except Exception as e:
                            st.error(f"AI Processing Error: {e}")

    with tab2:
        st.subheader("Live Borehole Registry")
        
        core_columns = ['borehole_id', 'lat', 'lon', 'yield_m3h', 'swl_m', 'total_depth_m']
        display_cols = [col for col in core_columns if col in df_full.columns]
        
        if display_cols:
            st.dataframe(df_full[display_cols], use_container_width=True)
        else:
            st.info("The required core columns could not be found in the uploaded data.")

if __name__ == '__main__':
    main()
