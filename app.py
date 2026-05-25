import os
import warnings

# --- 🚨 SILENCE TERMINAL WARNINGS 🚨 ---
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.pop('PROJ_LIB', None)
os.environ.pop('PROJ_DATA', None)

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import psycopg2
from groq import Groq
from geopy.geocoders import ArcGIS 
from streamlit_geolocation import streamlit_geolocation

import matplotlib as mpl
import matplotlib.pyplot as plt
import io
import base64
import branca.colormap as cm

# --- PAGE CONFIG ---
st.set_page_config(page_title="UoE Groundwater DSS", layout="wide", initial_sidebar_state="collapsed")

# --- AWWDA CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F4F9FD; }
    p, label, h1, h2, h3, li { color: #1A202C !important; }
    h1, h2, h3 { color: #003B73 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; }
    
    header[data-testid="stHeader"] { background-color: transparent !important; }
    header[data-testid="stHeader"] * { color: #1A202C !important; fill: #1A202C !important; }
    
    div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #1A202C !important; border-color: #E1E8ED !important; }
    div[data-baseweb="select"] svg { fill: #1A202C !important; color: #1A202C !important; }
    div[data-baseweb="popover"] div, ul[role="listbox"] li { background-color: #ffffff !important; color: #1A202C !important; }
    
    [data-testid="stToggle"] input:not(:checked) + div { background-color: #CBD5E0 !important; } 
    [data-testid="stToggle"] input:checked + div { background-color: #00A8E8 !important; } 
    [data-testid="stToggle"] input + div > div { background-color: #FFFFFF !important; border: 1px solid #E1E8ED !important; } 

    [data-testid="stMetric"] { background-color: #ffffff; border-left: 5px solid #00A8E8; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    [data-testid="stMetricValue"] { color: #005A9C !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #4A5568 !important; font-weight: 600; }
    iframe { border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid #E1E8ED; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .stTabs [data-baseweb="tab"] p { color: #003B73 !important; font-weight: bold; }
    .stTabs [aria-selected="true"] { border-bottom: 4px solid #00A8E8; }
    </style>
""", unsafe_allow_html=True)

# --- ROBUST STATE MANAGEMENT (Memory) ---
if "active_lat" not in st.session_state:
    st.session_state.active_lat = None
    st.session_state.active_lon = None
if "map_center" not in st.session_state:
    st.session_state.map_center = [0.5143, 35.2697]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 11
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
    # 🚨 UPDATED MESSAGE TO REFLECT NEW CAPABILITIES 🚨
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am HydroBot. I can analyze specific map locations **OR** perform data analysis on the entire Borehole Registry. Ask me for pump sizing, drilling costs, or data cleaning insights!"}
    ]

# --- PATHS & ENV ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
RASTER_PATH = os.path.join(BASE_DIR, "data", "Uasin_Gishu_AHP.tif")

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

# --- DATABASE LOGIC ---
@st.cache_data(ttl=60)
def get_live_data():
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("DB_NAME", "groundwater_dss"), 
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASS", "wasike23"), 
            host=os.environ.get("DB_HOST", "localhost")
        )
        query = "SELECT borehole_id, sub_county, yield_m3h, swl_m, total_depth_m, ST_Y(geom) as lat, ST_X(geom) as lon FROM boreholes"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return pd.DataFrame()

# --- SPATIAL LOGIC ---
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
            check_df = df.dropna(subset=['lat', 'lon'])
            lons = check_df['lon'].tolist()
            lats = check_df['lat'].tolist()
            
            point_lons, point_lats = transform(CRS.from_epsg(4326), src.crs, lons, lats)
            coords = list(zip(point_lons, point_lats))
            
            vals = src.sample(coords)
            is_inside = []
            for val in vals:
                is_inside.append(val[0] > 0)
                
            return check_df[pd.Series(is_inside, index=check_df.index)]
    except Exception as e:
        return df 

def run_idw_prediction(lat, lon, df):
    if df.empty: return 0.0, 0.0, 0.0
    df['dist'] = df.apply(lambda r: distance.euclidean((lat, lon), (r['lat'], r['lon'])), axis=1)
    neighbors = df.nsmallest(5, 'dist')
    weights = 1.0 / (neighbors['dist']**2 + 1e-12)
    
    def interpolate(col):
        valid_mask = pd.notna(neighbors[col])
        if len(neighbors.loc[valid_mask, col]) == 0: return 0.0
        return round(np.dot(weights[valid_mask], neighbors.loc[valid_mask, col]) / np.sum(weights[valid_mask]), 2)

    return interpolate('yield_m3h'), interpolate('swl_m'), interpolate('total_depth_m')

# --- FAST-IDW ENGINE ---
@st.cache_data(ttl=300)
def generate_continuous_overlay(df_live, target_col, colormap_name):
    if df_live.empty or not os.path.exists(RASTER_PATH):
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

        vmin, vmax = np.nanmin(idw_grid), np.nanmax(idw_grid)
        cmap = mpl.colormaps[colormap_name].copy()
        cmap.set_bad(alpha=0.0) 
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
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

        return img_url, img_bounds, float(vmin), float(vmax)

# --- MAIN UI DASHBOARD ---
def main():
    st.title("💧 Groundwater Decision Support System — Uasin Gishu County")
    
    df_full = get_live_data()
    df_visible = get_visible_boreholes(df_full)
    
    tab1, tab2 = st.tabs(["🗺️ DSS Map & Virtual Borehole", "📊 Borehole Registry"])

    with tab1:
        col_map, col_analysis = st.columns([3.5, 1.5])
        
        with col_map:
            c1, c2, c3 = st.columns([1.5, 1, 1])
            with c1:
                st.selectbox("Basemap", ["Google Maps", "Google Satellite", "OpenStreetMap"], key="basemap", label_visibility="collapsed")
            with c2:
                st.toggle("🌊 Show Yield Surface", key="show_yield")
            with c3:
                st.toggle("💧 Show SWL Surface", key="show_swl")

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

            if not df_visible.empty:
                for _, row in df_visible.dropna(subset=['lat', 'lon']).iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], radius=4, color='#003B73', fill_color='#00A8E8', fill=True, fill_opacity=0.8,
                        popup=f"ID: {row['borehole_id']}"
                    ).add_to(m)
                
            if st.session_state.show_yield and not df_full.empty:
                y_img, y_bounds, y_min, y_max = generate_continuous_overlay(df_full, 'yield_m3h', 'Spectral')
                if y_img:
                    folium.raster_layers.ImageOverlay(image=y_img, bounds=y_bounds, opacity=0.75).add_to(m)
                    colormap_y = cm.LinearColormap(colors=['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd'], vmin=y_min, vmax=y_max)
                    colormap_y.caption = "Expected Yield (m³/hr)"
                    m.add_child(colormap_y)

            if st.session_state.show_swl and not df_full.empty:
                s_img, s_bounds, s_min, s_max = generate_continuous_overlay(df_full, 'swl_m', 'jet')
                if s_img:
                    folium.raster_layers.ImageOverlay(image=s_img, bounds=s_bounds, opacity=0.75).add_to(m)
                    colormap_s = cm.LinearColormap(colors=['#000080', '#0000ff', '#00ffff', '#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800000'], vmin=s_min, vmax=s_max)
                    colormap_s.caption = "Static Water Level (m)"
                    m.add_child(colormap_s)

            if st.session_state.active_lat:
                folium.Marker(
                    [st.session_state.active_lat, st.session_state.active_lon],
                    icon=folium.Icon(color='red', icon='info-sign'),
                    popup="Active Target"
                ).add_to(m)

            map_response = st_folium(m, height=650, use_container_width=True, returned_objects=["last_clicked"])

            if map_response and map_response.get('last_clicked'):
                current_click = map_response['last_clicked']
                if current_click != st.session_state.last_processed_click:
                    st.session_state.last_processed_click = current_click
                    st.session_state.active_lat = current_click['lat']
                    st.session_state.active_lon = current_click['lng']
                    st.session_state.map_center = [current_click['lat'], current_click['lng']]
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
                    st.error("⚠️ Target is outside the Uasin Gishu boundary. No data available.")
                else:
                    e_yield, e_swl, e_depth = run_idw_prediction(lat, lon, df_full)
                    norm_yield = min(e_yield, 5.0) 
                    gwp_score = round((ahp_val * W1) + (norm_yield * W2), 2)
                    
                    st.caption(f"📍 Coordinates: {round(lat,5)}, {round(lon,5)}")
                    
                    st.metric("Estimated Drill Depth", f"{e_depth} m")
                    st.metric("Expected SWL", f"{e_swl} m")
                    st.metric("Expected Yield", f"{e_yield} m³/hr")
                    st.metric("GIS Suitability (AHP)", f"{ahp_val} / 5")
                    
                    st.divider()
                    st.subheader(f"Overall GWP Index: {gwp_score} / 5")
                    
                    if gwp_score > 3.5: st.success("🟢 Highly Favorable for Drilling")
                    elif gwp_score > 2.0: st.warning("🟡 Moderate Risk Level")
                    else: st.error("🔴 High Dry-Hole Risk")
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
                            # 1. Site Context (If the user clicked the map)
                            if st.session_state.active_lat:
                                lat, lon = st.session_state.active_lat, st.session_state.active_lon
                                ahp_val = round(get_ahp_suitability(lat, lon), 2)
                                e_yield, e_swl, e_depth = run_idw_prediction(lat, lon, df_full)
                                
                                site_context = (
                                    f"[ACTIVE MAP SITE: Coordinates {lat}, {lon}. "
                                    f"Dashboard calculated: Total Depth = {e_depth}m, Static Water Level = {e_swl}m, "
                                    f"Yield = {e_yield}m3/hr, AHP = {ahp_val}/5. Use this IF asking about a specific site.]\n"
                                )
                            else:
                                site_context = "[NO ACTIVE MAP SITE.]\n"

                            # 🚨 2. NEW: Global Registry Context (Data Analysis) 🚨
                            db_context = "[NO REGISTRY DATA AVAILABLE.]"
                            if not df_full.empty:
                                total_rows = len(df_full)
                                cols = ", ".join(df_full.columns)
                                missing_vals = df_full.isnull().sum().to_dict()
                                
                                # Safe averages
                                avg_yield = df_full['yield_m3h'].mean() if 'yield_m3h' in df_full.columns else 0
                                avg_depth = df_full['total_depth_m'].mean() if 'total_depth_m' in df_full.columns else 0
                                avg_swl = df_full['swl_m'].mean() if 'swl_m' in df_full.columns else 0
                                
                                db_context = (
                                    f"[GLOBAL BOREHOLE REGISTRY STATS: "
                                    f"Total Boreholes: {total_rows}. Columns available: {cols}. "
                                    f"County Averages -> Yield: {avg_yield:.2f} m3/hr | Depth: {avg_depth:.2f} m | SWL: {avg_swl:.2f} m. "
                                    f"Missing Data Count by Column: {missing_vals}. "
                                    f"Use this data IF the user asks you to analyze the registry, clean data, or find statistical trends.]\n"
                                )

                            system_instruction = (
                                "You are HydroBot, a dual-purpose Data Analyst and Hydrogeology consultant for Uasin Gishu county. "
                                "You have access to two sets of data: The user's specific ACTIVE MAP SITE, and the GLOBAL BOREHOLE REGISTRY stats. "
                                "If the user asks about their specific location, use the MAP SITE data. "
                                "If the user asks about overall data quality, averages, cleaning, or registry analysis, use the REGISTRY STATS. "
                                "CRITICAL LINK INSTRUCTION: If you suggest a supplier or location, provide a Google Maps search link formatted exactly like this: [Store Name](https://www.google.com/maps/search/?api=1&query=Store+Name+City+Kenya). Replace spaces with (+). "
                                "Be analytical, professional, and directly reference the stats provided to you. Do not output Python code."
                            )

                            # Combine instructions with both contexts
                            api_messages = [{"role": "system", "content": f"{system_instruction}\n\n{site_context}\n{db_context}"}]
                            
                            for m in st.session_state.messages[-4:]:
                                api_messages.append({"role": m["role"], "content": m["content"]})

                            chat_completion = client.chat.completions.create(
                                messages=api_messages,
                                model="llama-3.1-8b-instant", 
                            )
                            
                            bot_reply = chat_completion.choices[0].message.content
                            
                            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                            st.rerun()

                        except Exception as e:
                            st.error(f"AI Processing Error: {e}")

    with tab2:
        st.subheader("PostGIS Live Borehole Registry")
        st.dataframe(df_full, use_container_width=True)

if __name__ == '__main__':
    main()