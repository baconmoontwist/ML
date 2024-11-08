#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
from sklearn.impute import KNNImputer
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

#Evaluations
def evaluate(model: str, y_pred, y_test):

    n = len(y_test)
    p = X_test.shape[1]

    mse_latitude = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_longitude = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    r2_latitude = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_latitude_adj = 1 - (1 - r2_latitude) * ((n - 1) / (n - p - 1))
    r2_longitude = r2_score(y_test[:, 1], y_pred[:, 1])
    r2_longitude_adj = 1 - (1 - r2_longitude) * ((n - 1) / (n - p - 1))


    print(f"---- {model} Metrics ----")
    print(f"Mean Squared Error (Latitude): {mse_latitude:.4f}")
    print(f"R-squared (Latitude): {r2_latitude:.4f}, Adjusted R-squared (Latitude): {r2_latitude_adj: .4f}")
    print(f"Mean Squared Error (Longitude): {mse_longitude:.4f}")
    print(f"R-squared (Longitude): {r2_longitude:.4f}, Adjusted R-squared (Latitude): {r2_longitude_adj: .4f}")

def visualize_vessel_movements(df):
    """
    Visualize vessel movements on a map with lines and markers for each data point.

    Parameters:
    - df (pandas.DataFrame): A DataFrame with columns ['time', 'latitude', 'longitude', 'vesselId'].

    Returns:
    - A Plotly interactive figure.
    """
    # Ensure 'time' is in datetime format for better tooltip handling
    df['time'] = pd.to_datetime(df['time'])
    
    # Sorting the DataFrame by time to ensure lines are drawn correctly
    df = df.sort_values(by=['vesselId', 'time'])

    # Define a color palette
    color_map = px.colors.qualitative.Plotly

    # Mapping each vessel ID to a color
    unique_vessels = df['vesselId'].unique()
    colors = {vessel_id: color_map[i % len(color_map)] for i, vessel_id in enumerate(unique_vessels)}

    # Create the base map with lines
    fig = px.line_geo(df,
                      lat='latitude',
                      lon='longitude',
                      color='vesselId',
                      color_discrete_map=colors,
                      hover_name='vesselId',
                      hover_data={'time': True, 'latitude': ':.3f', 'longitude': ':.3f'},
                      projection='natural earth',
                      title='Vessel Movements Over Time')

    # Add markers for each data point
    for vessel_id in unique_vessels:
        vessel_data = df[df['vesselId'] == vessel_id]
        fig.add_trace(go.Scattergeo(
            lon=vessel_data['longitude'],
            lat=vessel_data['latitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=colors[vessel_id],
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=f'Markers for {vessel_id}',
            hoverinfo='text',
            text=vessel_data.apply(lambda row: f'ID: {vessel_id}<br>Time: {row["time"]}<br>Lat: {row["latitude"]:.3f}<br>Lon: {row["longitude"]:.3f}', axis=1)
        ))

    # Enhancing map and layout details
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="RebeccaPurple")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0},
                      coloraxis_colorbar=dict(title="Vessel ID"),
                      title_font_size=20)
    
    return fig

#Cleans
def delta(boat: pd.DataFrame) -> pd.DataFrame:
    """
    Given boat, return dataframe with delta_lat and delta_lon added
    """
    boat.sort_values("time", inplace=True)
    boat.reset_index(drop=True, inplace=True)

    boat["delta_time"] = boat["time"] - boat["time"].shift(1)
    boat["delta_time"] = boat["delta_time"].dt.total_seconds()
    boat["time_cum"] = boat["delta_time"].cumsum()

    boat["delta_lat"] = boat["latitude"] - boat["latitude"].shift(1)
    boat["delta_lon"] = boat["longitude"] - boat["longitude"].shift(1)

    boat["delta_lat_cum"] = boat["delta_lat"].cumsum()
    boat["delta_lon_cum"] = boat["delta_lon"].cumsum()

    #Degree stuff
    boat["cog"] = (boat["cog"] * np.pi )/180
    boat["heading"] = (boat["heading"] * np.pi)/180

    return boat

def speed(boat: pd.DataFrame) -> pd.DataFrame:
    """
    Given boat, upper and lower speed limit, calculate speed between observations and return boat dataframe with speed.
    Will also fix insane speed using upper/lower. Generally, vessels we care about will not go faster than 60km/h.
    Run AFTER delta()
    """
    boat.sort_values("time", inplace=True)
    boat.reset_index(drop=True, inplace=True)
    max_speed = boat["maxSpeed"].iloc[0] * 1.852

    boat["speed_from_prev"] = np.nan
    boat["dist_from_prev"] = np.nan
    l = len(boat)

    for i in range(1, l):
        boat.at[i, "dist_from_prev"] = distance(
            (boat.at[i-1, "latitude"], boat.at[i-1, "longitude"]),
            (boat.at[i, "latitude"], boat.at[i, "longitude"])
        ).km

    boat["speed_from_prev"] = (boat["dist_from_prev"] / (boat["delta_time"] / 3600))
    boat["speed_from_prev"].clip(lower=0, upper=max_speed, inplace=True)

    boat["dist_cum"] = boat["dist_from_prev"].cumsum()

    return boat

def moored(boat: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate when we are at a port and then calculate the time at sea, set as seconds. 
    Also calculate the distance from last port. Lastly, classify if boat is deep sea or not based on dist
    """
    boat["at_port"] = boat["navstat"].apply(lambda stat: True if stat == 5 else False)
    indices = boat[boat["at_port"] == True].index

    boat["timestamp_last_port"] = np.nan
    boat["lat_port"] = np.nan
    boat["lon_port"] = np.nan

    for i in indices:
        boat.at[i, "timestamp_last_port"] = boat.at[i, "time"]
        boat.at[i, "lat_port"] = boat.at[i, "latitude"]
        boat.at[i, "lon_port"] = boat.at[i, "longitude"]

    boat["timestamp_last_port"] = boat["timestamp_last_port"].ffill()
    boat["lat_port"] = boat["lat_port"].ffill()
    boat["lon_port"] = boat["lon_port"].ffill()

    origin = boat.at[0, "time"]
    origin_lat, origin_lon = boat.at[0, "latitude"], boat.at[0, "longitude"]
    boat["timestamp_last_port"] = boat["timestamp_last_port"].fillna(origin)
    boat["lat_port"] = boat["lat_port"].fillna(origin_lat)
    boat["lon_port"] = boat["lon_port"].fillna(origin_lon)

    #Insh'allah
    boat["time_at_sea"] = boat["time"]-boat["timestamp_last_port"]
    boat["time_at_sea"] = boat["time_at_sea"].apply(lambda x: x if x >= pd.Timedelta(0) else pd.Timedelta(0))
    boat["time_at_sea"] = boat["time_at_sea"].dt.total_seconds()

    #Distance since last port
    l = len(boat)
    boat["dist_last_port"] = np.nan

    for i in range(1,l):
        dist = distance((boat.at[i, "latitude"], boat.at[i, "longitude"]),
                        (boat.at[i, "lat_port"], boat.at[i, "lon_port"])).km
        
        boat.at[i, "dist_last_port"] = dist

    #Deep sea?
    boat["deep_sea"] = 0
    
    if boat["dist_last_port"].max() > 100:
        boat["deep_sea"] = 1

    boat = boat.drop(columns=["timestamp_last_port", "lat_port", "lon_port"])

    return boat

def navstat(boat: pd.DataFrame, speed_lim = 0.5) -> pd.DataFrame:
    """
    Clip NAVSTAT according to speed_lim. Run AFTER speed()
    """
    for i, row in boat.iterrows():
        speed = row["speed_from_prev"]
        navstat = row["navstat"]

        if navstat > 8:
            if speed <= speed_lim: #Standing still basically
                boat.at[i, "navstat"] = 1 #Anchored
            else:
                boat.at[i, "navstat"] = 0 #Moving with engine

    return boat

def resample(boat: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the boats, kinda stupid implementation but it kinda works???
    """
    t_0 = boat["time"].iloc[0].floor("20min") #Round down to nearest 15-min
    boat.set_index("time", drop=True, inplace=True)

    boat = boat.resample("20min", origin=t_0,).nearest()

    boat.reset_index(drop=False, inplace=True)

    return boat

def lagged(bruh: pd.DataFrame) -> pd.DataFrame:
    #Delta Lat/lon

    for i in range(1,6):
        bruh[f"delta_lat_lag_{i}"] = bruh["delta_lat"].shift(i)
        bruh[f"delta_lon_lag_{i}"] = bruh["delta_lon"].shift(i) 

    for i in range(1,6):
        bruh[f"lat_lag_{i}"] = bruh["latitude"].shift(i)
        bruh[f"lon_lag_{i}"] = bruh["longitude"].shift(i) 

    #Speed and distance
    # bruh["speed_lag_1"] = bruh["speed_from_prev"].shift(1)
    # bruh["speed_lag_2"] = bruh["speed_from_prev"].shift(2)

    # bruh["dist_lag_1"] = bruh["dist_from_prev"].shift(1)
    # bruh["dist_lag_2"] = bruh["dist_from_prev"].shift(2)

    #NAVSTAT
    # bruh["navstat_lag_1"] = bruh["navstat"].shift(1)
    # bruh["navstat_lag_2"] = bruh["navstat"].shift(2)

    bruh.dropna(inplace=True)
    return bruh

def cleanUp(data: pd.DataFrame, n=688, resample=False, eta=False, time=False) -> pd.DataFrame:
    """
    data: smirk, n: how many boats to clean, resample: smirk, eta: smrk, time: fix time or no?
    """

    wallahi = []

    #Time first
    data["time"] = pd.to_datetime(data["time"])

    #Individual boat stuff
    ids = data["vesselId"].unique()

    for i in ids[:n+1]:
        boat = data[data["vesselId"] == i].reset_index(drop=True)

        if resample:
            boat = resample(boat)    

        boat = lagged(navstat(speed(delta(moored(boat)))))

        for _, row in boat.iterrows():
            wallahi.append(row.to_dict())

    wallahi = pd.DataFrame(wallahi)

    #Fix etaRaw: Set to seconds from 2024-01-01 00:00:00 if possible, if invalid set to pd.NaT and then interpolat
    #using nearest()
    if eta:
        origin = pd.to_datetime("2024-01-01 00:00:00")
        
        wallahi["etaRaw"] = wallahi["etaRaw"].apply(lambda e: "2024-" + e)
        wallahi["etaRaw"] = pd.to_datetime(wallahi["etaRaw"], errors="coerce")
        wallahi["etaRaw"] = wallahi["etaRaw"].apply(lambda e: (e-origin) if not pd.isna(e) else e)
        wallahi["etaRaw"] = wallahi["etaRaw"].dt.total_seconds()
        wallahi["etaRaw"].interpolate("nearest", inplace=True)

    #TIME SHIT
    if time:
        wallahi["month"] = wallahi["time"].apply(lambda t: t.month)
        wallahi["day"] = wallahi["time"].apply(lambda t: t.day_of_week)
        wallahi["hour"] = wallahi["time"].apply(lambda t: t.hour)

        origin = pd.to_datetime("2024-01-01 00:00:00")
        wallahi["time"] = (wallahi["time"] - origin).dt.total_seconds()

    wallahi.bfill(inplace=True)

    wallahi.sort_values("time", inplace=True)

    return wallahi

print("ya allah")
