import pandas as pd
import numpy as np

# --- ais_train.csv preprocessing ---
def singleBoatCleanup(boat: pd.DataFrame, removeID: bool):
    """ 
    Input a Pandas dataframe of a single boat and output a cleaned version of the dataframe.
    """    
    #0-index everything
    boat.reset_index(drop=True, inplace=True)
    boat["time"] = boat["time"].astype("datetime64[ns]")

    if removeID:
        boat = boat.drop(columns=["vesselId", "portId"])

    boat["drift"] = boat["cog"] - boat["heading"]

    # --- Find last time we left a port ---
    #Figure out when navstat changes (2pac)
    boat["at_port"] = boat["navstat"].apply(lambda stat: True if stat == 5 else False)
    boat["change"] = boat["at_port"] != boat["at_port"].shift(-1)

    #Pick out rows with navstat == 5 and change == True
    wallah = boat[boat["navstat"] == 5]
    wallah = wallah[wallah["change"] == True]
    indices = wallah.index

    #Caveman fix
    boat["timestamp_last_port"] = np.nan

    for i in indices:
        boat.at[i, "timestamp_last_port"] = boat.at[i, "time"]
    boat["timestamp_last_port"] = boat["timestamp_last_port"].ffill()

    origin = boat.at[0, "time"]
    boat["timestamp_last_port"] = boat["timestamp_last_port"].fillna(origin)

    #Insh'allah
    boat["time_at_sea"] = boat["time"]-boat["timestamp_last_port"]
    boat["time_at_sea"] = boat["time_at_sea"].apply(lambda x: x if x >= pd.to_datetime(0) else pd.to_datetime(0))
    boat = boat.drop(columns=["change", "timestamp_last_port"])

    # --- etaRaw fixing ---
    boat["etaRaw"] = boat["etaRaw"].apply(lambda eta: "2024-" + eta)
    boat["etaRaw"] = pd.to_datetime(boat["etaRaw"], errors='coerce') #Set errors to NaT
    boat = boat.dropna(subset=["etaRaw"])

    boat["etaRaw"] = boat["etaRaw"].apply(lambda eta: pd.to_datetime(eta))

    # --- Fixing pd.timestamp and pd.timedelta to numerical ---
    boat["time"] = boat["time"].astype("int64") // (10**9)
    boat["etaRaw"] = boat["etaRaw"].astype("int64") // (10**9)
    boat["time_at_sea"] = boat["time_at_sea"].dt.total_seconds()

    return boat



