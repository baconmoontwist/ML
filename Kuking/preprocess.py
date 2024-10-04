import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from geopy import distance

# --- ais_train.csv preprocessing ---
def singleBoatCleanup(boat: pd.DataFrame, removeID: bool) -> pd.DataFrame:
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
    boat["time_at_sea"] = boat["time_at_sea"].apply(lambda x: x if x >= pd.Timedelta(0) else pd.Timedelta(0))
    boat = boat.drop(columns=["change", "timestamp_last_port"])

    # --- etaRaw fixing ---
    boat["etaRaw"] = boat["etaRaw"].apply(lambda eta: "2024-" + eta)
    boat["etaRaw"] = pd.to_datetime(boat["etaRaw"], errors='coerce') #Set errors to NaT
    boat = boat.dropna(subset=["etaRaw"])

    boat["etaRaw"] = boat["etaRaw"].apply(lambda eta: pd.to_datetime(eta))

    # --- Fixing pd.timestamp and pd.timedelta to numerical ---
    """ boat["time"] = boat["time"].astype("int64") // (10**9)
    boat["etaRaw"] = boat["etaRaw"].astype("int64") // (10**9)
    boat["time_at_sea"] = boat["time_at_sea"].dt.total_seconds() """
    
    return boat

def fart(boat):
    boat["dist_s_l"] = 0
    boat["tot_dist"] = 0
    boat["time_s_l"] = 0
    boat["speed"] = 0
    k = 0
    boat = boat.reset_index(drop = True)
    for _,i in boat.iterrows():
        if k==0:
            k+=1
            j=i
        else:
            min1 = j
            min2 = i
            dist = distance.geodesic((min2['latitude'],min2['longitude']),(min1['latitude'],min1['longitude'])).km
            boat.at[k,"dist_s_l"] = dist
            boat.at[k,"tot_dist"] = boat.at[k-1,"tot_dist"] + dist
            tdddd = abs(j['time'].timestamp()-i['time'].timestamp())
            boat.at[k,"time_s_l"] = tdddd
            boat.at[k,"speed"] = dist/tdddd*3600
            k+=1
            j=i

    return boat

def ais_trainCleanup(path: str, name: str):
    """
    Given path to ais_train.csv, return cleaned dataframe with given name "name".
    """

    ais_train = pd.read_csv(path, sep="|")
    pathPre = "/".join(path.split("/")[:-1]) + "/"
    path_processed = pathPre + "/" + name

    #Find all unique ships
    uniqueVesselId = ais_train["vesselId"].unique()
    ais_train_processed_list = []

    for id in uniqueVesselId:
        #Isolate a single boat and clean
        boat = ais_train[ais_train["vesselId"] == id]
        boat = singleBoatCleanup(boat, False)

        #Appending rows to list
        for _, row in boat.iterrows():
            ais_train_processed_list.append(row.to_dict())

    ais_train_processed = pd.DataFrame(ais_train_processed_list).sort_values("time").reset_index(drop=True)

    ais_train_processed.to_csv(path_processed, sep="|", index=False)

