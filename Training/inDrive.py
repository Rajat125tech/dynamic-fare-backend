import pandas as pd
indrive = pd.read_csv("indrive.csv")

indrive.columns = indrive.columns.str.strip()

# ---------- CREATE DATETIME ----------
indrive['datetime'] = pd.to_datetime(
    indrive['Ride_DateTime'],
    format='%d-%m-%y %H:%M'
)

# ---------- RENAME COLUMNS ----------
indrive.rename(columns={
    'Pickup_Location':'pickup',
    'Dropoff_Location':'drop',
    'Distance_km':'distance',
    'Ride_Duration_min':'duration',
    'Fare_Amount':'fare',
    'Vehicle_Type':'vehicle_type',
    'Ride_Status':'status',
    'Weather_Condition':'weather',
    'Traffic_Level':'traffic',
    'Surge_Pricing':'surge'
}, inplace=True)

# ---------- KEEP IMPORTANT COLUMNS ----------
indrive = indrive[[
    'datetime',
    'pickup',
    'drop',
    'distance',
    'duration',
    'fare',
    'vehicle_type',
    'status',
    'weather',
    'traffic',
    'surge'
]]
print(indrive['status'].unique())   # check values first

indrive = indrive[indrive['status'] == 'Completed']  # adjust if needed

indrive['hour'] = indrive['datetime'].dt.hour
indrive['day'] = indrive['datetime'].dt.dayofweek
indrive['month'] = indrive['datetime'].dt.month
indrive['is_weekend'] = indrive['day'].isin([5,6]).astype(int)

indrive = indrive[indrive['fare'] < indrive['fare'].quantile(0.99)]
indrive.to_csv("clean_indrive.csv", index=False)

print("inDrive cleaned dataset saved!")