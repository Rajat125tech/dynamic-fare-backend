import pandas as pd

ola = pd.read_csv("new_ola.csv", low_memory=False)

# Clean column names properly
ola.columns = (
    ola.columns
        .str.replace(r'\+AF8-', '_', regex=True)  # remove encoding artifacts
        .str.replace('+', '_', regex=False)       # replace + with _
        .str.strip()
        .str.lower()
)

# Convert important numeric columns properly
numeric_cols = [
    'distance',
    'driver_tip',
    'mta_tax',
    'toll_amount',
    'extra_charges',
    'improvement_charge',
    'total_amount',
    'num_passengers'
]

for col in numeric_cols:
    ola[col] = (
        ola[col]
        .astype(str)
        .str.replace(r'[^0-9.]', '', regex=True)
    )
    ola[col] = pd.to_numeric(ola[col], errors='coerce')

# Drop rows with missing important values
ola = ola.dropna(subset=['distance', 'total_amount'])

print(ola[['distance','total_amount']].corr())

'''print(ola.columns)

ola['total_amount'] = (
    ola['total_amount']
    .astype(str)
    .str.replace(r'[^0-9.]', '', regex=True)
)

ola['distance'] = pd.to_numeric(ola['distance'], errors='coerce')
ola['total_amount'] = pd.to_numeric(ola['total_amount'], errors='coerce')

ola = ola.dropna(subset=['distance', 'total_amount'])'''

