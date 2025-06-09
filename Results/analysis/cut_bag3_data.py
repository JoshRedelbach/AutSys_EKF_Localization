import pandas as pd

# Load the CSV file
path_og = "Final_Data/0_ICP_Pure_ONLY/00_Good_Init/bag3_downsample.csv"
path_dst = "Final_Data/0_ICP_Pure_ONLY/00_Good_Init/bag3_downsample_cut.csv"
df = pd.read_csv(path_og)

# Get the initial timestamp
start_time = df['time'].iloc[0]
end_time = df['time'].iloc[-1]

# Calculate the cutoff time
cutoff_time = start_time + 540  # 540 seconds later

# Print the start and end times
print(f"Start time: {start_time - start_time}, End time: {end_time - start_time}")
print(f"Cutoff time: {cutoff_time - start_time}")

# Filter the dataframe
df_cut = df[df['time'] <= cutoff_time]

# Save to a new file (or overwrite the original)
df_cut.to_csv(path_dst, index=False)
