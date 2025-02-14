import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium 
from scipy.signal import butter, filtfilt
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Paths to the CSV files
path_acc = "https://raw.githubusercontent.com/sannatikk/walking-data-analysis/refs/heads/main/Linear%20Accelerometer.csv"
path_gps = "https://raw.githubusercontent.com/sannatikk/walking-data-analysis/refs/heads/main/Location.csv"

# Load data
df_acc = pd.read_csv(path_acc)
df_gps = pd.read_csv(path_gps)

# Streamlit UI
st.title("Walking Data Analysis")
st.write("This app analyzes walking data taken using the Phyphox app for smartphones. The data includes linear acceleration and GPS data.")


### DISPLAY RAW DATA ###


# Y-axis Linear Acceleration vs. Time
st.header("Linear Acceleration vs. Time")

st.subheader("Raw Acceleration Data")
st.line_chart(df_acc, x = 'Time (s)', y = 'Y (m/s^2)', y_label = 'Linear Acceleration (m/s^2)', x_label = 'Time (s)')

# Zoomed-in version
st.subheader("Raw Acceleration Data, Zoomed-in View")
time_min, time_max = st.slider("Select Time Range", min_value=float(df_acc['Time (s)'].min()), max_value=float(df_acc['Time (s)'].max()), value=(200.0, 230.0))

df_zoomed = df_acc[(df_acc['Time (s)'] >= time_min) & (df_acc['Time (s)'] <= time_max)]
st.line_chart(df_zoomed, x = 'Time (s)', y = 'Y (m/s^2)', y_label = 'Linear Acceleration (m/s^2)', x_label = 'Time (s)')


### FOURIER TRANSFORM & POWER SPETRAL DENSITY ###


# Fourier Transform and Power Spectral Density (PSD)
st.header("Power Spectral Density (Fourier Transform)")
f = df_acc['Y (m/s^2)']
t = df_acc['Time (s)']
N = len(f)
dt = np.max(t) / N
fourier = np.fft.fft(f, N)
psd = (fourier * np.conj(fourier) / N).real 
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, np.floor(N / 2), dtype='int')

# DataFrame for the power spectral density
df_psd = pd.DataFrame({"Frequency (Hz)": freq[L], "Power": psd[L]})
df_psd = df_psd[df_psd["Frequency (Hz)"] > 0]

st.line_chart(df_psd, x = 'Frequency (Hz)', y = 'Power', y_label = 'Power', x_label = 'Frequency (Hz)')

# Zoomed-in version
st.subheader("Power Spectral Density (Zoomed-in View)")
freq_min, freq_max = st.slider("Select Frequency Range", min_value=1.0, max_value=6.0, value=(1.5, 5.0))

df_psd_zoomed = df_psd[(df_psd["Frequency (Hz)"] >= freq_min) & (df_psd["Frequency (Hz)"] <= freq_max)]
st.line_chart(df_psd_zoomed, x = 'Frequency (Hz)', y = 'Power', y_label = 'Power', x_label = 'Frequency (Hz)')


### STEP COUNTING USING FOURIER TRANSFORM ###


# Fourier Transform for Step Counting
st.subheader("Step Detection using Fourier Transform")

f = df_acc['Y (m/s^2)']
t = df_acc['Time (s)']
N = len(f)
dt = np.max(t) / N
fourier = np.fft.fft(f, N)
psd = (fourier * np.conj(fourier) / N).real
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, np.floor(N / 2), dtype='int')

# Find max frequency
f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1 / f_max  # Step period in seconds
steps_fourier = np.max(t) * f_max  # Estimated steps

fourier_step_period = T

st.write(f"**Maximum Frequency:** {f_max:.2f} Hz")
st.write(f"**Step Period (Fourier-based):** {fourier_step_period:.2f} seconds")
st.write(f"**Estimated Steps (Fourier-based):** {steps_fourier:.0f}")



### LOWPASS FILTERING ###



# Lowpass Filtering
st.header("Filtered Acceleration Data")

st.write("**Filtering method**: Butterworth Lowpass Filter")

def butter_lowpass(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Parameters
aika = df_acc['Time (s)']
kiihtyvyys = df_acc['Y (m/s^2)']

T = aika.max() - aika.min()
n = len(aika)
fs = n / T 
nyq = fs / 2
order = 3
cutoff = 1 / 0.5  # Adjustable cutoff frequency

filtered_signal = butter_lowpass(kiihtyvyys, cutoff, fs, nyq, order)

# Plot Unfiltered vs. Filtered Acceleration
st.subheader("Raw vs. Lowpass-Filtered Acceleration Data")
df_filtered = pd.DataFrame({
    "Time (s)": aika,
    "Raw Acceleration": kiihtyvyys,
    "Filtered Acceleration": filtered_signal
}).set_index("Time (s)")
st.line_chart(df_filtered, x_label="Time (s)", y_label="Acceleration (m/s^2)")

# Zoomed-in View
st.subheader("Zoomed-in View")
time_min, time_max = st.slider("Select Time Range", min_value=float(aika.min()), max_value=float(aika.max()), value=(100.0, 200.0))

df_zoomed = df_filtered.loc[time_min:time_max]
st.line_chart(df_zoomed, x_label="Time (s)", y_label="Acceleration (m/s^2)")

# Filtered Signal Only
st.subheader("Filtered Acceleration Only")

# Create a DataFrame with only the filtered signal
df_filtered_signal = pd.DataFrame({
    "Time (s)": aika,
    "Filtered Acceleration": filtered_signal
}).set_index("Time (s)")

# Display the filtered signal in a line chart
st.line_chart(df_filtered_signal, x_label="Time (s)", y_label="Acceleration (m/s²)")


# Zoomed-in Filtered View
st.subheader("Filtered Acceleration (Zoomed-in View)")
time_min_zoom, time_max_zoom = st.slider("Select Zoomed Time Range", min_value=float(aika.min()), max_value=float(aika.max()), value=(100.0, 150.0))

df_filtered_zoomed = df_filtered.loc[time_min_zoom:time_max_zoom]
st.line_chart(df_filtered_zoomed[["Filtered Acceleration"]], x_label="Time (s)", y_label="Acceleration (m/s²)")


### STEP COUNTING USING LOWPASS FILTER ###


st.subheader("Step Detection using Butterworth Filter")

ylitykset = 0
for i in range(n-1): 
    if filtered_signal[i] / filtered_signal[i+1] < 0: 
        ylitykset = ylitykset + 1

steps_lowpass = ylitykset / 2

st.write(f"**Estimated Steps (Lowpass Filtered):** {steps_lowpass:.0f}")



### GPS ANALYSIS ###


st.header("GPS Analysis")

# Haversine Distance Calculation
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in km
    return c * r * 1000  # Convert to meters

# Calculate distances, time differences, and velocity
df_gps['dist'] = np.zeros(len(df_gps))
df_gps['time_diff'] = np.zeros(len(df_gps))

for i in range(len(df_gps) - 1):
    df_gps.loc[i, 'dist'] = haversine(df_gps.loc[i, 'Longitude (°)'], df_gps.loc[i, 'Latitude (°)'], df_gps.loc[i + 1, 'Longitude (°)'], df_gps.loc[i + 1, 'Latitude (°)'])
    df_gps.loc[i, 'time_diff'] = df_gps.loc[i + 1, 'Time (s)'] - df_gps.loc[i, 'Time (s)']

df_gps['velocity'] = df_gps['dist'] / df_gps['time_diff']
df_gps['tot_dist'] = np.cumsum(df_gps['dist'])

# Calculate total distance and average velocity
total_distance = df_gps['tot_dist'].iloc[-1]
velocity_avg = total_distance / df_gps['Time (s)'].iloc[-1]

# GPS Accuracy and Velocity Plots

st.subheader("GPS Accuracy")
st.line_chart(df_gps.set_index("Time (s)")[["Horizontal Accuracy (m)", "Vertical Accuracy (°)"]], x_label="Time (s)", y_label="Accuracy (m)")

st.subheader("Velocity")
st.line_chart(df_gps.set_index("Time (s)")[["velocity"]], x_label="Time (s)", y_label="Velocity (m/s)")


# Display total distance and average velocity
st.write(f"**Total Distance, calculated with Haversine formula:** {total_distance:.2f} meters")
st.write(f"**Average Velocity:** {velocity_avg:.2f} m/s")

# Interactive Folium Map
st.subheader("GPS Path Visualization")

my_map = folium.Map(location=[df_gps['Latitude (°)'].mean(), df_gps['Longitude (°)'].mean()], zoom_start=16)
folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']], color="blue", weight=2.5, opacity=1).add_to(my_map)
st_map = st_folium(my_map, width=900, height=650)

# Step Length Calculation
steps_avg = (steps_lowpass + steps_fourier) / 2
step_length = total_distance / steps_avg

# Final Results Display
st.subheader("Final Results Summary")

st.write(f"**Fourier-Based Step Count:** {steps_fourier:.0f}")
st.write(f"**Butterworth-Based Step Count:** {steps_lowpass:.0f}")
st.write(f"**Average Step Count:** {steps_avg:.0f}")
st.write(f"**Average Step Length:** {step_length:.2f} meters")
st.write(f"**Fourier Step Period:** {fourier_step_period:.2f} seconds")
st.write(f"**Average Velocity:** {velocity_avg:.2f} m/s")
st.write(f"**Total Distance:** {total_distance:.2f} meters")
