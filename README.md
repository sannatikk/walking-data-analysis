# Walking Analytics: Accelerometer & GPS data
## Applied Physics in Software Development (2.5 ECTS), course project

For my project, I went on a quick walk with Linear Acceleration and GPS tracking turned on in the Phyphox smartphone app. I analyzed accelerometer data to calculate step statistics based on both Fourier transformations and a lowpass filter, observed GPS and subsequent velocity accuracy, and visualized my route on an open-source map.

**Run the following code in your terminal to view my Streamlit app:**  

`streamlit run https://raw.githubusercontent.com/sannatikk/walking-data-analysis/main/quick_walk.py`  

_Note: please make sure you have the Streamlit Python library installed on your computer before attempting to run._

**Required elements in the project:**
- Choose correct accelerometer axis to analyze for optimal step count data
- Define the following elements from accelerometer and GPS data:
    - Step count using filtering of acceleration data
    - Step count using Fourier analysis of acceleration data
    - Average velocity 
    - Total distance
    - Average step length
- Present visualization for the following:
    - Filtered acceleration data used for calculating step count 
    - Power Spectral Density of acceleration data from chosen data axis 
    - Walked route on a map

My app includes **all of the above, as well as** some extra visualization of steps taken and data analysis. A **summary of results** is at the very bottom of the Streamlit browser app.

Present in my GitHub repository is also my Jupyter Notebook file used for initial sketching out of the project, metadata folder with device used and experiment time info, as well as a map of my route saved as an html file. 