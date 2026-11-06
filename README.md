# SmartGrid-Campus-AI-Energy-Manager
A Prototype that is based on SmartGrid Campus called as an AI Energy Manager. This is used to control the things like amount of energy consumed and amount of time that energy and current consumed. Then it's been evaluating about how much volt it taken and remaining the amount of volt and the time when exceeding larger amount of voltage. 
# SmartGrid Campus (Track 5)
AI energy management: monitor campus load, forecast the next 24 hours, flag peaks, find idle-load waste and detect abnormal consumption.
## Run on your laptop
    pip install -r requirements.txt
    streamlit run app.py
Open the link shown (usually http://localhost:8501). Phones on the same Wi-Fi can open the "Network URL".
## Use your own data
Upload a CSV in the sidebar: first column = timestamp, then one column per zone in kW (download the template from the sidebar). At least 3 days of readings.
## Share it online (free)
Push these files to a GitHub repo and deploy on Streamlit Community Cloud.
