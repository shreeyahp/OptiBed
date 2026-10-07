# OptiBed — Hospital Bed Allocation Simulator

OptiBed is a hospital bed-allocation simulator for Normal and COVID wards. It combines a finite-state policy optimizer with a Streamlit dashboard for exploring demand, bed transfers, and patient discharges.

## Quick start

Open PowerShell in the project folder (currently `Rl-Project`) and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

To stop the dashboard, press `Ctrl+C` in the PowerShell window.

## Using the dashboard

1. Set the total number of beds and how many start in the Normal ward. Remaining beds are assigned to COVID.
2. Adjust each ward's average daily requests and discharges, the future-reward discount, simulation length, and random seed.
3. Select **Apply settings & reset** to calculate a policy for the new configuration.
4. Use **Start / resume** to run automatically, **Pause** to stop, **Step one day** to advance manually, or **Reset run** to restart.
5. Adjust **Time per simulated day** to control the animation pace. The default is 0.8 seconds per day.

The dashboard displays ward occupancy, the policy's suggested transfer, expected reward, a policy heatmap, daily results, and an event log. A positive transfer means Normal → COVID; a negative transfer means COVID → Normal.

## Model and reward

The policy's state tracks each ward's bed capacity and currently available beds. Independent Poisson distributions model daily requests and discharges. Discharges are limited to the number of occupied beds.

- Each unmet Normal request costs 10 reward points.
- Each unmet COVID request costs 20 reward points.
- Each transferred bed costs 5 reward points.

The simulation uses the configured random seed, so the same settings and seed produce the same run. Total capacity is limited to 15 beds to keep policy optimization interactive.

## Dependencies

Dependencies are listed in `requirements.txt`: NumPy, SciPy, pandas, and Streamlit.

Run the simulator tests with:

```powershell
python -m unittest discover -s tests -v
```
