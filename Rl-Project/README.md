# OptiBed

OptiBed models bed allocation between Normal and COVID wards. Its Streamlit frontend lets you configure demand and discharge rates, inspect the learned transfer policy, and run a repeatable, step-by-step hospital simulation.

## Run the frontend

From the `OptiBed` project folder, install the dependencies and start the dashboard:

```powershell
cd "C:\Users\shreeya\Desktop\btech\sem7\lab\rl lab\hospital\OptiBed"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser.

Run the simulator tests with:

```powershell
python -m unittest discover -s tests -v
```

## Dashboard features

- Configure total beds, the initial ward split, per-ward Poisson arrival and discharge rates, the discount rate, simulation duration, and random seed.
- Start/resume, pause, manually step, or reset the run.
- Adjust the delay between simulated days (0.2–2.0 seconds).
- Watch ward availability and occupancy, unmet requests, rewards, and transfers update as days progress.
- Inspect the learned policy heatmap for the currently selected capacity split and the event log for each simulated day.
- Compare the expected one-day reward with no transfer versus the current policy recommendation.

The model uses available beds as the state. A positive transfer moves a free bed from Normal to COVID; a negative transfer moves one from COVID to Normal. Unmet Normal requests cost 10 reward points each, unmet COVID requests cost 20, and each transferred bed costs 5. Poisson arrivals/discharges are independent, and discharges cannot exceed the number of occupied beds in their ward.

The simulation is stochastic but reproducible for a given seed. Total capacity is limited to 15 beds so the finite-state policy can be optimized interactively.
