# OptiBed

OptiBed is a hospital bed-allocation simulation in which a tabular Q-learning agent learns how many waiting patients to treat at each step. The agent is trained on the environment, then runs a learned policy in the dashboard. Emergency and high-severity patients are prioritized when beds are allocated. The episode uses 8 beds, starts with 8 patients, and runs for 20 steps.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard shows the OptiBed title and episode description, followed by the agent overview. The **Simulation** tab has **Start**, **Stop**, and **Reset** controls. Start automatically advances the episode at a slow pace (one step every 1.8 seconds); no manual allocation is required. Hospital beds and the waiting-patient queue are shown in equal-width, equal-height panels. Occupied beds identify their current patient, and waiting patients use severity-colored user icons with spaced labels; non-emergency patients do not repeat a "Waiting" badge inside the waiting queue. Empty queues show a quiet empty state, and the initial screen avoids an extra prompt banner. After every decision, the dashboard displays the observed state, chosen allocation, Q-value before and after its online update, reward breakdown, treated patients, next state, and new arrivals. A step-by-step reward table shows gross reward, penalty, and final net result for each action. The **Graphs** tab shows waiting patients by severity, beds in use versus available, and step/cumulative reward, each with a legend. The **Training** tab shows training metrics, a frozen post-training Q-table, and Q-learning updates from steps executed in the current simulation.

The dashboard trains the agent once when the app starts (1,200 training episodes; subsequent page reruns reuse the trained agent). It uses epsilon-greedy exploration during training and the greedy learned policy during the displayed episode. Its compact state tracks available beds, waiting high/medium/low patient counts, emergency count, and steps remaining; each queue count is capped at 16 to keep the tabular state space bounded. The Q-learning update is:

```text
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

This is a small educational tabular agent, not a clinical decision-support system.

For a detailed walkthrough of the architecture, simulation rules, Q-learning implementation, dashboard, API, and tests, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

Start the API separately if needed:

```powershell
python -m uvicorn api:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`.

## Environment rules

- Regular arrivals each step: Poisson(1.5). One emergency patient has an independent 20% chance to arrive each step.
- Regular patient severity: low (50%), medium (30%), or high (20%). Emergency patients are high severity.
- Waiting patients have a 20% chance per step of worsening by one level.
- A treated patient occupies a bed for 1–3 steps.
- The agent chooses an integer allocation from zero to the smaller of available beds and waiting patients. Emergency patients are selected first, then high/medium/low severity, then earlier arrivals. The HTTP API still exposes the environment directly for external agents that submit their own allocation.

### Rewards

| Event | Reward |
|---|---:|
| Treat low / medium / high severity | +1 / +2 / +3 |
| Treat an emergency patient | +1 bonus |
| Leave a high-severity patient untreated for a step | −2 |
| Leave an emergency patient untreated for a step | −1.5 |
| Leave an available bed unused while people are waiting | −0.3 per bed |
| Patient deterioration | −0.5 per severity level |

## API

Run tests with `python -m unittest discover -s tests -v`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode; optional `seed` query parameter. |
| `POST` | `/step` | Submit `{"allocate": 2}` and receive observation, reward, and details. |
| `GET` | `/state` | Current beds, waiting patients, and episode progress. |
| `GET` | `/grade` | Normalized grade and cumulative reward. |
| `GET` | `/health` | API health check. |
