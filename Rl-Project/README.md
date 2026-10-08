# OptiBed

OptiBed is an educational hospital bed-allocation simulation. A tabular Q-learning agent learns how many waiting patients to treat at each step. The standard episode has 8 beds, starts with 8 patients, and lasts 5 steps. This is a teaching project, not a clinical decision-support system.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard has **Simulation**, **Graphs**, and **Training** tabs. Simulation plays the learned policy; Training shows learning settings, episode summaries, and Q-learning updates. The agent trains for 600 episodes by default.

## Learning setup

The state summarizes available beds, waiting patients by severity, emergencies, and steps remaining. Actions are the number of patients treated. Training uses epsilon-greedy exploration. Default learning rate (`α`) is 0.15 and discount factor (`γ`) is 0.95.

## API and tests

Run the API with `python -m uvicorn api:app --reload`. It provides `/health`, `/reset`, `/state`, `/step`, and `/grade`.

Run tests with `python -m unittest discover -s tests -v`.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for architecture, simulation rules, and implementation details.
