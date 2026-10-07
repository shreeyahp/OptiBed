"""OptiBed dashboard for the hospital bed-allocation simulator."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from rl_model import HospitalSimulator, PolicySolver, SimulationConfig


st.set_page_config(
    page_title="OptiBed | Hospital Bed Allocation",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #f5f8fc;
        border: 1px solid #e4eaf2;
        padding: 0.9rem 1rem;
        border-radius: 0.75rem;
    }
    .ward-card {
        border: 1px solid #e4eaf2;
        border-radius: 0.9rem;
        padding: 1.1rem 1.25rem;
        background: #ffffff;
    }
    .subtle {color: #65758b; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OptiBed")
st.caption(
    "Hospital bed allocation simulation: explore how transfers, patient arrivals, "
    "and discharges affect two wards."
)

if "config" not in st.session_state:
    st.session_state.config = SimulationConfig()
    st.session_state.solver = PolicySolver(st.session_state.config)
    st.session_state.simulator = HospitalSimulator(st.session_state.config)
    st.session_state.running = False
    st.session_state.last_tick = time.monotonic()

with st.sidebar:
    st.header("Hospital setup")
    with st.form("configuration"):
        total_beds = st.number_input(
            "Total hospital beds", min_value=2, max_value=15,
            value=st.session_state.config.total_beds, step=1,
        )
        normal_beds = st.slider(
            "Beds initially allocated to Normal",
            min_value=0,
            max_value=int(total_beds),
            value=min(st.session_state.config.normal_beds, int(total_beds)),
        )
        st.caption(f"COVID ward starts with {int(total_beds) - normal_beds} beds.")

        st.subheader("Daily patient flow")
        normal_arrivals = st.number_input(
            "Normal requests (Poisson mean)", 0.0, 10.0,
            st.session_state.config.normal_arrivals, step=0.5,
        )
        covid_arrivals = st.number_input(
            "COVID requests (Poisson mean)", 0.0, 10.0,
            st.session_state.config.covid_arrivals, step=0.5,
        )
        normal_discharges = st.number_input(
            "Normal discharges (Poisson mean)", 0.0, 10.0,
            st.session_state.config.normal_discharges, step=0.5,
        )
        covid_discharges = st.number_input(
            "COVID discharges (Poisson mean)", 0.0, 10.0,
            st.session_state.config.covid_discharges, step=0.5,
        )
        discount_rate = st.slider(
            "Future reward discount", min_value=0.1, max_value=0.99,
            value=st.session_state.config.discount_rate, step=0.01,
        )
        simulation_days = st.number_input(
            "Simulation duration (days)", min_value=1, max_value=100,
            value=st.session_state.config.simulation_days, step=1,
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=999999,
            value=st.session_state.config.seed, step=1,
        )
        apply_settings = st.form_submit_button(
            "Apply settings & reset", type="primary", width="stretch"
        )

    step_delay = st.slider(
        "Time per simulated day",
        min_value=0.2,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="The slower setting makes arrivals and bed changes easier to follow.",
    )

    if apply_settings:
        try:
            new_config = SimulationConfig(
                total_beds=int(total_beds),
                normal_beds=int(normal_beds),
                normal_arrivals=float(normal_arrivals),
                covid_arrivals=float(covid_arrivals),
                normal_discharges=float(normal_discharges),
                covid_discharges=float(covid_discharges),
                discount_rate=float(discount_rate),
                simulation_days=int(simulation_days),
                seed=int(seed),
            )
            new_config.validate()
            with st.spinner("Optimizing the bed-transfer policy..."):
                new_solver = PolicySolver(new_config)
            st.session_state.config = new_config
            st.session_state.solver = new_solver
            st.session_state.simulator = HospitalSimulator(new_config)
            st.session_state.running = False
            st.session_state.last_tick = time.monotonic()
            st.rerun()
        except ValueError as error:
            st.error(str(error))

config: SimulationConfig = st.session_state.config
solver: PolicySolver = st.session_state.solver


@st.fragment(run_every=0.2)
def simulation_view() -> None:
    sim: HospitalSimulator = st.session_state.simulator
    policy_solver: PolicySolver = st.session_state.solver

    if (
        st.session_state.running
        and not sim.finished
        and time.monotonic() - st.session_state.last_tick >= step_delay
    ):
        sim.step(policy_solver)
        st.session_state.last_tick = time.monotonic()
        if sim.finished:
            st.session_state.running = False

    start_col, pause_col, step_col, reset_col = st.columns([1.1, 1.1, 1.0, 1.0])
    with start_col:
        if st.button(
            "▶  Start / resume",
            disabled=sim.finished,
            type="primary",
            width="stretch",
            key="start_simulation",
        ):
            st.session_state.running = True
            st.session_state.last_tick = time.monotonic()
    with pause_col:
        if st.button(
            "Ⅱ  Pause",
            disabled=not st.session_state.running,
            width="stretch",
            key="pause_simulation",
        ):
            st.session_state.running = False
    with step_col:
        if st.button(
            "Step one day",
            disabled=sim.finished,
            width="stretch",
            key="step_simulation",
        ):
            st.session_state.running = False
            sim.step(policy_solver)
            st.session_state.last_tick = time.monotonic()
    with reset_col:
        if st.button("Reset run", width="stretch", key="reset_simulation"):
            st.session_state.running = False
            st.session_state.simulator = HospitalSimulator(st.session_state.config)
            st.session_state.last_tick = time.monotonic()
            st.rerun()

    st.progress(sim.day / config.simulation_days, text=f"Day {sim.day} of {config.simulation_days}")

    normal_state = sim.state
    recommendation = policy_solver.action_for(normal_state)
    if recommendation > 0:
        recommendation_text = f"Move {recommendation} free bed(s) from Normal to COVID"
    elif recommendation < 0:
        recommendation_text = f"Move {abs(recommendation)} free bed(s) from COVID to Normal"
    else:
        recommendation_text = "Keep the current bed allocation"
    st.success(f"Policy recommendation: **{recommendation_text}**")

    expected_before = policy_solver.expected_reward(normal_state)
    _, next_state = next(
        (action, next_state)
        for action, next_state in policy_solver.actions(normal_state)
        if action == recommendation
    )
    expected_after = (
        policy_solver.expected_reward(next_state) - 5.0 * abs(recommendation)
    )
    score_before, score_after = st.columns(2)
    score_before.metric("Expected daily reward · no transfer", f"{expected_before:.1f}")
    score_after.metric("Expected daily reward · recommended", f"{expected_after:.1f}")

    normal_col, covid_col = st.columns(2)
    with normal_col:
        st.markdown('<div class="ward-card">', unsafe_allow_html=True)
        st.subheader("Normal ward")
        available_col, occupied_col = st.columns(2)
        available_col.metric("Available", f"{sim.normal_available} / {sim.normal_capacity}")
        occupied_col.metric(
            "Occupied", f"{sim.normal_capacity - sim.normal_available} / {sim.normal_capacity}"
        )
        st.progress(
            sim.normal_available / sim.normal_capacity
            if sim.normal_capacity
            else 0.0,
            text="Free-bed capacity",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with covid_col:
        st.markdown('<div class="ward-card">', unsafe_allow_html=True)
        st.subheader("COVID ward")
        available_col, occupied_col = st.columns(2)
        available_col.metric("Available", f"{sim.covid_available} / {sim.covid_capacity}")
        occupied_col.metric(
            "Occupied", f"{sim.covid_capacity - sim.covid_available} / {sim.covid_capacity}"
        )
        st.progress(
            sim.covid_available / sim.covid_capacity
            if sim.covid_capacity
            else 0.0,
            text="Free-bed capacity",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Live simulation")
    history = pd.DataFrame(sim.history)
    chart_col, results_col = st.columns([1.7, 1.0])
    with chart_col:
        chart_data = history.set_index("day")[
            ["normal_available", "covid_available", "normal_occupied", "covid_occupied"]
        ]
        st.line_chart(chart_data, height=300)
        st.caption("Available and occupied beds by ward over the simulated days.")
    with results_col:
        st.metric("Normal unmet requests", sim.total_unmet_normal)
        st.metric("COVID unmet requests", sim.total_unmet_covid)
        st.metric("Bed transfers made", sim.total_moved)
        st.metric("Cumulative reward", f"{sim.total_reward:.0f}")

    recent = history.iloc[1:].tail(1)
    if not recent.empty:
        latest = recent.iloc[0]
        st.info(
            f"Latest day: **{int(latest['normal_unmet'])}** unmet Normal request(s), "
            f"**{int(latest['covid_unmet'])}** unmet COVID request(s), "
            f"reward **{latest['reward']:.0f}**."
        )

    st.subheader("Policy heatmap")
    st.caption(
        "Each cell shows the transfer recommendation for that free-bed state. "
        "Positive values move beds Normal → COVID; negative values move COVID → Normal. "
        "The current state is outlined."
    )
    allocation_normal = sim.normal_capacity
    allocation_covid = sim.covid_capacity
    values = []
    row_labels = list(range(allocation_covid, -1, -1))
    for covid_available in row_labels:
        row = []
        for normal_available in range(allocation_normal + 1):
            cell_state = (allocation_normal, normal_available, covid_available)
            row.append(
                policy_solver.action_for(cell_state)
                if cell_state in policy_solver.policy
                else None
            )
        values.append(row)
    heatmap = pd.DataFrame(
        values,
        index=[f"COVID {value}" for value in row_labels],
        columns=[f"Normal {value}" for value in range(allocation_normal + 1)],
    )
    current_row = f"COVID {sim.covid_available}"
    current_col = f"Normal {sim.normal_available}"

    def highlight_current(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        if current_row in styles.index and current_col in styles.columns:
            styles.loc[current_row, current_col] = (
                "outline: 3px solid #1e78d2; outline-offset: -3px; font-weight: bold"
            )
        return styles

    st.dataframe(
        heatmap.style
        .background_gradient(cmap="RdYlGn", axis=None, vmin=-config.total_beds, vmax=config.total_beds)
        .format("{:+.0f}", na_rep="—")
        .apply(highlight_current, axis=None),
        width="stretch",
        height=min(440, 45 + 35 * len(heatmap.index)),
    )
    st.caption(
        f"Policy optimization converged in {policy_solver.iterations} value-iteration passes. "
        "The simulation uses a fixed random seed for repeatable comparisons."
    )

    st.subheader("Daily event log")
    if len(history) > 1:
        event_columns = [
            "day",
            "requests_normal",
            "requests_covid",
            "admissions_normal",
            "admissions_covid",
            "discharged_normal",
            "discharged_covid",
            "normal_unmet",
            "covid_unmet",
            "action",
            "reward",
        ]
        event_log = history.iloc[1:][event_columns].rename(
            columns={
                "day": "Day",
                "requests_normal": "Normal requests",
                "requests_covid": "COVID requests",
                "admissions_normal": "Normal admitted",
                "admissions_covid": "COVID admitted",
                "discharged_normal": "Normal discharged",
                "discharged_covid": "COVID discharged",
                "normal_unmet": "Normal unmet",
                "covid_unmet": "COVID unmet",
                "action": "Transfer (+ Normal → COVID)",
                "reward": "Reward",
            }
        )
        st.dataframe(event_log.iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("Start or step the simulation to see the arrivals and discharges here.")


simulation_view()
