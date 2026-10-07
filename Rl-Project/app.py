"""Automated, full-width OptiBed simulation and episode graphs."""

from __future__ import annotations

import copy
import time
from typing import Any

import pandas as pd
import streamlit as st

from rl_model import (
    DETERIORATION_PENALTY,
    EMERGENCY_BONUS,
    EMERGENCY_UNTREATED_PENALTY,
    HIGH_UNTREATED_PENALTY,
    HospitalEnvironment,
    QLearningAgent,
    TREATMENT_REWARD,
    WASTED_BED_PENALTY,
)


PLAYBACK_INTERVAL_SECONDS = 1.8

st.set_page_config(page_title="OptiBed", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 100%; padding: 1.4rem 2.2rem 2rem;}
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #dbe7f5; border-radius: 12px;
        padding: 0.7rem 1rem; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.04);
        color: #24364b;
    }
    [data-testid="stMetricLabel"] {color: #62758a;}
    [data-testid="stMetricValue"] {color: #24364b;}
    .bed-grid {
        display: grid; grid-template-columns: repeat(4, minmax(92px, 1fr));
        gap: 12px; margin: 1rem 0;
    }
    .bed {
        min-height: 94px; display: flex; flex-direction: column;
        justify-content: center; align-items: center; gap: 5px;
        border-radius: 11px; background: #eaf3ff; color: #1683f8;
        font: 600 0.78rem sans-serif;
    }
    .bed.free {background: #f0f3f7; color: #9aa9b9;}
    .bed svg {width: 50px; height: 42px;}
    .patient-card {
        min-height: 104px; border-radius: 11px; padding: 10px;
        text-align: center; margin: 0 0 10px; border: 1px solid #e5eaf0;
    }
    .patient-card.high {background: #fff0f0; border-color: #ffd1d1;}
    .patient-card.medium {background: #fff8e8; border-color: #ffe8af;}
    .patient-card.low {background: #eafaf3; border-color: #c5f1dd;}
    .patient-id {font-weight: 700; color: #24364b; margin-bottom: 8px;}
    .severity {
        display: inline-block; border-radius: 20px; padding: 3px 10px;
        font-size: 0.78rem; font-weight: 700;
    }
    .high .severity {background: #ffd7d7; color: #b42318;}
    .medium .severity {background: #ffebbd; color: #9a5b00;}
    .low .severity {background: #c8f2dd; color: #087443;}
    .emergency {margin-top: 7px; font-size: 0.75rem; color: #c62828; font-weight: 700;}
    .regular {margin-top: 7px; font-size: 0.75rem; color: #62758a;}
    </style>
    """,
    unsafe_allow_html=True,
)

BED_ICON = """
<svg viewBox="0 0 64 48" aria-hidden="true">
  <path d="M8 7v32M8 26h46a5 5 0 0 1 5 5v8M8 20h13a8 8 0 0 1 8 8v-2h25"
        fill="none" stroke="currentColor" stroke-width="5"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M14 39v5m39-5v5" fill="none" stroke="currentColor"
        stroke-width="4" stroke-linecap="round"/>
</svg>
"""


@st.cache_resource
def get_trained_agent() -> QLearningAgent:
    agent = QLearningAgent()
    agent.train()
    return agent


with st.spinner("Training the Q-learning agent..."):
    trained_agent = get_trained_agent()


def initialize_session_state() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = copy.deepcopy(trained_agent)
    if "environment" not in st.session_state:
        st.session_state.environment = HospitalEnvironment()
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "last_step_time" not in st.session_state:
        st.session_state.last_step_time = 0.0
    if "last_step" not in st.session_state:
        st.session_state.last_step = None


initialize_session_state()
agent: QLearningAgent = st.session_state.agent

st.title("OptiBed")
st.caption(
    "Watch a Q-learning agent make automatic bed-allocation decisions in a "
    "20-step hospital episode."
)
simulation_tab, graphs_tab, training_tab = st.tabs(
    ["Simulation", "Graphs", "Training"]
)


def q_table_dataframe(
    q_table: dict[tuple[int, ...], dict[int, float]],
) -> pd.DataFrame:
    """Format every stored state/action value as one row per state."""
    state_columns = [
        "Beds available",
        "High",
        "Medium",
        "Low",
        "Emergencies",
        "Steps remaining",
    ]
    actions = sorted(
        {action for action_values in q_table.values() for action in action_values}
    )
    action_columns = {action: f"Q(a={action})" for action in actions}
    rows = []
    for state, action_values in sorted(q_table.items()):
        row: dict[str, int | float] = dict(zip(state_columns, state))
        row["Actions tried"] = len(action_values)
        row.update({column: float("nan") for column in action_columns.values()})
        row.update(
            {
                action_columns[action]: value
                for action, value in action_values.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def execution_dataframe(metrics: list[dict[str, object]]) -> pd.DataFrame:
    playback = pd.DataFrame(metrics)
    if playback.empty:
        return playback
    playback["reward_breakdown"] = playback["reward_components"].map(
        lambda components: "; ".join(
            f"{name.replace('_', ' ').title()} {amount:+.2f}"
            for name, amount in components.items()
            if amount
        )
        or "No reward or penalty"
    )
    return playback.drop(columns=["reward", "reward_components"]).rename(
        columns={
            "action": "action (beds)",
            "gross_reward": "reward",
            "penalty": "penalty",
            "final_result": "final result",
            "reward_breakdown": "reward / penalty details",
        }
    )


def render_patient_cards(waiting: list[dict[str, Any]]) -> None:
    if not waiting:
        st.success("Everyone in the waiting list has been treated.")
        return
    for start in range(0, len(waiting), 4):
        patient_columns = st.columns(4)
        for column, patient in zip(patient_columns, waiting[start : start + 4]):
            severity = patient["severity"]
            emergency_label = (
                '<div class="emergency">Emergency</div>'
                if patient["emergency"]
                else '<div class="regular">Waiting</div>'
            )
            column.markdown(
                f'<div class="patient-card {severity}">'
                f'<div class="patient-id">Patient {patient["id"]}</div>'
                f'<span class="severity">{severity.title()}</span>'
                f"{emergency_label}</div>",
                unsafe_allow_html=True,
            )


def render_transition(transition: dict[str, Any]) -> None:
    agent: QLearningAgent = st.session_state.agent
    with st.container(border=True):
        before = transition["observation"]
        after = transition["next_observation"]
        st.subheader(f"RL decision · step {before['step']}")
        st.write(
            f"**State / observation** `sₜ` = "
            f"`{agent.state_key(before)}` "
            "(beds, high, medium, low, emergencies, steps remaining)"
        )
        st.write(
            f"**Action** `aₜ`: allocate **{transition['action']}** bed(s). "
            f"**Policy value** `Q(sₜ, aₜ)`: "
            f"**{transition['q_value']:+.2f} → "
            f"{transition['updated_q_value']:+.2f}** after online update "
            f"(TD error {transition['td_error']:+.2f})"
        )
        reward, reward_components = transition["reward"], transition["info"][
            "reward_components"
        ]
        gross_reward = sum(max(amount, 0.0) for amount in reward_components.values())
        penalty = -sum(min(amount, 0.0) for amount in reward_components.values())
        st.write(
            f"**Reward calculation:** {gross_reward:+.2f} reward "
            f"− {penalty:.2f} penalty = **{reward:+.2f} final result** "
            "(`rₜ`, used by Q-learning)"
        )
        nonzero_rewards = {
            name.replace("_", " ").title(): amount
            for name, amount in reward_components.items()
            if amount
        }
        if nonzero_rewards:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Reward component": name, "Value": f"{amount:+.2f}"}
                        for name, amount in nonzero_rewards.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No reward or penalty was recorded for this step.")

        treated = transition["info"]["treated_patients"]
        if treated:
            treated_summary = ", ".join(
                f"#{patient['id']} {patient['severity']}"
                + (" emergency" if patient["emergency"] else "")
                for patient in treated
            )
            st.write(f"**Patients treated:** {treated_summary}")
        else:
            st.write("**Patients treated:** none")
        st.write(
            f"**Environment transition** `sₜ → sₜ₊₁`: "
            f"`{agent.state_key(after)}`"
        )
        st.caption(
            f"Next arrivals: {transition['info']['arrivals']} "
            f"({transition['info']['emergency_arrivals']} emergency). "
            "The learned Q-table was trained with the Bellman update "
            "`Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,a′) − Q(s,a)]`."
        )


@st.fragment(
    run_every=0.2 if st.session_state.simulation_running else None
)
def render_simulation() -> None:
    initialize_session_state()
    agent: QLearningAgent = st.session_state.agent
    environment: HospitalEnvironment = st.session_state.environment
    state = environment.state()
    controls = st.columns([1, 1, 1, 3])
    if controls[0].button(
        "Start",
        type="primary",
        disabled=state["done"] or st.session_state.simulation_running,
        width="stretch",
        key="start_simulation",
    ):
        st.session_state.simulation_running = True
        st.session_state.last_step_time = time.monotonic()
        st.rerun(scope="app")
    if controls[1].button(
        "Stop",
        disabled=not st.session_state.simulation_running,
        width="stretch",
        key="stop_simulation",
    ):
        st.session_state.simulation_running = False
        st.rerun(scope="app")
    if controls[2].button(
        "Reset",
        width="stretch",
        key="reset_simulation",
    ):
        st.session_state.environment = HospitalEnvironment()
        st.session_state.agent = copy.deepcopy(trained_agent)
        agent = st.session_state.agent
        st.session_state.simulation_running = False
        st.session_state.last_step_time = 0.0
        st.session_state.last_step = None
        st.rerun(scope="app")
    if state["done"]:
        controls[3].caption("Status: episode complete")
    elif st.session_state.simulation_running:
        controls[3].caption("Status: running automatically")
    else:
        controls[3].caption("Status: paused")

    if st.session_state.simulation_running and not state["done"]:
        now = time.monotonic()
        if now - st.session_state.last_step_time >= PLAYBACK_INTERVAL_SECONDS:
            action = agent.choose_action(state)
            q_value = agent.q_value(state, action)
            next_state, reward, done, info = environment.step(action)
            updated_q_value, td_error = agent.learn_from_transition(
                state,
                action,
                reward,
                next_state,
                reward_components=info["reward_components"],
            )
            st.session_state.last_step = {
                "observation": state,
                "action": action,
                "q_value": q_value,
                "updated_q_value": updated_q_value,
                "td_error": td_error,
                "next_observation": next_state,
                "reward": reward,
                "info": info,
            }
            st.session_state.last_step_time = now
            if done:
                st.session_state.simulation_running = False
                st.rerun(scope="app")
            state = next_state

    waiting = state["patients"]
    occupied = len(environment.occupied)
    beds_total = environment.config.beds
    metrics = st.columns(4)
    metrics[0].metric("Beds available", f"{state['beds']} / {beds_total}")
    metrics[1].metric("Patients waiting", len(waiting))
    metrics[2].metric("Total reward", f"{environment.total_reward:+.1f}")
    metrics[3].metric("Episode grade", f"{environment.grade():.3f}")
    st.progress(
        state["step"] / state["max_steps"],
        text=f"Episode progress · step {state['step']} of {state['max_steps']}",
    )

    hospital_column, queue_column = st.columns([1, 2.4], gap="large")
    with hospital_column:
        with st.container(border=True):
            st.subheader("Hospital beds")
            st.caption(
                f"{occupied} of {beds_total} beds are in use. "
                f"{state['beds']} are ready for patients."
            )
            bed_slots = []
            for bed_number in range(beds_total):
                in_use = bed_number < occupied
                style = "bed" if in_use else "bed free"
                label = "In use" if in_use else "Available"
                bed_slots.append(
                    f'<div class="{style}">{BED_ICON}<span>'
                    f"Bed {bed_number + 1:02d} · {label}</span></div>"
                )
            st.markdown(
                f'<div class="bed-grid">{"".join(bed_slots)}</div>',
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.subheader("Agent")
            st.write("**Algorithm:** tabular Q-learning")
            st.write(f"**Training episodes:** {agent.training_episodes:,}")
            st.write(f"**Learned states:** {len(agent.q_table):,}")
            recent_return = sum(agent.episode_returns[-100:]) / min(
                100, len(agent.episode_returns)
            )
            st.write(f"**Mean return (last 100 training episodes):** {recent_return:+.1f}")
            st.write(f"**Playback:** one decision every {PLAYBACK_INTERVAL_SECONDS:.1f}s")
            st.caption(
                "The agent was trained before playback. Start runs its learned "
                "policy; Stop pauses it; Reset starts a fresh episode."
            )

    with queue_column:
        with st.container(border=True):
            st.subheader(f"Patients waiting · {len(waiting)}")
            st.caption("Emergency patients and severity are identified on each card.")
            render_patient_cards(waiting)
        if st.session_state.last_step is not None:
            render_transition(st.session_state.last_step)
        elif state["step"] == 0:
            st.info("Press Start to watch the trained agent make its first decision.")
        if state["done"]:
            st.success("Episode complete. Press Reset to run another episode.")

    playback = execution_dataframe(agent.execution_metrics)
    with st.container(border=True):
        st.subheader("Reward by simulation step")
        st.caption(
            "Reward is shown before deductions; penalty is the positive amount "
            "subtracted. Final result is the net reward used by Q-learning."
        )
        if playback.empty:
            st.info("Step rewards and penalties will appear after the first action.")
        else:
            st.dataframe(
                playback[
                    [
                        "step",
                        "action (beds)",
                        "reward",
                        "penalty",
                        "final result",
                        "reward / penalty details",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

    with st.container(border=True):
        st.subheader("Reward mechanism")
        st.caption(
            "These fixed rewards and penalties are applied by the simulation "
            "on each step. An untreated emergency patient also receives the "
            "high-severity penalty."
        )
        reward_rules = pd.DataFrame(
            [
                {
                    "Event": "Treat low / medium / high severity",
                    "Reward": (
                        f"{TREATMENT_REWARD['low']:+g} / "
                        f"{TREATMENT_REWARD['medium']:+g} / "
                        f"{TREATMENT_REWARD['high']:+g}"
                    ),
                },
                {"Event": "Treat an emergency patient", "Reward": f"{EMERGENCY_BONUS:+g} bonus"},
                {
                    "Event": "Leave a high-severity patient untreated for a step",
                    "Reward": f"{HIGH_UNTREATED_PENALTY:+g}",
                },
                {
                    "Event": "Leave an emergency patient untreated for a step",
                    "Reward": f"{EMERGENCY_UNTREATED_PENALTY:+g}",
                },
                {
                    "Event": "Leave an available bed unused while people are waiting",
                    "Reward": f"{WASTED_BED_PENALTY:+g} per bed",
                },
                {
                    "Event": "Patient deterioration",
                    "Reward": f"{DETERIORATION_PENALTY:+g} per severity level",
                },
            ]
        )
        st.dataframe(reward_rules, hide_index=True, width="stretch")


@st.fragment(
    run_every=1.0 if st.session_state.simulation_running else None
)
def render_graphs() -> None:
    environment: HospitalEnvironment = st.session_state.environment
    if environment.step_number == 0:
        st.info("Graphs will appear here after the first simulation step.")
        return

    history = pd.DataFrame(environment.history).set_index("step")
    queue_column, bed_column = st.columns(2, gap="large")
    with queue_column:
        with st.container(border=True):
            st.subheader("Who is waiting?")
            st.caption("Waiting patients at each severity level after every step.")
            queue_trend = history[
                ["waiting_high", "waiting_medium", "waiting_low"]
            ].rename(
                columns={
                    "waiting_high": "High severity",
                    "waiting_medium": "Medium severity",
                    "waiting_low": "Low severity",
                }
            )
            st.line_chart(
                queue_trend,
                color=["#ef5350", "#f5b82e", "#24b47e"],
                x_label="Simulation step",
                y_label="Patients waiting",
            )
    with bed_column:
        with st.container(border=True):
            st.subheader("Bed use")
            st.caption("Beds in use compared with available beds.")
            st.line_chart(
                history[["occupied_beds", "available_beds"]],
                color=["#3388ee", "#9aa9b9"],
                x_label="Simulation step",
                y_label="Number of beds",
            )
    with st.container(border=True):
        st.subheader("Reward over time")
        st.caption("Step reward is each decision's score; episode total accumulates it.")
        reward_trend = history[["reward", "cumulative_reward"]].rename(
            columns={"reward": "This step", "cumulative_reward": "Episode total"}
        )
        st.line_chart(
            reward_trend,
            color=["#f5a623", "#17a673"],
            x_label="Simulation step",
            y_label="Reward",
        )
        st.metric("Current episode grade", f"{environment.grade():.3f}")


@st.fragment(run_every=0.5 if st.session_state.simulation_running else None)
def render_training() -> None:
    training = pd.DataFrame(agent.training_metrics)
    latest = agent.training_metrics[-1]
    st.subheader("Q-learning training")
    st.caption(
        "Training uses simulated episodes. The trained Q-table below is a "
        "snapshot taken immediately after training; playback starts from a "
        "separate per-session copy and updates that copy after every action."
    )

    metrics = st.columns(5)
    metrics[0].metric("Episodes completed", f"{len(agent.episode_returns):,}")
    metrics[1].metric("Environment transitions", f"{latest['total_transitions']:,}")
    metrics[2].metric("Learned states", f"{latest['learned_states']:,}")
    metrics[3].metric("State-action values", f"{latest['state_action_values']:,}")
    metrics[4].metric(
        "Actions tried / state",
        f"{latest['mean_actions_per_state']:.2f}",
    )

    settings = st.columns(5)
    settings[0].metric("Learning rate (α)", f"{agent.learning_rate:.2f}")
    settings[1].metric("Discount factor (γ)", f"{agent.discount_factor:.2f}")
    settings[2].metric("Initial ε", f"{training.iloc[0]['epsilon']:.3f}")
    settings[3].metric("Final ε", f"{latest['epsilon']:.3f}")
    settings[4].metric(
        "Exploration in final episode",
        f"{latest['exploration_rate']:.1%} "
        f"({latest['exploration_decisions']} decisions)",
    )
    recent_metrics = training.iloc[-100:]
    st.caption(
        "Last 100 training episodes: mean return "
        f"{recent_metrics['episode_return'].mean():+.2f}, "
        f"sample standard deviation "
        f"{recent_metrics['episode_return'].std(ddof=1):.2f}. "
        f"Training seed: {agent.seed}."
    )

    trained_table = q_table_dataframe(agent.trained_q_table)
    with st.expander(
        f"Q-table after training · {len(trained_table):,} states "
        f"· {latest['state_action_values']:,} state-action values"
    ):
        st.caption(
            "This frozen snapshot was captured when training completed. Blank "
            "cells mean that state/action pair was not tried during training."
        )
        st.dataframe(
            trained_table.round(3), hide_index=True, width="stretch", height=420
        )
        st.download_button(
            "Download trained Q-table as CSV",
            trained_table.to_csv(index=False),
            file_name="optibed_trained_q_table.csv",
            mime="text/csv",
            key="download_trained_q_table",
        )

    playback = pd.DataFrame(agent.execution_metrics)
    if playback.empty:
        st.info("Start the simulation to execute actions and update the Q-table.")
    else:
        playback_states = playback.drop_duplicates(subset=["state"])
        st.subheader("After executing simulation steps")
        playback_columns = st.columns(4)
        playback_columns[0].metric("Executed steps", len(playback))
        playback_columns[1].metric(
            "States visited", len(playback_states)
        )
        playback_columns[2].metric(
            "State-action values", sum(len(values) for values in agent.q_table.values())
        )
        playback_columns[3].metric(
            "Last Q update",
            f"{playback.iloc[-1]['q_before']:.3f} → "
            f"{playback.iloc[-1]['q_after']:.3f}",
        )
        with st.expander("Executed step Q-learning updates"):
            playback_table = execution_dataframe(agent.execution_metrics)
            st.caption(
                "Reward is shown before deductions; penalty is the amount "
                "subtracted; final result is the net reward used by Q-learning."
            )
            st.dataframe(playback_table, hide_index=True, width="stretch")
            st.download_button(
                "Download executed-step updates as CSV",
                playback_table.to_csv(index=False),
                file_name="optibed_execution_updates.csv",
                mime="text/csv",
                key="download_execution_metrics",
            )
    with st.expander("Per-episode training metrics"):
        st.dataframe(training, hide_index=True, width="stretch", height=420)
        st.download_button(
            "Download training metrics as CSV",
            training.to_csv(index=False),
            file_name="optibed_training_metrics.csv",
            mime="text/csv",
            key="download_training_metrics",
        )


with simulation_tab:
    render_simulation()

with graphs_tab:
    render_graphs()

with training_tab:
    render_training()
