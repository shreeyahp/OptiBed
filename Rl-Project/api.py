"""HTTP API for the OptiBed hospital allocation environment."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from rl_model import HospitalEnvironment

app = FastAPI(title="OptiBed API", version="1.0.0")
environment = HospitalEnvironment()


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allocate: StrictInt = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(
    seed: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, object]:
    global environment
    environment = HospitalEnvironment(seed=seed)
    return environment.state()


@app.post("/step")
def step(request: StepRequest) -> dict[str, object]:
    try:
        observation, reward, done, info = environment.step(request.allocate)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, object]:
    return environment.state()


@app.get("/grade")
def grade() -> dict[str, float | int]:
    return {
        "grade": environment.grade(),
        "raw_reward": environment.total_reward,
        "step": environment.step_number,
    }
