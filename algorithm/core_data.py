# algorithm/core_data.py

import json
from dataclasses import dataclass, field
from datetime import datetime
import random

@dataclass
class Exercise:
    name: str
    primary_muscles: list[str]
    secondary_muscles: list[str] = field(default_factory=list)
    exercise_type: str = "strength"  # e.g., "strength", "cardio", "stretching"
    equipment: list[str] = field(default_factory=list)
    base_fatigue_score: float = 1.0

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

@dataclass
class WorkoutLog:
    exercise_name: str
    sets: int
    reps: int
    weight: float
    log_date: datetime

    @classmethod
    def from_dict(cls, data: dict):
        data['log_date'] = datetime.fromisoformat(data['log_date'])
        return cls(**data)

def load_exercises_from_json(filepath: str) -> list[Exercise]:
    """Loads a list of Exercise objects from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [Exercise.from_dict(item) for item in data]

def load_workout_logs_from_json(filepath: str) -> list[WorkoutLog]:
    """Loads a list of WorkoutLog objects from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [WorkoutLog.from_dict(item) for item in data]

def select_random_exercises(exercises: list[Exercise], num_exercises: int) -> list[Exercise]:
    """Selects a fixed number of random exercises from a given list."""
    if num_exercises > len(exercises):
        raise ValueError("Number of exercises requested exceeds available exercises.")
    return random.sample(exercises, num_exercises)

