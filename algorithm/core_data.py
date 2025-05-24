import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import random
from typing import List, Optional

@dataclass
class Exercise:
    name: str
    primary_muscles: List[str]
    secondary_muscles: List[str] = field(default_factory=list)
    exercise_type: str = "strength"  # e.g., "strength", "cardio", "stretching"
    equipment: List[str] = field(default_factory=list)
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

    def to_dict(self):
        d = asdict(self)
        d['log_date'] = self.log_date.isoformat()
        return d

@dataclass
class UserProfile:
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    preferences: Optional[dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return asdict(self)

# -- Load functions --
def load_exercises_from_json(filepath: str) -> List[Exercise]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [Exercise.from_dict(item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def load_workout_logs_from_json(filepath: str) -> List[WorkoutLog]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [WorkoutLog.from_dict(item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def load_user_profile_from_json(filepath: str) -> Optional[UserProfile]:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return UserProfile.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# -- Save functions --
def save_exercises_to_json(exercises: List[Exercise], filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump([asdict(e) for e in exercises], f, indent=2)

def save_workout_logs_to_json(logs: List[WorkoutLog], filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump([log.to_dict() for log in logs], f, indent=2)

def save_user_profile_to_json(profile: UserProfile, filepath: str) -> None:
    with open(filepath, 'w') as f:
        json.dump(profile.to_dict(), f, indent=2)

def select_random_exercises(exercises: List[Exercise], num_exercises: int) -> List[Exercise]:
    if num_exercises > len(exercises):
        raise ValueError("Number of exercises requested exceeds available exercises.")
    return random.sample(exercises, num_exercises)
