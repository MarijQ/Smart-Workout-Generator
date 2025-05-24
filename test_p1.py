import os
from algorithm.core_data import (
    Exercise, WorkoutLog, UserProfile,
    load_exercises_from_json, save_exercises_to_json,
    load_workout_logs_from_json, save_workout_logs_to_json,
    load_user_profile_from_json, save_user_profile_to_json,
)
from datetime import datetime

data_dir = os.path.join(os.path.dirname(__file__), 'algorithm', 'data')
ex_path = os.path.join(data_dir, 'exercises.test.json')
wl_path = os.path.join(data_dir, 'workout_logs.test.json')
up_path = os.path.join(data_dir, 'user_profile.test.json')

def test_exercises_save_load():
    ex = [Exercise(name='Push-up', primary_muscles=['Pecs'], base_fatigue_score=1.1)]
    save_exercises_to_json(ex, ex_path)
    loaded = load_exercises_from_json(ex_path)
    assert loaded and loaded[0].name == 'Push-up'

def test_workout_logs_save_load():
    wl = [WorkoutLog(exercise_name='Push-up', sets=3, reps=10, weight=0.0, log_date=datetime.now())]
    save_workout_logs_to_json(wl, wl_path)
    loaded = load_workout_logs_from_json(wl_path)
    assert loaded and loaded[0].exercise_name == 'Push-up'

def test_user_profile_save_load():
    up = UserProfile(name='Alice', age=30, gender='F', preferences={'goal': 'hypertrophy'})
    save_user_profile_to_json(up, up_path)
    loaded = load_user_profile_from_json(up_path)
    assert loaded and loaded.name == 'Alice'

def test_missing_files():
    assert load_exercises_from_json('doesnotexist.json') == []
    assert load_workout_logs_from_json('doesnotexist.json') == []
    assert load_user_profile_from_json('doesnotexist.json') is None

def main():
    test_exercises_save_load()
    test_workout_logs_save_load()
    test_user_profile_save_load()
    test_missing_files()
    print('P1 tests passed.')

if __name__ == '__main__':
    main()
