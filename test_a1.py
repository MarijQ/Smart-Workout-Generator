# SmartLift/test_a1.py

import os
# import sys # This line is not needed given the current execution method

# The sys.path manipulation is only needed if running as a script from the parent dir.
# If running with `python -m SmartLift.test_a1`, SmartLift is already on the path.
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(project_root, 'Smart-Workout-Generator-[GH]'))

from SmartLift.algorithm.core_data import load_exercises_from_json, load_workout_logs_from_json, select_random_exercises

# Define the path to the data directory relative to the SmartLift package
# When running with `python -m SmartLift.test_a1`, __file__ is relative to the
# directory where the module is found, which is SmartLift in this case.
data_dir = os.path.join(os.path.dirname(__file__), 'algorithm', 'data')
exercises_filepath = os.path.join(data_dir, 'exercises.json')
workout_logs_filepath = os.path.join(data_dir, 'workout_logs.json')


def run_a1_test():
    print("--- A1 Test: Core Data Structures & Basic Selection ---")

    # Test loading exercises
    try:
        exercises = load_exercises_from_json(exercises_filepath)
        print(f"Successfully loaded {len(exercises)} exercises from {exercises_filepath}")
        if exercises:
            print(f"Example exercise: {exercises[0].name} (Primary: {exercises[0].primary_muscles})")
    except FileNotFoundError:
        print(f"Error: exercises.json not found at {exercises_filepath}")
        return
    except Exception as e:
        print(f"Error loading exercises: {e}")
        return

    # Test loading workout logs (expected to be empty initially)
    try:
        workout_logs = load_workout_logs_from_json(workout_logs_filepath)
        print(f"Successfully loaded {len(workout_logs)} workout logs from {workout_logs_filepath} (expected 0 for A1).")
    except FileNotFoundError:
        print(f"Error: workout_logs.json not found at {workout_logs_filepath}")
        return
    except Exception as e:
        print(f"Error loading workout logs: {e}")
        return

    # Test random exercise selection
    num_to_select = 5
    if len(exercises) >= num_to_select:
        try:
            selected_exercises = select_random_exercises(exercises, num_to_select)
            print(f"\nSelected {num_to_select} random exercises:")
            for i, exercise in enumerate(selected_exercises):
                print(f"  {i+1}. {exercise.name}")
        except ValueError as e:
            print(f"Error selecting exercises: {e}")
        except Exception as e:
            print(f"Unexpected error during exercise selection: {e}")
    else:
        print(f"\nNot enough exercises to select {num_to_select}. Available: {len(exercises)}")

    print("\n--- A1 Test Complete ---")

if __name__ == "__main__":
    run_a1_test()
