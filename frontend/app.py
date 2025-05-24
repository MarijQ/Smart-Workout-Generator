from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

sample_workout_data = [
    {"id": "ex1", "name": "Bench Press", "target_sets": 3, "target_reps": 8, "target_weight": 60.0},
    {"id": "ex2", "name": "Squat", "target_sets": 3, "target_reps": 5, "target_weight": 80.0},
    {"id": "ex3", "name": "Overhead Press", "target_sets": 3, "target_reps": 10, "target_weight": 40.0},
]

@app.route('/')
def workout_overview():
    return render_template('workout_overview.html', exercises=sample_workout_data)

@app.route('/exercise/<exercise_id>')
def exercise_detail(exercise_id):
    exercise = next((ex for ex in sample_workout_data if ex["id"] == exercise_id), None)
    if not exercise:
        return "Exercise not found", 404
    return render_template('exercise_detail.html', exercise=exercise)

@app.route('/log_set/<exercise_id>', methods=['POST'])
def log_set(exercise_id):
    actual_reps = request.form.get('actual_reps')
    actual_weight = request.form.get('actual_weight')
    print(f"LOG: Exercise: {exercise_id}, Reps: {actual_reps}, Weight: {actual_weight}")
    return redirect(url_for('exercise_detail', exercise_id=exercise_id))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
