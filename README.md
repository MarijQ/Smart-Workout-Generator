# SmartLift: Intelligent Workout Planner

## Overview
**SmartLift** is a Python-based fitness application designed to create customised workout plans tailored to individual goals, preferences, and physical conditions. It focuses on balancing muscle fatigue, workout variation, and proportional muscle training through adaptive algorithms, ensuring efficient progress while minimising the risk of overtraining.

## Features
- **Dynamic Workout Planning**: Generates personalised lifting and cardio plans based on a user's current muscle fatigue and historical workout data.
- **Fatigue Management**: Takes into account muscle-specific fatigue and limits exercise selection accordingly.
- **Balanced Training**: Prioritises proportional muscle targeting while avoiding overtraining and ensuring variety.
- **Customisable Priorities**: Allows users to define workout preferences such as growth focus, symmetry improvements, or comfort level.
- **Cardio Integration**: Incorporates cardio sessions and tracks calorie burn for a comprehensive fitness plan.
- **Historical Optimisation**: Uses Exponentially Weighted Moving Average (EMA) to adapt based on past workout history.
- **Forecasting**: Simulates future workouts and training progress for advanced planning.

## Installation
1. Install prerequisites:
   ```bash
   pip install pandas numpy odfpy
   ```

2. Place input files (`exercises.ods`, `muscle_targets.ods`, `history.ods`) in the `/inputs` directory.

3. Run the application:
   ```bash
   python "SmartLift v6.py"
   ```

## Folder Structure
```
combined/
    requirements.txt        # Python dependencies
    SmartLift v6.py         # Core application
    inputs/
        exercises.ods       # List of exercises with metadata
        muscle_targets.ods  # Muscle targeting and priorities
        history.ods         # User's workout history
```

## Usage
1. Launch the program.
2. Select your options: track stats, view exercises, forecast workouts, etc.
3. Generate and execute your personalised workout plan. Optionally view suggestions for additional exercises.

## Customisation
Adjust the constants within the `Engine.constants()` method:
- **Duration Goals**: Set total daily/weekly workout duration.
- **Focus Parameters**: Modify growth, symmetry, variety, and comfort factors.
- **Muscle Fatigue Limits**: Fine-tune the EMA bias and fatigue thresholds.

## Future Enhancements
Planned features include:
- Tracking bodyweight exercises.
- Injury management.
- Time-based workout adjustments.
- Integration with calorie-burning goals.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests. Ensure the changes are well-documented and tested.
