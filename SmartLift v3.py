# todo
# muscular fatigue (not flat 2.0, reduced custom to muscle based on higher frequency 50% EMA muscle usage)
# Injury
# Bodyweight workouts
# Time per workout slider instead of fatigue one

# imports
import pandas as pd
import numpy as np
import datetime as dt
import csv
import warnings
import re

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 400)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns', 20)


class Engine():
    def __init__(self):
        self.constants()
        self.set_up()
        self.TODAY = dt.date.today() + dt.timedelta(days=0)
        # self.output()

    def constants(self):
        # Set up lists/arrays/constants
        self.BASE_SETS_PER_WEEK = 6  # This can be adjusted between 4 and 12
        self.PIPELINE = []
        self.MUSCLE_FATIGUE_LIMIT = 1.5
        # Select user priorities (score of X means that top exercise is 2^X times more likely to be picked)
        # -1 Strongly avoid --- -0.5 Avoid --- 0 Don't care --- 0.5 Favour --- 1 Strongly favour
        self.COMPOUNDNESS_WEIGHT = 1     # STRENGTH - prioritise highly compound full-body exercises
        self.BALANCE_WEIGHT = 1          # PHYSIQUE - prioritise hitting target muscles to grow
        self.UNILATERAL_WEIGHT = 0.5       # BALANCE - prioritise single-limb exercises fixing asymmetry
        self.FITNESS_WEIGHT = 0          # FITNESS - prioritise cardio to improve fitness
        self.FLEXIBILITY_WEIGHT = 0      # FLEXIBILITY - prioritise stretches to improve flexibility
        self.FRESHNESS_WEIGHT = 0.5        # NOVELTY - try to mix up new exercises to keep things interesting
        self.DURATION_WEIGHT = 0          # DURATION - prioritise exercises that each take longer
        self.TARGET_DURATION = 40         # Total exercise duration in minutes
        self.IS_HOME = False               # Toggle for home exercises

    def set_up(self):
        self.history = pd.read_csv("inputs/history.csv")
        self.exercises_data = pd.read_csv("inputs/exercises.csv")
        self.muscle_targets_data = pd.read_csv("inputs/muscle_targets.csv")
        # set up dataframes
        self.exercise_score = self.exercises_data.loc[self.exercises_data["Active"] == 1, ["ID", "Exercise", "Type", "Unilateral", "Fitness", "Flexibility", "Compoundness", "Duration", "Home"]]
        if self.IS_HOME:
            self.exercise_score = self.exercise_score[self.exercise_score["Home"] == 1]
        self.exercise_score[["Freshness"]] = 0
        self.muscle_score = pd.DataFrame(list(self.exercises_data.columns[10:]), columns=["Muscle"])
        self.muscle_score[["Sets_per_week"]] = 0.0  # Initialize as float
        self.muscle_score[["Workout_fatigue"]] = 0.0  # Initialize as float
        self.muscle_score[["Temp_fatigue"]] = 0.0  # Initialize as float
        self.muscle_score[["EMA_fatigue"]] = 0.0  # Initialize as float
        self.PIPELINE = []
        # Convert types (in case they aren't already float)
        self.muscle_score["Sets_per_week"] = self.muscle_score["Sets_per_week"].astype(float)
        self.muscle_score["Workout_fatigue"] = self.muscle_score["Workout_fatigue"].astype(float)
        self.muscle_score["Temp_fatigue"] = self.muscle_score["Temp_fatigue"].astype(float)
        self.muscle_score["EMA_fatigue"] = self.muscle_score["EMA_fatigue"].astype(float)

    def filter_out_today_entries(self):
        """Remove entries from history that are already logged for today or future dates."""
        # Convert 'Date' to datetime objects and then to date objects with new format
        self.history['Date'] = self.history['Date'].apply(lambda x: dt.datetime.strptime(x, '%d/%m/%y').date())
        # Filter out entries that are today or in the future
        self.history = self.history[self.history['Date'] < self.TODAY]
        # Convert 'Date' back to string format with new format
        self.history['Date'] = self.history['Date'].apply(lambda x: x.strftime('%d/%m/%y'))
        # Save the updated history to CSV
        self.history.to_csv("inputs/history.csv", index=False)

    def decode(self, exercise_str):
        match = re.match(r"(\d+)(e?)", exercise_str)
        if match:
            exercise_id = int(match.group(1))
            is_easy = (match.group(2) == 'e')

            # Find the exercise name corresponding to the exercise ID
            exercise_name = self.exercises_data.loc[self.exercises_data['ID'] == exercise_id, 'Exercise']

            if not exercise_name.empty:
                return exercise_name.iloc[0], is_easy
            else:
                raise ValueError("Exercise ID not found in exercises_data.")
        else:
            raise ValueError("Invalid exercise string format.")

    def freshness_calc(self):
        # FRESHNESS: calculate exercise scores based on inverse days difference
        for index, row in self.exercise_score.iterrows():  # Loop through all exercises
            # find history instances with that exercise present
            mask = np.column_stack([self.history[col].apply(lambda x: re.sub(r'e', '', str(x)) == str(row["ID"])) for col in self.history.columns[1:11]])
            selected_history = self.history.loc[mask.any(axis=1)]
            # sum contribution from each past instance
            score = 0
            for index2, row2 in selected_history.iterrows():  # Loop through history records
                date_diff = (self.TODAY - dt.datetime.strptime(row2["Date"], '%d/%m/%y').date()).days
                score += 0.3 * 0.9 ** (date_diff - 1)
            # Rescale the freshness score to be between 1 and 10
            scaled_freshness = max(0, min(10, 10 * (1 - score)))
            # print(score, scaled_freshness)
            self.exercise_score.loc[index, ["Freshness"]] = round(scaled_freshness, 2)

    def muscle_calc(self):
        # BALANCE: calculate muscle sets per week EMA
        for index, row in self.history.iterrows():  # loop through history records
            for i in range(10):  # loop through exercises completed
                exercise_entry = row.iloc[i + 1]
                match = re.match(r"(\d+)(e?)", str(exercise_entry))
                if not match:
                    continue

                exercise_ID, is_easy = int(match.group(1)), match.group(2)
                is_easy = (is_easy == "e")
                for index2, row2 in self.muscle_score.iterrows():  # add EMA contributions for each muscle
                    score = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                    try:
                        int(score)
                    except ValueError:
                        continue
                    exercise_type = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[0]
                    if exercise_type == "Cardio":
                        score = float(score) / 2
                    else:
                        score = float(score)

                    # Adjust score for easy exercises before calculating EMA_score_fatigue
                    fatigue_score = round(score * (1 - 0.2 * is_easy), 2)
                    balance_score = round(score * (1 - 0.2 * is_easy), 2)
                    # print(score, fatigue_score, self.decode(exercise_entry), row2["Muscle"])

                    date_diff = (self.TODAY - dt.datetime.strptime(row["Date"], '%d/%m/%y').date()).days
                    EMA_score_balance = 0.1 * balance_score * 0.9 ** date_diff
                    EMA_score_fatigue = 0.5 * fatigue_score * 0.5 ** (date_diff - 1) if exercise_type != "Cardio" else 0

                    self.muscle_score.loc[index2, ["Sets_per_week"]] += EMA_score_balance * 3 * 7
                    self.muscle_score.loc[index2, ["EMA_fatigue"]] += max(0, EMA_score_fatigue)  # Ensure non-negative

    def muscle_target_calc(self):
        # Calculate muscle EMA target directly using BASE_SETS_PER_WEEK and Priority
        self.muscle_score["Target"] = self.muscle_targets_data["Priority"] * self.BASE_SETS_PER_WEEK
        # Calculate balance as the difference between Target and actual Sets_per_week
        self.muscle_score["Balance"] = self.muscle_score.apply(lambda row: max(0, row["Target"] - row["Sets_per_week"]), axis=1)

        # Debug statement to print Sets_per_week, Target, and Balance
        # print("Muscle Score Debug:")
        # print(self.muscle_score[["Muscle", "Sets_per_week", "Target", "Balance"]])

    def new_balance_calc(self):
        ## calculate projected balance scores for each exercise
        for index, row in self.exercise_score.iterrows():  # loop through all (active) exercises
            self.muscle_score[["New_balance"]] = self.muscle_score[["Balance"]]
            exercise_ID = row["ID"]
            for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
                # if index2 != 14:
                #     continue
                score = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                if self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[0] == "Cardio":
                    score = float(score) / 2
                else:
                    score = float(score)
                new_balance = max(0, row2["Target"] - (0.9 * row2['Sets_per_week'] + 0.1 * score * 3 * 7))
                # print(self.decode(str(exercise_ID)), row2['Muscle'], round(row2['Balance'], 2), round(score, 2), round(new_balance, 2))
                self.muscle_score.loc[index2, ["New_balance"]] = new_balance
            RMS = (self.muscle_score["New_balance"] ** 2).sum()
            self.exercise_score.loc[index, ["Balance"]] = RMS
            # print(self.decode(str(exercise_ID)), RMS)

    def score_range_fix(self, best, worst):
        if best == worst:
            return 1
        else:
            return best - worst

    def overall_scores_calc(self):
        ## calculate overall scores for each exercise
        self.exercise_score[["Score"]] = 0
        freshness_best = self.exercise_score["Freshness"].max()
        freshness_worst = self.exercise_score["Freshness"].min()
        balance_best = self.exercise_score["Balance"].min()
        balance_worst = self.exercise_score["Balance"].max()
        compoundness_best = self.exercise_score["Compoundness"].max()
        compoundness_worst = self.exercise_score["Compoundness"].min()
        unilateral_best = self.exercise_score["Unilateral"].max()  # Best score for Unilateral
        unilateral_worst = self.exercise_score["Unilateral"].min()  # Worst score for Unilateral
        fitness_best = self.exercise_score["Fitness"].max()  # Best score for Fitness
        fitness_worst = self.exercise_score["Fitness"].min()  # Worst score for Fitness
        flexibility_best = self.exercise_score["Flexibility"].max()  # Best score for Flexibility
        flexibility_worst = self.exercise_score["Flexibility"].min()  # Worst score for Flexibility
        duration_best = self.exercise_score["Duration"].max()  # New best score for Duration
        duration_worst = self.exercise_score["Duration"].min()  # New worst score for Duration

        fn3 = lambda row: max(0,
                              (1 + (row["Freshness"] - freshness_worst) / self.score_range_fix(freshness_best, freshness_worst))
                              ** self.FRESHNESS_WEIGHT *
                              (1 + (row["Balance"] - balance_worst) / self.score_range_fix(balance_best, balance_worst))
                              ** self.BALANCE_WEIGHT *
                              (1 + (row["Compoundness"] - compoundness_worst) / self.score_range_fix(compoundness_best, compoundness_worst))
                              ** self.COMPOUNDNESS_WEIGHT *
                              (1 + (row["Unilateral"] - unilateral_worst) / self.score_range_fix(unilateral_best, unilateral_worst))
                              ** self.UNILATERAL_WEIGHT *
                              (1 + (row["Fitness"] - fitness_worst) / self.score_range_fix(fitness_best, fitness_worst))
                              ** self.FITNESS_WEIGHT *
                              (1 + (row["Flexibility"] - flexibility_worst) / self.score_range_fix(flexibility_best, flexibility_worst))
                              ** self.FLEXIBILITY_WEIGHT *
                              (1 + (row["Duration"] - duration_worst) / self.score_range_fix(duration_best, duration_worst))
                              ** self.DURATION_WEIGHT)  # New scoring component for Duration
        # print(balance_worst,balance_best)
        self.exercise_score["Score"] = self.exercise_score.apply(fn3, axis=1)
        # print(self.exercise_score, "\n", self.muscle_score)

    def generate_workout(self, date):
        # set up and read files
        self.constants()
        self.set_up()
        self.filter_out_today_entries()
        # refresh score baselines from history
        self.freshness_calc()
        self.muscle_calc()
        self.muscle_target_calc()
        # calculate projected scores for exercises
        self.new_balance_calc()
        self.overall_scores_calc()
        # print(self.exercise_score)
        # reset workout fatigue values
        self.muscle_score[["Workout_fatigue"]] = 0
        # calculate duration cap and initialise self.remaining_duration, copy self.exercise_score
        self.remaining_duration = self.TARGET_DURATION
        self.exercise_score_init = self.exercise_score.copy()
        # print start matter
        print("Date: ", date)
        print("Target duration: ", self.TARGET_DURATION)
        print("Old total balance:", round((self.muscle_score["Balance"] ** 2).sum(), 0))
        print("Old sets per week:", round(self.muscle_score["Sets_per_week"].mean(), 1))

        # Generate as many exercises as possible until the duration cap is reached
        while self.remaining_duration >= 7:  # Assuming minimum exercise duration is 5 minutes
            if self.exercise_score.empty:
                print("No more suitable exercises!")
                break
            # Select the top exercise
            selected = self.select_top_exercise()
            if not selected:
                break

        # print(self.muscle_score)
        print("New total balance:", round((self.muscle_score["Balance"] ** 2).sum(), 0))
        self.muscle_score["Sets_per_week"]*=0.9
        print("New sets per week:", round(self.muscle_score["Sets_per_week"].mean(), 1))

        # Add the new row to the history DataFrame
        new_row = pd.DataFrame([[date.strftime('%d/%m/%y')] + self.PIPELINE], columns=self.history.columns[:len([date] + self.PIPELINE)])
        self.history = pd.concat([self.history, new_row], ignore_index=True)

        # Write the updated history to the file
        self.history.to_csv("inputs/history.csv", index=False)

        print("Extras:")
        self.select_top_extras(5)
        print("------------")

        # Update history with names and save
        self.update_history_with_names()

        # Final balance calculation after all exercises have been selected
        self.new_balance_calc()
        self.overall_scores_calc()
        # print(self.muscle_score)

    def select_top_extras(self, count):
        # Filter out rows where ID is in the pipeline
        filtered_exercises = self.exercise_score_init[~self.exercise_score_init['ID'].isin([int(x) for x in self.PIPELINE])]

        # Select the top N exercises based on overall score without considering fatigue
        top_exercises = filtered_exercises.sort_values(by="Score", ascending=False).head(count)

        for index, row in top_exercises.iterrows():
            print("Extra option:", row["ID"], row["Exercise"], "(", row["Duration"], "mins )")

    def select_top_exercise(self):
        # Filter for duration limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            if row["Duration"] > self.remaining_duration:
                self.exercise_score.drop(index, inplace=True)

        # Filter for muscle fatigue limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            self.muscle_score[["Temp_fatigue"]] = 0
            for index2, row2 in self.muscle_score.iterrows():  # add fatigue for each muscle
                score = self.exercises_data.loc[self.exercises_data["ID"] == row["ID"], row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                if self.exercises_data.loc[self.exercises_data["ID"] == row["ID"], "Type"].iloc[0] == "Cardio":
                    score = float(score) / 2
                else:
                    score = float(score)
                self.muscle_score.loc[index2, ["Temp_fatigue"]] += float(score)
                expected_fatigue = self.muscle_score.loc[index2, ["Workout_fatigue"]].iloc[0] + self.muscle_score.loc[index2, ["Temp_fatigue"]].iloc[0]
                if expected_fatigue > self.MUSCLE_FATIGUE_LIMIT - row2["EMA_fatigue"]:
                    self.exercise_score.drop(index, inplace=True)
                    break

        # Find top exercise regardless of type
        if self.exercise_score.empty:
            print("No more suitable exercises!")
            return False

        row = self.exercise_score.sort_values(by="Score", ascending=False).head(1)
        try:
            exercise_index = row.index.tolist()[0]
        except IndexError:
            print("No more suitable exercises!")
            return False

        exercise_ID = row["ID"].iloc[0]
        duration = row["Duration"].iloc[0]

        # Update muscle balance + fatigue
        self.muscle_score[["Temp_fatigue"]] = 0
        self.muscle_score[["New_balance"]] = self.muscle_score[["Balance"]]
        for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
            score = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
            try:
                int(score)
            except ValueError:
                continue
            if self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[0] == "Cardio":
                score = float(score) / 2
            else:
                score = float(score)
            new_balance = max(0, row2["Target"] - (0.9 * row2['Sets_per_week'] + 0.1 * score * 3 * 7))
            self.muscle_score.loc[index2, ["New_balance"]] = new_balance
            self.muscle_score.loc[index2, ["Temp_fatigue"]] += float(score)
            self.muscle_score.loc[index2, ["Sets_per_week"]] += 0.1 * float(score) * 3 * 7
        self.muscle_score["Balance"] = self.muscle_score["New_balance"]
        self.muscle_score["Workout_fatigue"] += self.muscle_score["Temp_fatigue"]

        self.remaining_duration -= duration

        # Write to list and remove from exercises
        self.PIPELINE.append(str(exercise_ID))
        self.exercise_score.drop([exercise_index], inplace=True)
        print("Selected", exercise_ID, row["Exercise"].iloc[0], "(", duration, "mins )")

        # Refresh projected scores for exercises
        self.new_balance_calc()
        self.overall_scores_calc()

        return True

    def output(self):
        ## select top 5 exercises
        print(self.exercise_score.sort_values(by="Score", ascending=False))

    def write_to_file(self, date):
        try:
            with open("inputs/history.csv", "a", newline='') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                row = [date] + self.PIPELINE
                csvWriter.writerow(row)
        except Exception as e:
            print(f"Failed to write to file: {e}")

    def forecast(self, days):
        for i in range(days):
            self.TODAY = dt.date.today() + dt.timedelta(days=1 + i)
            self.generate_workout(self.TODAY)
            print(self.muscle_score)

    def update_history_with_names(self):
        # Create a dictionary to map exercise IDs to names
        exercise_dict = pd.Series(self.exercises_data['Exercise'].values, index=self.exercises_data['ID']).to_dict()

        # Get the exercise columns
        columns = self.history.columns[1:11]  # Assuming the first column is 'Date' and next 10 are Exercise_1 to Exercise_10

        # Add new columns for exercise names
        for i in range(1, 11):
            self.history[f'Name_{i}'] = self.history.apply(lambda row: exercise_dict.get(self.safe_convert_to_int(row[columns[i - 1]]), ''), axis=1)

        # Save the updated history to CSV
        self.history.to_csv("inputs/history.csv", index=False)

    def safe_convert_to_int(self, value):
        try:
            # Remove 'e' suffix if present and convert to int
            return int(re.sub(r'e', '', str(value)))
        except (ValueError, TypeError):
            return np.nan


if __name__ == "__main__":
    x = Engine()
    x.forecast(1)
    # x.generate_workout(x.TODAY)
