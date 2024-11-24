# todo
# muscular fatigue (not flat 2.0, reduced custom to muscle based on higher frequency 50% EMA muscle usage)
# Injury
# Bodyweight workouts
# Time per workout slider instead of fatigue one
# Generate written text description of user preferences / read in user preferences from LLM
# Calories for cardio only - reward for cardio as well as lifting
# User irrational preference - prefer X rather than Y for the same benefit
# How are you feeling today - easy / medium / hard
# Imbalance > remaining

# imports
import pandas as pd
import numpy as np
import datetime as dt
import csv
import warnings
import re
import random as rd

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 400)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns', 20)


class Engine():
    def __init__(self):
        self.TODAY = dt.date.today()

    def constants(self):
        # Hard-coded constants
        self.EMA_LIFT_RATIO = 0
        self.PIPELINE = []
        self.MUSCLE_FATIGUE_EMA_BIAS = 0.5
        # User-defined constants
        self.SUNNY = False  # Toggle for Sunny
        if self.SUNNY:
            self.BASE_SETS_PER_WEEK = 4  # Target sets per week per muscle (further multiplied by muscle target weighting) - 4 to 12 is a reasonable range
            self.MUSCLE_FATIGUE_LIMIT = 0.8  # How frequently you can use the same muscle: e.g. 1 = Heavy biceps workout every 2 days, 0.5 = Heavy biceps workout every 4 days (2/x)
            self.TARGET_LIFT_RATIO = 0.4  # Target proportion of workout duration for lifts - e.g. 0.7 for strength focus, 0.3 for cardio focus
            self.TARGET_DURATION = 40  # Total exercise duration in minutes
            self.WEEKDAY_TARGET_DURATION = 20  # Custom total duration for weekdays
            # Select user priorities (score of X means that top exercise is 2^X times more likely to be picked)
            # -1 Strongly avoid --- -0.5 Avoid --- 0 Don't care --- 0.5 Favour --- 1 Strongly favour
            self.COMPOUNDNESS_WEIGHT = 0.5  # STRENGTH - prioritise highly compound full-body exercises
            self.BALANCE_WEIGHT = 0.5  # PHYSIQUE - prioritise hitting target muscles to grow
            self.UNILATERAL_WEIGHT = 0  # BALANCE - prioritise single-limb exercises fixing asymmetry

            self.FITNESS_WEIGHT = 1  # FITNESS - prioritise higher-intensity/longer cardio to improve fitness (compare with freshness)

            self.FRESHNESS_WEIGHT = 0.5  # NOVELTY - try to mix up new exercises to keep things interesting
            self.DURATION_WEIGHT = 0.5  # DURATION - prioritise exercises that each take longer

            self.FLEXIBILITY_WEIGHT = 0  # FLEXIBILITY - prioritise stretches to improve flexibility

            self.IS_HOME = False  # Toggle for home exercises
        else:
            self.BASE_SETS_PER_WEEK = 6  # Target sets per week per muscle (further multiplied by muscle target weighting) - 4 to 12 is a reasonable range
            self.MUSCLE_FATIGUE_LIMIT = 1  # How frequently you can use the same muscle: e.g. 1 = Heavy biceps workout every 2 days, 0.5 = Heavy biceps workout every 4 days (2/x)
            self.TARGET_LIFT_RATIO = 0.7  # Target proportion of workout duration for lifts - e.g. 0.7 for strength focus, 0.3 for cardio focus
            self.TARGET_DURATION = 35  # Total exercise duration in minutes
            # Select user priorities (score of X means that top exercise is 2^X times more likely to be picked)
            # -1 Strongly avoid --- -0.5 Avoid --- 0 Don't care --- 0.5 Favour --- 1 Strongly favour
            self.COMPOUNDNESS_WEIGHT = 0  # COMPOUNDNESS - prioritise highly compound full-body exercises
            self.BALANCE_WEIGHT = 1  # PHYSIQUE - prioritise hitting target muscles to grow
            self.UNILATERAL_WEIGHT = 0  # BALANCE - prioritise single-limb exercises fixing asymmetry

            self.FITNESS_WEIGHT = 0.5  # FITNESS - prioritise higher-intensity/longer cardio to improve fitness (compare with freshness)
            self.FLEXIBILITY_WEIGHT = 0  # FLEXIBILITY - prioritise stretches to improve flexibility
            self.FRESHNESS_WEIGHT = 0.5  # NOVELTY - try to mix up new exercises to keep things interesting
            self.DURATION_WEIGHT = 0  # DURATION - prioritise exercises that each take longer

            self.IS_HOME = False  # Toggle for home exercises

    def set_up(self):
        suffix = "_s" if self.SUNNY else ""
        self.history = pd.read_csv(f"inputs/history{suffix}.csv")
        self.exercises_data = pd.read_csv(f"inputs/exercises{suffix}.csv")
        self.muscle_targets_data = pd.read_csv(f"inputs/muscle_targets{suffix}.csv")
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
        # print(self.TODAY)
        # Convert 'Date' to datetime objects and then to date objects with new format
        self.history['Date'] = self.history['Date'].apply(lambda x: dt.datetime.strptime(x, '%d/%m/%y').date())
        # Filter out entries that are today or in the future
        self.history = self.history[self.history['Date'] < self.TODAY]
        # Convert 'Date' back to string format with new format
        self.history['Date'] = self.history['Date'].apply(lambda x: x.strftime('%d/%m/%y'))
        # Save the updated history to CSV
        suffix = "_s" if self.SUNNY else ""
        self.history.to_csv(f"inputs/history{suffix}.csv", index=False)

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

    def ratio_EMA_calc(self):
        EMA_score = EMA_counts = 0
        for index, row in self.history.iterrows():  # loop through history records
            daily_lift_duration = daily_total_duration = 0
            for i in range(10):  # loop through exercises completed
                exercise_entry = row[i + 1]
                match = re.match(r"(\d+)(e?)", str(exercise_entry))
                if not match:
                    continue

                exercise_ID = int(match.group(1))
                row2 = self.exercises_data[self.exercises_data["ID"] == exercise_ID]
                type = row2["Type"].iloc[0]
                duration = row2["Duration"].iloc[0]
                if type == "Lift":
                    daily_lift_duration += duration
                daily_total_duration += duration
            daily_lift_ratio = daily_lift_duration / daily_total_duration

            date_diff = (self.TODAY - dt.datetime.strptime(row["Date"], '%d/%m/%y').date()).days
            EMA_score += 0.1 * daily_lift_ratio * daily_total_duration * 0.9 ** (date_diff - 1)
            EMA_counts += 0.1 * 1 * daily_total_duration * 0.9 ** (date_diff - 1)
        self.EMA_LIFT_RATIO = EMA_score / EMA_counts

    def freshness_calc(self):
        # FRESHNESS: calculate exercise scores based on inverse days difference
        for index, row in self.exercise_score.iterrows():  # Loop through all exercises
            score = 0
            for index2, row2 in self.history.iterrows():  # Loop through history records
                date_diff = (self.TODAY - dt.datetime.strptime(row2["Date"], '%d/%m/%y').date()).days
                for i in range(10):  # Loop through exercises completed
                    exercise_entry = row2[i + 1]
                    match = re.match(r"(\d+)(e?)", str(exercise_entry))
                    if not match:
                        continue

                    exercise_ID = int(match.group(1))
                    if exercise_ID == row["ID"]:
                        score += 0.3 * 0.9 ** (date_diff - 1)
                    elif row["Type"] == "Cardio" and row["Exercise"].split()[0] == self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Exercise"].iloc[0].split()[0]:
                        score += 0.5 * 0.3 * 0.9 ** (date_diff - 1)
                        # print(row['Exercise'])

            # Rescale the freshness score to be between 1 and 10
            scaled_freshness = max(0, min(10, 10 * (1 - score)))
            self.exercise_score.loc[index, ["Freshness"]] = round(scaled_freshness, 2)

    def muscle_imbalance_calc(self):
        # BALANCE: calculate muscle sets per week EMA
        self.muscle_score[["Sets_per_week"]] = 0
        self.muscle_score[["EMA_fatigue"]] = 0

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
                        score = 0
                    else:
                        score = float(score)

                    # Adjust score for easy exercises before calculating EMA_score_fatigue
                    fatigue_score = round(score * (1 - 0.2 * is_easy), 2)
                    balance_score = round(score * (1 - 0.2 * is_easy), 2)
                    # print(score, fatigue_score, self.decode(exercise_entry), row2["Muscle"])

                    date_diff = (self.TODAY - dt.datetime.strptime(row["Date"], '%d/%m/%y').date()).days
                    EMA_score_balance = 0.1 * balance_score * 0.9 ** date_diff
                    EMA_score_fatigue = self.MUSCLE_FATIGUE_EMA_BIAS * fatigue_score * (1 - self.MUSCLE_FATIGUE_EMA_BIAS) ** date_diff if exercise_type != "Cardio" else 0
                    # #
                    # if index2 == 14:
                    #     print(date_diff, balance_score, EMA_score_balance)

                    self.muscle_score.loc[index2, ["Sets_per_week"]] += EMA_score_balance * 3 * 7
                    self.muscle_score.loc[index2, ["EMA_fatigue"]] += max(0, EMA_score_fatigue)  # Ensure non-negative
            # print(round(self.muscle_score.loc[14, ["Sets_per_week"]][0],2), row["Date"], date_diff)

    def muscle_target_calc(self):
        # Calculate muscle EMA target directly using BASE_SETS_PER_WEEK and Priority
        self.muscle_score["Target"] = self.muscle_targets_data["Priority"] * self.BASE_SETS_PER_WEEK
        # Calculate imbalance as the difference between Target and actual Sets_per_week
        self.muscle_score["Imbalance"] = self.muscle_score.apply(lambda row: max(0, row["Target"] - row["Sets_per_week"]), axis=1)

        # Debug statement to print Sets_per_week, Target, and Imbalance
        # print("Muscle Score Debug:")
        # print(self.muscle_score[["Muscle", "Sets_per_week", "Target", "Imbalance"]])

    def new_imbalance_calc(self):
        ## calculate projected imbalance scores for each exercise
        for index, row in self.exercise_score.iterrows():  # loop through all (active) exercises
            self.muscle_score[["New_imbalance"]] = self.muscle_score[["Imbalance"]]
            exercise_ID = row["ID"]
            exercise_type = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[0]

            if exercise_type == "Cardio":
                # Assign a fixed high value as a placeholder for cardio exercises
                self.exercise_score.loc[index, ["Imbalance"]] = 0
                continue

            for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
                score = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                score = float(score)
                new_imbalance = max(0, row2["Target"] - (0.9 * row2['Sets_per_week'] + 0.1 * score * 3 * 7))
                self.muscle_score.loc[index2, ["New_imbalance"]] = new_imbalance
            RMS = (self.muscle_score["New_imbalance"] ** 2).sum()
            self.exercise_score.loc[index, ["Imbalance"]] = RMS
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
        balance_best = self.exercise_score["Imbalance"].min()
        balance_worst = self.exercise_score["Imbalance"].max()
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

        def score_lift(row):
            return max(0,
                       (1 + (row["Freshness"] - freshness_worst) / self.score_range_fix(freshness_best, freshness_worst))
                       ** self.FRESHNESS_WEIGHT *
                       (1 + (row["Imbalance"] - balance_worst) / self.score_range_fix(balance_best, balance_worst))
                       ** self.BALANCE_WEIGHT *
                       (1 + (row["Compoundness"] - compoundness_worst) / self.score_range_fix(compoundness_best, compoundness_worst))
                       ** self.COMPOUNDNESS_WEIGHT *
                       (1 + (row["Unilateral"] - unilateral_worst) / self.score_range_fix(unilateral_best, unilateral_worst))
                       ** self.UNILATERAL_WEIGHT *
                       (1 + (row["Duration"] - duration_worst) / self.score_range_fix(duration_best, duration_worst))
                       ** self.DURATION_WEIGHT)

        def score_cardio(row):
            return max(0,
                       (1 + (row["Freshness"] - freshness_worst) / self.score_range_fix(freshness_best, freshness_worst))
                       ** self.FRESHNESS_WEIGHT *
                       (1 + (row["Fitness"] - fitness_worst) / self.score_range_fix(fitness_best, fitness_worst))
                       ** self.FITNESS_WEIGHT *
                       (1 + (row["Duration"] - duration_worst) / self.score_range_fix(duration_best, duration_worst))
                       ** self.DURATION_WEIGHT)

        for index, row in self.exercise_score.iterrows():
            if row["Type"] == "Lift":
                self.exercise_score.at[index, "Score"] = score_lift(row)
            elif row["Type"] == "Cardio":
                self.exercise_score.at[index, "Score"] = score_cardio(row)

    def prep_workout(self):
        # set up and read files
        self.constants()
        # Adjust target duration and home setting based on day of the week if SUNNY
        if self.SUNNY:
            if self.TODAY.weekday() < 5:  # Weekday
                self.TARGET_DURATION = self.WEEKDAY_TARGET_DURATION
                self.IS_HOME = True
            else:  # Weekend
                self.TARGET_DURATION = 40  # Reset to default weekend duration
                self.IS_HOME = False
        self.set_up()
        self.filter_out_today_entries()
        # refresh score baselines from history
        self.freshness_calc()
        self.muscle_imbalance_calc()
        self.muscle_target_calc()
        self.ratio_EMA_calc()

        # calculate projected scores for exercises
        self.new_imbalance_calc()
        self.overall_scores_calc()
        # reset workout fatigue values
        self.muscle_score[["Workout_fatigue"]] = 0
        # Calculate the lift probabilities for the current workout
        if self.IS_HOME:
            self.lift_probability = 1
        else:
            self.lift_probability = min(self.TARGET_LIFT_RATIO + 0.2, max(self.TARGET_LIFT_RATIO - 0.2, (2 * self.TARGET_LIFT_RATIO - self.EMA_LIFT_RATIO)))
        # print(self.lift_probability)

    def generate_workout(self, date, extras):
        # Initialise
        self.exercise_score_init = self.exercise_score.copy()
        self.remaining_duration = self.TARGET_DURATION
        # Track the first words of selected exercises
        self.selected_first_words = set()

        # Generate as many exercises as possible until the duration cap is reached
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        while self.remaining_duration >= 7:  # Assuming minimum exercise duration is 7 minutes
            self.exercise_score = self.exercise_score_init.copy()
            seeds.pop(0)

            # Convert date to an integer seed based on timestamp
            date_time = dt.datetime.combine(date, dt.datetime.min.time())
            timestamp = int(date_time.timestamp())
            seed_value = timestamp + seeds[0] * 3600  # Adding hours to ensure distinct values at the hour level
            rd.seed(seed_value)

            # Filter for the specified exercise type
            if rd.random() < self.lift_probability:
                self.exercise_score = self.exercise_score[self.exercise_score["Type"] == "Lift"]
            else:
                self.exercise_score = self.exercise_score[self.exercise_score["Type"] == "Cardio"]
            if self.exercise_score.empty:
                print("No more suitable exercises!")
                break
            # Select the top exercise
            selected = self.select_top_exercise()
            if not selected:
                break

        # Add the new row to the history DataFrame
        new_row = pd.DataFrame([[date.strftime('%d/%m/%y')] + self.PIPELINE], columns=self.history.columns[:len([date] + self.PIPELINE)])
        self.history = pd.concat([self.history, new_row], ignore_index=True)

        # Write the updated history to the file
        suffix = "_s" if self.SUNNY else ""
        self.history.to_csv(f"inputs/history{suffix}.csv", index=False)

        if extras:
            print("------Extras------")
            self.select_top_extras(5)

        # Recalculate muscle stats + update history
        self.muscle_imbalance_calc()
        self.muscle_target_calc()
        self.update_history_with_names()

    def select_top_exercise(self):
        # Filter for duration limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            if row["Duration"] > self.remaining_duration:
                self.exercise_score.drop(index, inplace=True)

        # Filter out exercises with the same first word as already selected ones
        self.exercise_score['First_Word'] = self.exercise_score['Exercise'].apply(lambda x: x.split()[0])
        self.exercise_score = self.exercise_score[~self.exercise_score['First_Word'].isin(self.selected_first_words)]

        # Filter for muscle fatigue limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            if row["Type"] != "Cardio":
                self.muscle_score[["Temp_fatigue"]] = 0
                for index2, row2 in self.muscle_score.iterrows():  # add fatigue for each muscle
                    score = self.exercises_data.loc[self.exercises_data["ID"] == row["ID"], row2["Muscle"]].iloc[0]
                    try:
                        int(score)
                    except ValueError:
                        continue
                    score = float(score)
                    self.muscle_score.loc[index2, ["Temp_fatigue"]] += float(score)
                    expected_new_EMA_fatigue = self.MUSCLE_FATIGUE_EMA_BIAS * (self.muscle_score.loc[index2, ["Workout_fatigue"]][0] + self.muscle_score.loc[index2, ["Temp_fatigue"]][0]) + \
                                               self.muscle_score.loc[index2, ["EMA_fatigue"]][0]
                    # if row["ID"] == 77 and row2["Muscle"] == "Quads":
                    #     print(row2["Workout_fatigue"], row2["Temp_fatigue"], expected_new_EMA_fatigue)
                    if expected_new_EMA_fatigue > self.MUSCLE_FATIGUE_LIMIT:
                        self.exercise_score.drop(index, inplace=True)
                        break

        # Find top exercise
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
        first_word = row["First_Word"].iloc[0]

        # Update mid-workout muscle imbalance + fatigue (only for lifts)
        if row["Type"].iloc[0] != "Cardio":
            self.muscle_score[["Temp_fatigue"]] = 0
            self.muscle_score[["New_imbalance"]] = self.muscle_score[["Imbalance"]]
            for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
                score = self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                score = float(score)
                new_imbalance = max(0, row2["Target"] - (0.9 * row2['Sets_per_week'] + 0.1 * score * 3 * 7))
                self.muscle_score.loc[index2, ["New_imbalance"]] = new_imbalance
                self.muscle_score.loc[index2, ["Temp_fatigue"]] += float(score)
                self.muscle_score.loc[index2, ["Sets_per_week"]] += 0.1 * float(score) * 3 * 7
            self.muscle_score["Imbalance"] = self.muscle_score["New_imbalance"]
            self.muscle_score["Workout_fatigue"] += self.muscle_score["Temp_fatigue"]

        self.remaining_duration -= duration

        # Write to list and remove from exercises
        self.PIPELINE.append(str(exercise_ID))
        self.exercise_score_init.drop([exercise_index], inplace=True)
        print("Selected", exercise_ID, row["Exercise"].iloc[0], "(", duration, "mins )")

        # Add the first word to the set of selected first words
        self.selected_first_words.add(first_word)

        # Refresh projected scores for exercises
        self.new_imbalance_calc()
        self.overall_scores_calc()

        return True

    def select_top_extras(self, count):
        # Filter out rows where ID is in the pipeline
        filtered_exercises = self.exercise_score_init[~self.exercise_score_init['ID'].isin([int(x) for x in self.PIPELINE])]
        filtered_exercises = filtered_exercises[filtered_exercises['Type'] == "Lift"]

        # Select the top N exercises based on overall score without considering fatigue
        top_exercises = filtered_exercises.sort_values(by="Score", ascending=False).head(count)

        for index, row in top_exercises.iterrows():
            print("Extra option:", row["ID"], row["Exercise"], "(", row["Duration"], "mins )")

    def write_to_file(self, date):
        try:
            with open("inputs/history.csv", "a", newline='') as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=',')
                row = [date] + self.PIPELINE
                csvWriter.writerow(row)
        except Exception as e:
            print(f"Failed to write to file: {e}")

    def forecast(self, offset, days, extras, stats):
        self.TODAY += dt.timedelta(days=offset)
        for i in range(days):
            self.prep_workout()
            formatted_date = self.TODAY.strftime("%a %d %b")
            home_flag = "(home) " if self.IS_HOME else ""
            print(f"\n=============== Day {i + 1}: {formatted_date}, Duration: {self.TARGET_DURATION} mins {home_flag}===============\n")
            if 2 in stats:
                print(self.exercise_score[["Exercise", "Freshness", "Imbalance", "Score"]])
            if 0 in stats:
                print("------Starting Stats------")
                print(f"EMA lift ratio: {round(self.EMA_LIFT_RATIO, 2)}, Lift probability today: {round(self.lift_probability, 2)}")
                print(f"Old avg. sets per week: {round(self.muscle_score["Sets_per_week"].mean(), 1)} (higher = better)")
                print(f"Old avg. imbalance: {round((self.muscle_score["Imbalance"] ** 2).mean() ** 0.5, 2)} (lower = better)")
            if 1 in stats and i == 0:
                print(self.muscle_score[["Muscle", "Workout_fatigue", "EMA_fatigue", "Sets_per_week", "Target", "Imbalance"]])
            print("------Generated Workout------")
            self.generate_workout(self.TODAY, extras)
            if 0 in stats:
                print("------Updated Stats------")
                print(f"New avg. sets per week: {round(self.muscle_score["Sets_per_week"].mean(), 1)} (higher = better)")
                print(f"New avg. imbalance: {round((self.muscle_score["Imbalance"] ** 2).mean() ** 0.5, 2)} (lower = better)")
            if 1 in stats:
                print(self.muscle_score[["Muscle", "Workout_fatigue", "EMA_fatigue", "Sets_per_week", "Target", "Imbalance"]])

            self.TODAY += dt.timedelta(days=1)

    def update_history_with_names(self):
        # Create a dictionary to map exercise IDs to names
        exercise_dict = pd.Series(self.exercises_data['Exercise'].values, index=self.exercises_data['ID']).to_dict()

        # Get the exercise columns
        columns = self.history.columns[1:11]  # Assuming the first column is 'Date' and next 10 are Exercise_1 to Exercise_10

        # Add new columns for exercise names
        for i in range(1, 11):
            self.history[f'Name_{i}'] = self.history.apply(lambda row: exercise_dict.get(self.safe_convert_to_int(row[columns[i - 1]]), ''), axis=1)

        # Save the updated history to CSV
        suffix = "_s" if self.SUNNY else ""
        self.history.to_csv(f"inputs/history{suffix}.csv", index=False)

    def safe_convert_to_int(self, value):
        try:
            # Remove 'e' suffix if present and convert to int
            return int(re.sub(r'e', '', str(value)))
        except (ValueError, TypeError):
            return np.nan


if __name__ == "__main__":
    x = Engine()
    x.forecast(0, 1, True, [0])
