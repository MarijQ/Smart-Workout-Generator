# todo
# muscular fatigue (not flat 2.0, reduced custom to muscle based on higher frequency 50% EMA muscle usage)
# Injury
# Bodyweight workouts
# Time per workout slider instead of fatigue one
# Generate written text description of user preferences / read in user preferences from LLM
# Calories for cardio only - reward for cardio as well as lifting
# User irrational preference - prefer X rather than Y for the same benefit
# How are you feeling today - easy / medium / hard
# Deviation > remaining

# imports
import pandas as pd
import numpy as np
import datetime as dt
import csv
import warnings
import re
import random as rd

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.width", 400)
pd.set_option("display.min_rows", 20)
pd.set_option("display.max_columns", 20)


class Engine:
    def __init__(self):
        self.TODAY = dt.date.today()

    # -------------------- Constants and Setup --------------------

    def constants(self):
        # Hard-coded constants
        self.EMA_LIFT_RATIO = 0
        self.PIPELINE = []
        self.MUSCLE_FATIGUE_EMA_BIAS = 0.5
        # User-defined constants
        self.SUNNY = False  # Toggle for Sunny
        if self.SUNNY:
            # How frequently you can use the same muscle: e.g. 1 = Heavy biceps workout every 2 days, 0.5 = Heavy biceps workout every 4 days (2/x)
            self.MUSCLE_FATIGUE_LIMIT = 1
            # Target proportion of workout duration for lifts - e.g. 0.7 for strength focus, 0.3 for cardio focus
            self.TARGET_LIFT_RATIO = 0.7
            # Total exercise duration in minutes (weekends)
            self.TARGET_DURATION = 35
            self.WEEKDAY_TARGET_DURATION = 20  # Custom total duration for weekdays
            self.IS_HOME = False  # Toggle for home exercises

            # Select user priorities (score of X means that top exercise is 2^X times more likely to be picked)
            # -1 Strongly avoid --- -0.5 Avoid --- 0 Don't care --- 0.5 Favour --- 1 Strongly favour

            # lifts
            # growing as much total muscle as possible, anywhere on the body (optimal)
            self.GROWTH_FACTOR = 1
            # prioritise neglected muscles according to muscle_targets preferences (slightly less optimal)
            self.PROPORTIONS_FACTOR = 0
            # prioritise single-limb exercises fixing asymmetry (moderately less optimal)
            self.SYMMETRY_FACTOR = 0
            # prioritise exercises you haven't done recently to mix things up (moderately less optimal)
            self.VARIETY_FACTOR = 0
            # prioritise user-defined favourites e.g. to train for competitions (moderately less optimal)
            self.FAVOURITES_FACTOR = 0
            # prioritise less fatiguing exercises to minimise overall exhaustion (significantly less optimal)
            self.COMFORT_FACTOR = 0

            # cardio
            # prioritise burning the most calories (optimal)
            self.CALORIES_FACTOR = 0.5
            # prioritise exercises you haven't done recently to mix things up (moderately less optimal)
            self.VARIETY_FACTOR_C = 0
            # prioritise user-defined favourites e.g. to train for competitions (moderately less optimal)
            self.FAVOURITES_FACTOR_C = 0
            # prioritise less fatiguing exercises to minimise overall exhaustion (significantly less optimal)
            self.COMFORT_FACTOR_C = 0

        else:
            # How frequently you can use the same muscle: e.g. 1 = Heavy biceps workout every 2 days, 0.5 = Heavy biceps workout every 4 days (2/x)
            self.MUSCLE_FATIGUE_LIMIT = 0.7
            # Target proportion of workout duration for lifts - e.g. 0.7 for strength focus, 0.3 for cardio focus
            self.TARGET_LIFT_RATIO = 0.7
            self.TARGET_DURATION = 35  # Total exercise duration in minutes
            self.IS_HOME = False  # Toggle for home exercises

            # Select user priorities (score of X means that top exercise is 2^X times more likely to be picked)
            # -1 Strongly avoid --- -0.5 Avoid --- 0 Don't care --- 0.5 Favour --- 1 Strongly favour

            # lifts
            # growing as much total muscle as possible, anywhere on the body (optimal)
            self.GROWTH_FACTOR = 0.5
            # prioritise neglected muscles according to muscle_targets preferences (slightly less optimal)
            self.PROPORTIONS_FACTOR = 1
            # prioritise single-limb exercises fixing asymmetry (moderately less optimal)
            self.SYMMETRY_FACTOR = 0
            # prioritise exercises you haven't done recently to mix things up (moderately less optimal)
            self.VARIETY_FACTOR = 0.25
            # prioritise user-defined favourites e.g. to train for competitions (depends)
            self.FAVOURITES_FACTOR = 0.25
            # prioritise less fatiguing exercises to minimise overall exhaustion (significantly less optimal)
            self.COMFORT_FACTOR = 0

            # cardio
            # prioritise burning the most calories (optimal)
            self.CALORIES_FACTOR = 0
            # prioritise exercises you haven't done recently to mix things up (moderately less optimal)
            self.VARIETY_FACTOR_C = 1
            # prioritise user-defined favourites e.g. to train for competitions (moderately less optimal)
            self.FAVOURITES_FACTOR_C = 0
            # prioritise less fatiguing exercises to minimise overall exhaustion (significantly less optimal)
            self.COMFORT_FACTOR_C = 0

    def set_up(self):
        suffix = "_s" if self.SUNNY else ""
        # Update to read .ods instead of .xlsx
        self.history = pd.read_excel(
            f"Smartlift-[GH]/inputs/history{suffix}.ods", engine="odf", sheet_name="history"
        )
        self.exercises_data = pd.read_excel(
            f"Smartlift-[GH]/inputs/exercises{suffix}.ods",
            engine="odf",
            sheet_name="exercises",
            skiprows=1,
        )
        self.muscle_targets_data = pd.read_excel(
            f"Smartlift-[GH]/inputs/muscle_targets{suffix}.ods",
            engine="odf",
            sheet_name="muscle_targets",
        )

        # Set up dataframes similar to before
        self.exercise_score = self.exercises_data.loc[
            self.exercises_data["Active"] >= 1,
            [
                "ID",
                "Exercise",
                "Type",
                "Active",
                "Unilateral",
                "Calories",
                "Fatigue",
                "Growth",
                "Duration",
                "Home",
            ],
        ]

        if self.IS_HOME:
            self.exercise_score = self.exercise_score[self.exercise_score["Home"] == 1]

        self.exercise_score[["Freshness"]] = 0
        self.muscle_score = pd.DataFrame(
            list(self.exercises_data.columns[12:32]), columns=["Muscle"]
        )
        self.muscle_score[["Sets_per_week"]] = 0.0  # Initialize as float
        self.muscle_score[["Workout_fatigue"]] = 0.0  # Initialize as float
        self.muscle_score[["Temp_fatigue"]] = 0.0  # Initialize as float
        self.muscle_score[["EMA_fatigue"]] = 0.0  # Initialize as float
        self.PIPELINE = []

        # Convert types (in case they aren't already float)
        self.muscle_score["Sets_per_week"] = self.muscle_score["Sets_per_week"].astype(
            float
        )
        self.muscle_score["Workout_fatigue"] = self.muscle_score[
            "Workout_fatigue"
        ].astype(float)
        self.muscle_score["Temp_fatigue"] = self.muscle_score["Temp_fatigue"].astype(
            float
        )
        self.muscle_score["EMA_fatigue"] = self.muscle_score["EMA_fatigue"].astype(
            float
        )

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
        self.muscle_deviation_calc()
        self.muscle_target_calc()
        self.history_EMA_calc()

        # calculate projected scores for exercises
        self.new_deviation_calc()
        self.overall_scores_calc()
        # reset workout fatigue values
        self.muscle_score[["Workout_fatigue"]] = 0
        # Calculate the lift probabilities for the current workout
        if self.IS_HOME:
            self.lift_probability = 1
        else:
            self.lift_probability = min(
                self.TARGET_LIFT_RATIO + 0.2,
                max(
                    self.TARGET_LIFT_RATIO - 0.2,
                    (2 * self.TARGET_LIFT_RATIO - self.EMA_LIFT_RATIO),
                ),
            )
        # print(self.lift_probability)

    # -------------------- History Management --------------------

    def filter_out_today_entries(self):
        """Remove entries from history that are already logged for today or future dates."""
        # Ensure 'Date' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.history["Date"]):
            self.history["Date"] = pd.to_datetime(
                self.history["Date"], format="%d/%m/%y"
            )

        # Convert 'Date' to date object for comparison
        self.history["Date"] = self.history["Date"].dt.date

        # Filter out entries that are today or in the future
        self.history = self.history[self.history["Date"] < self.TODAY]

        # Convert 'Date' back to string format with new format
        self.history["Date"] = self.history["Date"].apply(
            lambda x: x.strftime("%d/%m/%y")
        )

        # Save the updated history to .ods
        suffix = "_s" if self.SUNNY else ""
        with pd.ExcelWriter(f"Smartlift-[GH]/inputs/history{suffix}.ods", engine="odf") as writer:
            self.history.to_excel(writer, sheet_name="history", index=False)

    def update_history_with_names(self):
        # Create a dictionary to map exercise IDs to names
        exercise_dict = pd.Series(
            self.exercises_data["Exercise"].values, index=self.exercises_data["ID"]
        ).to_dict()

        # Get the exercise columns
        columns = self.history.columns[
            1:11
        ]  # Assuming the first column is 'Date' and the next 10 are Exercise_1 to Exercise_10

        # Add new columns for exercise names
        for i in range(1, 11):
            self.history[f"Name_{i}"] = self.history.apply(
                lambda row: exercise_dict.get(
                    self.safe_convert_to_int(row[columns[i - 1]]), ""
                ),
                axis=1,
            )

        # Save the updated history to Excel (ODS format)
        suffix = "_s" if self.SUNNY else ""
        with pd.ExcelWriter(f"Smartlift-[GH]/inputs/history{suffix}.ods", engine="odf") as writer:
            self.history.to_excel(writer, sheet_name="history", index=False)

    def write_to_file(self, date):
        try:
            with open("Smartlift-[GH]/inputs/history.csv", "a", newline="") as my_csv:
                csvWriter = csv.writer(my_csv, delimiter=",")
                row = [date] + self.PIPELINE
                csvWriter.writerow(row)
        except Exception as e:
            print(f"Failed to write to file: {e}")

    # -------------------- Calculations --------------------

    def history_EMA_calc(self):
        EMA_ratio_score = EMA_calories_score = EMA_counts = 0
        for index, row in self.history.iterrows():  # loop through history records
            daily_lift_duration = daily_total_duration = daily_total_calories = 0
            for i in range(10):  # loop through exercises completed
                exercise_entry = row[i + 1]
                match = re.match(r"(\d+)(e?)", str(exercise_entry))
                if not match:
                    continue

                exercise_ID = int(match.group(1))
                row2 = self.exercises_data[self.exercises_data["ID"] == exercise_ID]
                type = row2["Type"].iloc[0]
                duration = row2["Duration"].iloc[0]
                calories = row2["Calories"].iloc[0]
                is_easy = match.group(2) == "e"
                if type != "Cardio":
                    daily_lift_duration += duration
                else:
                    daily_total_calories += calories * (1 - is_easy * 0.2)
                daily_total_duration += duration
            daily_lift_ratio = daily_lift_duration / daily_total_duration

            date_diff = (
                self.TODAY - dt.datetime.strptime(row["Date"], "%d/%m/%y").date()
            ).days
            EMA_ratio_score += (
                0.1 * daily_lift_ratio * daily_total_duration * 0.9 ** (date_diff - 1)
            )
            EMA_calories_score += 0.1 * daily_total_calories * 0.9 ** (date_diff - 1)
            EMA_counts += 0.1 * 1 * daily_total_duration * 0.9 ** (date_diff - 1)
        self.EMA_LIFT_RATIO = EMA_ratio_score / EMA_counts
        self.EMA_CALORIES = EMA_calories_score

    def freshness_calc(self):
        # FRESHNESS: calculate exercise scores based on inverse days difference
        for index, row in self.exercise_score.iterrows():  # Loop through all exercises
            score = 0
            for index2, row2 in self.history.iterrows():  # Loop through history records
                date_diff = (
                    self.TODAY - dt.datetime.strptime(row2["Date"], "%d/%m/%y").date()
                ).days
                for i in range(10):  # Loop through exercises completed
                    exercise_entry = row2[i + 1]
                    match = re.match(r"(\d+)(e?)", str(exercise_entry))
                    if not match:
                        continue

                    exercise_ID = int(match.group(1))
                    if exercise_ID == row["ID"]:
                        score += 0.3 * 0.9 ** (date_diff - 1)
                    elif (
                        row["Type"] == "Cardio"
                        and row["Exercise"].split()[0]
                        == self.exercises_data.loc[
                            self.exercises_data["ID"] == exercise_ID, "Exercise"
                        ]
                        .iloc[0]
                        .split()[0]
                    ):
                        score += 0.5 * 0.3 * 0.9 ** (date_diff - 1)
                        # print(row['Exercise'])

            # Rescale the freshness score to be between 1 and 10
            scaled_freshness = max(0, min(10, 10 * (1 - score)))
            self.exercise_score.loc[index, ["Freshness"]] = round(scaled_freshness, 2)

    def muscle_deviation_calc(self):
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
                is_easy = is_easy == "e"
                for (
                    index2,
                    row2,
                ) in (
                    self.muscle_score.iterrows()
                ):  # add EMA contributions for each muscle
                    score = self.exercises_data.loc[
                        self.exercises_data["ID"] == exercise_ID, row2["Muscle"]
                    ].iloc[0]
                    try:
                        int(score)
                    except ValueError:
                        continue
                    exercise_type = self.exercises_data.loc[
                        self.exercises_data["ID"] == exercise_ID, "Type"
                    ].iloc[0]
                    score = float(score)

                    # Adjust score for easy exercises before calculating EMA_score_fatigue
                    fatigue_score = round(score * (1 - 0.2 * is_easy), 2)
                    deviation_score = round(score * (1 - 0.2 * is_easy), 2)
                    # print(score, fatigue_score, self.decode(exercise_entry), row2["Muscle"])

                    date_diff = (
                        self.TODAY
                        - dt.datetime.strptime(row["Date"], "%d/%m/%y").date()
                    ).days
                    EMA_score_deviation = 0.1 * deviation_score * 0.9**date_diff
                    EMA_score_fatigue = (
                        self.MUSCLE_FATIGUE_EMA_BIAS
                        * fatigue_score
                        * (1 - self.MUSCLE_FATIGUE_EMA_BIAS) ** date_diff
                        if exercise_type != "Cardio"
                        else 0
                    )
                    # #
                    # if index2 == 14:
                    #     print(date_diff, deviation_score, EMA_score_deviation)

                    self.muscle_score.loc[index2, ["Sets_per_week"]] += (
                        EMA_score_deviation * 3 * 7
                    )
                    # Ensure non-negative
                    self.muscle_score.loc[index2, ["EMA_fatigue"]] += max(
                        0, EMA_score_fatigue
                    )
            # print(round(self.muscle_score.loc[14, ['Sets_per_week']][0],2), row["Date"], date_diff)

    def muscle_target_calc(self, eod=False):
        if eod:
            self.muscle_score["Sets_per_week"] *= 0.9
            self.EMA_CALORIES *= 0.9

        # Calculate the current average EMA sets per week
        avg_ema_sets_per_week = self.muscle_score["Sets_per_week"].mean()
        # Calculate the total priority sum
        total_priority = self.muscle_targets_data["Priority"].sum()

        # Calculate the target for each muscle based on its priority
        self.muscle_score["Target"] = (
            (self.muscle_targets_data["Priority"] / total_priority)
            * avg_ema_sets_per_week
            * len(self.muscle_targets_data)
        )

        # Calculate deviation as the absolute difference between Target and actual Sets_per_week
        self.muscle_score["Deviation"] = (
            self.muscle_score["Sets_per_week"] - self.muscle_score["Target"]
        )

        # Debug statement to print Sets_per_week, Target, and Deviation
        # print("Muscle Score Debug:")
        # print(self.muscle_score[["Muscle", 'Sets_per_week', "Target", "Deviation"]])

    def new_deviation_calc(self, exercise_ID=None):
        # Calculate projected deviation scores for each exercise
        exercises_to_calculate = (
            self.exercise_score
            if exercise_ID is None
            else self.exercise_score[self.exercise_score["ID"] == exercise_ID]
        )

        for (
            index,
            row,
        ) in exercises_to_calculate.iterrows():  # Loop through all (active) exercises
            exercise_ID = row["ID"]
            exercise_type = self.exercises_data.loc[
                self.exercises_data["ID"] == exercise_ID, "Type"
            ].iloc[0]

            # Calculate projected sets per week for each muscle
            for index2, row2 in self.muscle_score.iterrows():
                score = self.exercises_data.loc[
                    self.exercises_data["ID"] == exercise_ID, row2["Muscle"]
                ].iloc[0]
                try:
                    int(score)
                except ValueError:
                    score = 0
                score = float(score)
                projected_sets_per_week = (
                    0.9 * row2["Sets_per_week"] + 0.1 * score * 3 * 7
                )
                self.muscle_score.loc[index2, ["Projected_Sets_per_week"]] = (
                    projected_sets_per_week
                )

            # Calculate the projected mean EMA sets per week
            projected_avg_ema_sets_per_week = self.muscle_score[
                "Projected_Sets_per_week"
            ].mean()

            # Calculate the projected target for each muscle based on its priority
            total_priority = self.muscle_targets_data["Priority"].sum()
            self.muscle_score["Projected_Target"] = (
                (self.muscle_targets_data["Priority"] / total_priority)
                * projected_avg_ema_sets_per_week
                * len(self.muscle_targets_data)
            )

            # Calculate the projected deviation as the difference between the projected target and the projected sets per week
            self.muscle_score["New_deviation"] = (
                self.muscle_score["Projected_Sets_per_week"]
                - self.muscle_score["Projected_Target"]
            )

            # Calculate the RMS of the new deviations for the exercise
            RMS = (self.muscle_score["New_deviation"] ** 2).mean() ** 0.5 - (
                self.muscle_score["Deviation"] ** 2
            ).mean() ** 0.5
            self.exercise_score.loc[index, ["Deviation_change"]] = RMS

            # Debug statement to print Sets_per_week, Target, and Deviation
            # print(f"Muscle Deviation Calc Debug for {self.decode(str(exercise_ID))}")
            # print(self.muscle_score[["Muscle", 'Sets_per_week', "Target", "Deviation_change", "Projected_Sets_per_week", "Projected_Target", "New_deviation"]])

    def overall_scores_calc(self):
        # calculate overall scores for each exercise
        self.exercise_score[["Score"]] = 0

        # Best and worst values for each metric
        growth_best = self.exercise_score["Growth"].max()
        growth_worst = self.exercise_score["Growth"].min()

        # Lower is better
        proportions_best = self.exercise_score["Deviation_change"].min()
        proportions_worst = self.exercise_score["Deviation_change"].max()

        symmetry_best = self.exercise_score["Unilateral"].max()
        symmetry_worst = self.exercise_score["Unilateral"].min()

        variety_best = self.exercise_score["Freshness"].max()
        variety_worst = self.exercise_score["Freshness"].min()

        favourites_best = self.exercise_score["Active"].max()
        favourites_worst = self.exercise_score["Active"].min()

        comfort_best = self.exercise_score["Fatigue"].min()  # Lower is better
        comfort_worst = self.exercise_score["Fatigue"].max()

        calories_best = self.exercise_score["Calories"].max()
        calories_worst = self.exercise_score["Calories"].min()

        def score_lift(row):
            return max(
                0,
                (
                    1
                    + (row["Growth"] - growth_worst)
                    / self.score_range_fix(growth_best, growth_worst)
                )
                ** self.GROWTH_FACTOR
                * (
                    1
                    + (row["Deviation_change"] - proportions_worst)
                    / self.score_range_fix(proportions_best, proportions_worst)
                )
                ** self.PROPORTIONS_FACTOR
                * (
                    1
                    + (row["Unilateral"] - symmetry_worst)
                    / self.score_range_fix(symmetry_best, symmetry_worst)
                )
                ** self.SYMMETRY_FACTOR
                * (
                    1
                    + (row["Freshness"] - variety_worst)
                    / self.score_range_fix(variety_best, variety_worst)
                )
                ** self.VARIETY_FACTOR
                * (
                    1
                    + (row["Active"] - favourites_worst)
                    / self.score_range_fix(favourites_best, favourites_worst)
                )
                ** self.FAVOURITES_FACTOR
                * (
                    1
                    + (comfort_best - row["Fatigue"])
                    / self.score_range_fix(comfort_best, comfort_worst)
                )
                ** self.COMFORT_FACTOR,
            )

        def score_cardio(row):
            return max(
                0,
                (
                    1
                    + (row["Calories"] - calories_worst)
                    / self.score_range_fix(calories_best, calories_worst)
                )
                ** self.CALORIES_FACTOR
                * (
                    1
                    + (row["Freshness"] - variety_worst)
                    / self.score_range_fix(variety_best, variety_worst)
                )
                ** self.VARIETY_FACTOR_C
                * (
                    1
                    + (row["Active"] - favourites_worst)
                    / self.score_range_fix(favourites_best, favourites_worst)
                )
                ** self.FAVOURITES_FACTOR_C
                * (
                    1
                    + (comfort_best - row["Fatigue"])
                    / self.score_range_fix(comfort_best, comfort_worst)
                )
                ** self.COMFORT_FACTOR_C,
            )

        for index, row in self.exercise_score.iterrows():
            if row["Type"] == "Cardio":
                self.exercise_score.at[index, "Score"] = score_cardio(row)
            else:
                self.exercise_score.at[index, "Score"] = score_lift(row)

    def score_range_fix(self, best, worst):
        if best == worst:
            return 1
        else:
            return best - worst

    # -------------------- Workout Generation --------------------

    def generate_workout(self, date, extras):
        # Initialise
        self.exercise_score_init = self.exercise_score.copy()
        self.remaining_duration = self.TARGET_DURATION
        self.selected_first_words = set()

        # Expected cardio workout time calculation
        expected_cardio_time = (1 - self.lift_probability) * self.TARGET_DURATION

        # Generate as many exercises as possible until the duration cap is reached
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        while (
            self.remaining_duration >= 7
        ):  # Assuming minimum exercise duration is 7 minutes
            self.exercise_score = self.exercise_score_init.copy()
            self.muscle_target_calc()
            self.new_deviation_calc()
            self.overall_scores_calc()

            date_time = dt.datetime.combine(date, dt.datetime.min.time())
            timestamp = int(date_time.timestamp())
            # Adding hours for distinct values
            seed_value = timestamp + seeds[0] * 3600
            rd.seed(seed_value)
            seeds.pop(0)

            if rd.random() < self.lift_probability:
                self.exercise_score = self.exercise_score[
                    self.exercise_score["Type"] != "Cardio"
                ]
                reduce_cardio_prob = False
            else:
                self.exercise_score = self.exercise_score[
                    self.exercise_score["Type"] == "Cardio"
                ]
                reduce_cardio_prob = True

            if self.exercise_score.empty:
                print("No more suitable exercises!")
                break

            selected = self.select_top_exercise()
            if not selected:
                break

            # If a cardio exercise is selected, adjust lift_probability
            if reduce_cardio_prob:
                selected_exercise_id = int(self.PIPELINE[-1])
                selected_row = self.exercises_data[
                    self.exercises_data["ID"] == selected_exercise_id
                ]
                selected_duration = selected_row["Duration"].values[0]
                self.lift_probability = min(
                    1, self.lift_probability + selected_duration / expected_cardio_time
                )

        new_row = pd.DataFrame(
            [[date.strftime("%d/%m/%y")] + self.PIPELINE],
            columns=self.history.columns[: len([date] + self.PIPELINE)],
        )
        self.history = pd.concat([self.history, new_row], ignore_index=True)

        suffix = "_s" if self.SUNNY else ""
        with pd.ExcelWriter(f"Smartlift-[GH]/inputs/history{suffix}.ods", engine="odf") as writer:
            self.history.to_excel(writer, sheet_name="history", index=False)

        if extras:
            print("------Extras------")
            self.select_top_extras(5)

        # Recalculate muscle stats + update history
        self.muscle_deviation_calc()
        self.muscle_target_calc()
        self.update_history_with_names()

    def select_top_exercise(self):
        # Filter for duration limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            if row["Duration"] > self.remaining_duration:
                self.exercise_score.drop(index, inplace=True)

        # Filter out exercises with the same first word as already selected ones
        self.exercise_score["First_Word"] = self.exercise_score["Exercise"].apply(
            lambda x: x.split()[0]
        )
        self.exercise_score = self.exercise_score[
            ~self.exercise_score["First_Word"].isin(self.selected_first_words)
        ]

        # Filter for muscle fatigue limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            if row["Type"] != "Cardio":
                self.muscle_score[["Temp_fatigue"]] = 0
                for (
                    index2,
                    row2,
                ) in self.muscle_score.iterrows():  # add fatigue for each muscle
                    score = self.exercises_data.loc[
                        self.exercises_data["ID"] == row["ID"], row2["Muscle"]
                    ].iloc[0]
                    try:
                        int(score)
                    except ValueError:
                        continue
                    score = float(score)
                    self.muscle_score.loc[index2, ["Temp_fatigue"]] += float(score)
                    expected_new_EMA_fatigue = (
                        self.MUSCLE_FATIGUE_EMA_BIAS
                        * (
                            self.muscle_score.loc[index2, ["Workout_fatigue"]][0]
                            + self.muscle_score.loc[index2, ["Temp_fatigue"]][0]
                        )
                        + self.muscle_score.loc[index2, ["EMA_fatigue"]][0]
                    )
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

        # Update mid-workout muscle sets per week and workout fatigue
        for (
            index2,
            row2,
        ) in self.muscle_score.iterrows():  # add EMA contribution for each muscle
            score = self.exercises_data.loc[
                self.exercises_data["ID"] == exercise_ID, row2["Muscle"]
            ].iloc[0]
            try:
                int(score)
            except ValueError:
                continue
            score = float(score)
            self.muscle_score.loc[index2, ["Sets_per_week"]] += 0.1 * score * 3 * 7
            self.muscle_score.loc[index2, ["Workout_fatigue"]] += score

        # reset temp fatigue and update duration and EMA calories (if cardio)
        self.muscle_score[["Temp_fatigue"]] = 0
        self.remaining_duration -= duration
        if row["Type"].iloc[0] == "Cardio":
            self.EMA_CALORIES += 0.1 * row["Calories"].iloc[0]

        # Write to list and remove from exercises
        self.PIPELINE.append(str(int(exercise_ID)))
        self.exercise_score_init.drop([exercise_index], inplace=True)
        print(
            "Selected",
            int(exercise_ID),
            row["Exercise"].iloc[0],
            "(",
            int(duration),
            "mins )",
        )

        # Add the first word to the set of selected first words
        self.selected_first_words.add(first_word)

        return True

    def select_top_extras(self, count):
        # Filter out rows where ID is in the pipeline
        filtered_exercises = self.exercise_score_init[
            ~self.exercise_score_init["ID"].isin([int(x) for x in self.PIPELINE])
        ]
        filtered_exercises = filtered_exercises[filtered_exercises["Type"] != "Cardio"]

        # Select the top N exercises based on overall score without considering fatigue
        top_exercises = filtered_exercises.sort_values(
            by="Score", ascending=False
        ).head(count)

        for index, row in top_exercises.iterrows():
            print(
                "Extra option:",
                int(row["ID"]),
                row["Exercise"],
                "(",
                int(row["Duration"]),
                "mins )",
            )

    # -------------------- Utility Functions --------------------

    def decode(self, exercise_str):
        match = re.match(r"(\d+)(e?)", exercise_str)
        if match:
            exercise_id = int(match.group(1))
            is_easy = match.group(2) == "e"

            # Find the exercise name corresponding to the exercise ID
            exercise_name = self.exercises_data.loc[
                self.exercises_data["ID"] == exercise_id, "Exercise"
            ]

            if not exercise_name.empty:
                return exercise_name.iloc[0], is_easy
            else:
                raise ValueError("Exercise ID not found in exercises_data.")
        else:
            raise ValueError("Invalid exercise string format.")

    def safe_convert_to_int(self, value):
        try:
            # Remove 'e' suffix if present and convert to int
            return int(re.sub(r"e", "", str(value)))
        except (ValueError, TypeError):
            return np.nan

    # -------------------- Forecasting --------------------

    def forecast(self, offset, days, options):
        self.TODAY += dt.timedelta(days=offset)
        for i in range(days):
            self.prep_workout()
            formatted_date = self.TODAY.strftime("%a %d %b")
            home_flag = "(home) " if self.IS_HOME else ""
            print(
                f"\n=============== Day {i + 1}: {formatted_date}, Duration: {self.TARGET_DURATION} mins {home_flag}===============\n"
            )
            if "exercises" in options:
                print(
                    self.exercise_score[
                        ["Exercise", "Freshness", "Deviation_change", "Score"]
                    ]
                )
            if "stats" in options:
                print("------Starting Stats------")
                print(
                    f"EMA lift ratio: {round(self.EMA_LIFT_RATIO, 2)}, Lift probability today: {round(self.lift_probability, 2)}"
                )
                print(
                    f"Old avg. sets/week/muscle:\t\t{round(self.muscle_score['Sets_per_week'].mean(), 1)} ({round(self.muscle_score['Sets_per_week'].mean() / 10 * 100)} % of optimal)"
                )
                print(
                    f"Old avg. calories/week:\t\t\t{round(self.EMA_CALORIES * 7)} ({round(self.EMA_CALORIES * 7 / 2000 * 100)} % of optimal)"
                )
                print(
                    f"Old avg. disproportion sets:\t{round((self.muscle_score['Deviation'] ** 2).mean() ** 0.5, 1)} ({100 - round((self.muscle_score['Deviation'] ** 2).mean() ** 0.5 / self.muscle_score['Sets_per_week'].mean() * 100)} % of optimal)"
                )
            if "muscles" in options and i == 0:
                print(
                    self.muscle_score[
                        [
                            "Muscle",
                            "Workout_fatigue",
                            "EMA_fatigue",
                            "Sets_per_week",
                            "Target",
                            "Deviation",
                        ]
                    ].sort_values(by="Deviation", ascending=True)
                )
            print("------Generated Workout------")
            self.generate_workout(self.TODAY, "extras" in options)
            self.muscle_target_calc(eod=True)
            if "stats" in options:
                print("------Updated Stats------")
                print(
                    f"New avg. sets/week/muscle:\t\t{round(self.muscle_score['Sets_per_week'].mean(), 1)} ({round(self.muscle_score['Sets_per_week'].mean() / 10 * 100)} % of optimal)"
                )
                print(
                    f"New avg. calories/week:\t\t\t{round(self.EMA_CALORIES * 7)} ({round(self.EMA_CALORIES * 7 / 2000 * 100)} % of optimal)"
                )
                print(
                    f"New avg. disproportion sets:\t{round((self.muscle_score['Deviation'] ** 2).mean() ** 0.5, 1)} ({100 - round((self.muscle_score['Deviation'] ** 2).mean() ** 0.5 / self.muscle_score['Sets_per_week'].mean() * 100)} % of optimal)"
                )
            if "muscles" in options:
                print(
                    self.muscle_score[
                        [
                            "Muscle",
                            "Workout_fatigue",
                            "EMA_fatigue",
                            "Sets_per_week",
                            "Target",
                            "Deviation",
                        ]
                    ].sort_values(by="Deviation", ascending=True)
                )

            self.TODAY += dt.timedelta(days=1)


if __name__ == "__main__":
    x = Engine()
    options = []
    options.append("stats")
    options.append("extras")
    # options.append("exercises")
    options.append("muscles")
    x.forecast(0, 1, options)
