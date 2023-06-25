# todo
# workout exercise selection (CNS fatigue, during workout and EMA over time)
# muscular fatigue (not flat 2.0, reduced custom to muscle based on higher frequency 50% EMA muscle usage)
# set up forecast
# write to history file (function with input)
# CNS fatigue (tag exercises, current EMA, workout cap)
# Injury

# imports
import pandas as pd
import numpy as np
import datetime as dt

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 400)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_columns', 20)


class Engine():
    def __init__(self):
        self.set_up()
        self.generate_workout(self.TODAY)
        # self.output()

    # set up lists/arrays/constants
    TODAY = dt.date.today() + dt.timedelta(days=0)
    FRESHNESS_WEIGHT = 0
    BALANCE_WEIGHT = 1
    EFFICIENCY_WEIGHT = 0.5
    PIPELINE = []

    def set_up(self):
        self.history = pd.read_csv("inputs/history.csv")
        self.exercises_data = pd.read_csv("inputs/exercises.csv")
        self.muscle_targets_data = pd.read_csv("inputs/muscle_targets.csv")
        # set up dataframes
        self.exercise_score = self.exercises_data.loc[
            self.exercises_data["Active"] == 1, ["ID", "Exercise", "Type", "Efficiency"]]
        self.exercise_score[["Freshness"]] = 0
        self.muscle_score = pd.DataFrame(list(self.exercises_data.columns[5:]), columns=["Muscle"])
        self.muscle_score[["Sets_per_week"]] = 0
        self.muscle_score[["Workout_fatigue"]] = 0
        self.muscle_score[["Temp_fatigue"]] = 0

    def freshness_calc(self):
        # FRESHNESS: calculate exercise scores EMA
        for index, row in self.exercise_score.iterrows():  # Loop through all exercises
            # find history instances with that exercise present
            mask = np.column_stack([self.history[col] == row["ID"] for col in self.history])
            selected_history = self.history.loc[mask.any(axis=1)]
            # sum EMA contribution from each past instance
            EMA_score = 0
            for index2, row2 in selected_history.iterrows():  # Loop through history records
                date_diff = (
                        self.TODAY - dt.datetime.strptime(row2["Date"], '%d/%m/%Y').date()).days
                EMA_score += 0.1 * 1 * 0.9 ** date_diff
            self.exercise_score.loc[index, ["Freshness"]] = round(EMA_score * 100, 0)

    def balance_calc(self):
        # BALANCE: calculate muscle sets per week EMA
        for index, row in self.history.iterrows():  # loop through history records
            for i in range(10):  # loop through exercises completed
                try:
                    int(row[i + 1])
                except ValueError:
                    continue
                exercise_ID = int(row[i + 1])
                for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
                    # if index2 != 14:
                    #     continue
                    score = \
                        self.exercises_data.loc[
                            self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                    try:
                        int(score)
                    except ValueError:
                        continue
                    if \
                            self.exercises_data.loc[
                                self.exercises_data["ID"] == exercise_ID, "Type"].iloc[
                                0] == "Cardio":
                        score = float(score) / 2
                    else:
                        score = float(score)
                    date_diff = (self.TODAY - dt.datetime.strptime(row["Date"],
                                                                   '%d/%m/%Y').date()).days
                    EMA_score = 0.1 * score * 0.9 ** date_diff
                    self.muscle_score.loc[index2, ["Sets_per_week"]] += EMA_score * 3 * 7
                    # print(row["Date"], exercise_ID, score, date_diff, EMA_score)

    def muscle_target_calc(self):
        # calculate muscle EMA target
        self.muscle_score[["Target"]] = self.muscle_targets_data[["Priority"]]
        fn = lambda row: row["Target"] / self.muscle_score["Target"].sum() * (
                5 * 5 * 3 * 2.6)  # workouts/week * exercises * sets * avg muscles
        self.muscle_score["Target"] = self.muscle_score.apply(fn, axis=1)
        fn2 = lambda row: max(0, row["Target"] - row["Sets_per_week"])
        self.muscle_score["Balance"] = self.muscle_score.apply(fn2, axis=1)

    def new_balance_calc(self):
        ## calculate projected balance scores for each exercise
        for index, row in self.exercise_score.iterrows():  # loop through all (active) exercises
            self.muscle_score[["New_balance"]] = self.muscle_score[["Balance"]]
            exercise_ID = row["ID"]
            for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
                # if index2 != 14:
                #     continue
                score = \
                    self.exercises_data.loc[
                        self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                if self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[
                    0] == "Cardio":
                    score = float(score) / 2
                else:
                    score = float(score)
                new_balance = max(0, row2["Balance"] - 0.1 * score * 3 * 7)
                self.muscle_score.loc[index2, ["New_balance"]] = new_balance
            RMS = (self.muscle_score["New_balance"] ** 2).sum()
            self.exercise_score.loc[index, ["Balance"]] = RMS

    def overall_scores_calc(self):
        ## calculate overall scores for each exercise
        self.exercise_score[["Score"]] = 0
        freshness_best = self.exercise_score["Freshness"].min()
        freshness_worst = self.exercise_score["Freshness"].max()
        balance_best = self.exercise_score["Balance"].min()
        balance_worst = self.exercise_score["Balance"].max()
        efficiency_best = self.exercise_score["Efficiency"].max()
        efficiency_worst = self.exercise_score["Efficiency"].min()
        fn3 = lambda row: max(0,
                              (1 + (row["Freshness"] - freshness_worst) / (
                                      freshness_best - freshness_worst))
                              ** self.FRESHNESS_WEIGHT *
                              (1 + (row["Balance"] - balance_worst) / (
                                      balance_best - balance_worst))
                              ** self.BALANCE_WEIGHT *
                              (1 + (row["Efficiency"] - efficiency_worst) / (
                                      efficiency_best - efficiency_worst))
                              ** self.EFFICIENCY_WEIGHT)
        self.exercise_score["Score"] = self.exercise_score.apply(fn3, axis=1)
        # print(self.exercise_score, "\n", self.muscle_score)

    def generate_workout(self, date):
        # refresh score baselines from history
        self.freshness_calc()
        self.balance_calc()
        self.muscle_target_calc()
        # calculate projected scores for exercises
        self.new_balance_calc()
        self.overall_scores_calc()
        # reset workout fatigue values
        self.muscle_score[["Workout_fatigue"]] = 0
        print("Old total balance:", (self.muscle_score["Balance"] ** 2).sum())
        # select lifts exercise
        for i in range(4):
            self.select_top_exercise("Lift")
        for i in range(1):
            self.select_top_exercise("Cardio")
        print("New total balance:", (self.muscle_score["Balance"] ** 2).sum())
        print("Extras:")
        for i in range(5):
            self.select_top_exercise("Lift")

    def select_top_exercise(self, type):
        # Filter for muscle fatigue limits
        for index, row in self.exercise_score.iterrows():  # loop through all exercises
            self.muscle_score[["Temp_fatigue"]] = 0
            for index2, row2 in self.muscle_score.iterrows():  # add fatigue for each muscle
                score = \
                    self.exercises_data.loc[
                        self.exercises_data["ID"] == row["ID"], row2["Muscle"]].iloc[0]
                try:
                    int(score)
                except ValueError:
                    continue
                if self.exercises_data.loc[self.exercises_data["ID"] == row["ID"], "Type"].iloc[
                    0] == "Cardio":
                    score = float(score) / 2
                else:
                    score = float(score)
                self.muscle_score.loc[index2, ["Temp_fatigue"]] += score
                expected_fatigue = self.muscle_score.loc[index2, ["Workout_fatigue"]].iloc[0] + \
                                   self.muscle_score.loc[index2, ["Temp_fatigue"]].iloc[0]
                if expected_fatigue > 2:
                    self.exercise_score.drop(index, inplace=True)
                    # print("Dropped", row["Exercise"], " > ", row2["Muscle"], expected_fatigue)
                    break

        # Find top exercise (lift)
        row = self.exercise_score[self.exercise_score["Type"] == type].sort_values(by="Score",
                                                                                   ascending=False).head(
            1)
        exercise_index = row.index.tolist()[0]

        # Update muscle balance + fatigue
        self.muscle_score[["Temp_fatigue"]] = 0
        self.muscle_score[["New_balance"]] = self.muscle_score[["Balance"]]
        exercise_ID = row["ID"].iloc[0]
        for index2, row2 in self.muscle_score.iterrows():  # add EMA contribution for each muscle
            score = \
                self.exercises_data.loc[
                    self.exercises_data["ID"] == exercise_ID, row2["Muscle"]].iloc[0]
            try:
                int(score)
            except ValueError:
                continue
            if self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[
                0] == "Cardio":
                score = float(score) / 2
            else:
                score = float(score)
            new_balance = max(0, row2["Balance"] - 0.1 * score * 3 * 7)
            self.muscle_score.loc[index2, ["New_balance"]] = new_balance
            self.muscle_score.loc[index2, ["Temp_fatigue"]] += score
        self.muscle_score["Balance"] = self.muscle_score["New_balance"]
        self.muscle_score["Workout_fatigue"] += self.muscle_score["Temp_fatigue"]

        # Write to list and remove from exercises
        self.PIPELINE.append(row["ID"].iloc[0])
        self.exercise_score.drop([exercise_index], inplace=True)
        print("Selected", row["ID"].iloc[0], row["Exercise"].iloc[0])

        # Refresh projected scores for exercises
        self.new_balance_calc()
        self.overall_scores_calc()

    def output(self):
        ## select top 5 exercises
        print(self.exercise_score.sort_values(by="Score", ascending=False))


if __name__ == "__main__":
    Engine()
