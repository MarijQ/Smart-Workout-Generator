# todo
# set up forecast
# workout exercise selection (muscle fatigue - 2.0 muscle use, CNS fatigue, re-score after each selection)
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
        self.read_files()
        # set up dataframes
        self.exercise_score = self.exercises_data.loc[
            self.exercises_data["Active"] == 1, ["ID", "Exercise", "Type", "Efficiency"]]
        self.exercise_score[["Freshness"]] = 0
        self.muscle_score = pd.DataFrame(list(self.exercises_data.columns[5:]), columns=["Muscle"])
        self.muscle_score[["Sets_per_week"]] = 0
        # calculate scores
        self.freshness_calc()
        self.balance_calc()
        self.muscle_target_calc()
        # select exercises for next workout
        self.new_balance_calc()
        self.overall_scores_calc()
        self.output()

    # set up lists/arrays/constants
    TODAY = dt.date.today() + dt.timedelta(days=0)
    FRESHNESS_WEIGHT = 0.5
    BALANCE_WEIGHT = 1
    EFFICIENCY_WEIGHT = 0.5

    def read_files(self):
        self.history = pd.read_csv("inputs/history.csv")
        self.exercises_data = pd.read_csv("inputs/exercises.csv")
        self.muscle_targets_data = pd.read_csv("inputs/muscle_targets.csv")

    def freshness_calc(self):
        # FRESHNESS: calculate exercise scores EMA
        for index, row in self.exercise_score.iterrows():
            # find history instances with that exercise present
            mask = np.column_stack([self.history[col] == row["ID"] for col in self.history])
            selected_history = self.history.loc[mask.any(axis=1)]
            # sum EMA contribution from each past instance
            EMA_score = 0
            for index2, row2 in selected_history.iterrows():
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
                    self.exercises_data.loc[self.exercises_data["ID"] == exercise_ID, "Type"].iloc[
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
        ## calculate balance scores
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
        ## calculate overall scores
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
        print(self.exercise_score, "\n", self.muscle_score)

    def output(self):
        ## select top 5 exercises
        print(self.exercise_score.sort_values(by="Score", ascending=False))


if __name__ == "__main__":
    Engine()
