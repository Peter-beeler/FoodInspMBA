import pandas as pd
import json
import random
def CsvToDict(file_path):
    rel = {}
    food_inspection_df = pd.read_csv(file_path)
    food_inspection_df.dropna(axis = 0, subset = ["License #"], inplace=True)
    results_values = food_inspection_df["Results"].value_counts()
    rel_value_map = dict((v, i) for i, v in enumerate(results_values.index)) # {'Pass': 0, 'Fail': 1, 'Pass w/ Conditions': 2, 'Out of Business': 3, 'No Entry': 4, 'Not Ready': 5, 'Business Not Located': 6}
    food_inspection_df["License #"] = food_inspection_df["License #"].astype("int")
    food_inspection_df = food_inspection_df.replace({"Results": rel_value_map})
    food_inspection_df = food_inspection_df[food_inspection_df["Results"] <= 2]
    for i in range(len(food_inspection_df)):
        inspection = food_inspection_df.iloc[i]
        if(inspection["License #"]):
            key = int(inspection["License #"])
            if(key in rel.keys()):
                rel[key].append((inspection["Inspection Date"], int(inspection["Results"])))
            else:
                rel[key] = [(inspection["Inspection Date"], int(inspection["Results"]))]
    json_file = open("food_inspection.json", "w")
    json.dump(rel, json_file)

def LoadDataDict(file_path):
    food_inspection_data = open(file_path, "r")
    data_dict = json.load(food_inspection_data)
    return data_dict

def SampleRestuarnts(from_file_path, to_file_path, sample_num = 1000):
    origin_data = LoadDataDict(from_file_path)
    keys = random.sample(list(origin_data), sample_num)
    values = [origin_data[k] for k in keys]
    samples = dict(zip(keys, values))
    json_file = open(to_file_path, "w")
    json.dump(samples, json_file)

print(SampleRestuarnts("food_inspection.json", "inspection_samples.json", 10))
# CsvToDict("Food_Inspections.csv")
# food_inspection = LoadDataDict("food_inspection.json")
# max_inspections = []
# for license in food_inspection.keys():
#     max_inspections.append(len(food_inspection[license]))
#
# from collections import Counter
# print(Counter(max_inspections))


