import csv

csv_file = "model_metrics.csv"

models = []
with open(csv_file, mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        models.append({"Model Name": row["Model Name"], "FLOPs": float(row["FLOPs"])})

models = sorted(models, key=lambda x: x["FLOPs"])

print("Top 10 Fastest Models (Based on FLOPs):")
for i, model in enumerate(models[:], start=1):
    print(f"{i}. {model['Model Name']} - FLOPs: {model['FLOPs']}")
