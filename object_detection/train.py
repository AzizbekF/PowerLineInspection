from ultralytics import RTDETR
import os

model = RTDETR("rtdetr-l.pt")

model.info()

yaml_path = os.path.abspath("data/InsPLAD-det/data.yaml")

results  = model.train(data=yaml_path, epochs=100, batch=8)

print(results)