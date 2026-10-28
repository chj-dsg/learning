import os
import fasttext

model = fasttext.load_model(os.path.expanduser("~/work/models/lid.176.bin"))

samples = [
    "a dog running on the beach",
    "一只小狗在海滩上奔跑",
    "un chien qui court sur la plage",
    "IMG_2034.JPG",
    "<PERSON> attends the premiere of the film",
]

for text in samples:
    labels, scores = model.predict(text)
    print(f"{labels[0]:<15} {scores[0]:.2f}  {text}")
