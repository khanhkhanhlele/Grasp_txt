import os
import json
import pickle

template = {
    "id": "unique_id",
    "image": "image_file.jpg",
    "conversations": [
        {

            "from": "human",
            "value": "What is shown in the image?"

        },
        {
            "from": "gpt",
            "value": "formatted_answers"
        }
    ]
}

datapoints = []

for txt_file in os.listdir("grasp-anything++/seen/txt_label"):
    with open(f"grasp-anything++/seen/txt_label/{txt_file}", "r") as f:
        gpt_value = f.read().strip()
    
    with open(f"grasp-anything++/seen/grasp_instructions/{txt_file.split('.')[0]+'.pkl'}", "rb") as f:
        instruction = pickle.load(f)
        
    new_datapoint = {}
    new_datapoint["id"] = txt_file.split('.')[0]
    new_datapoint["image"] = txt_file.split('_')[0] + ".jpg"
    if not os.path.exists(f"grasp-anything++/seen/image/{new_datapoint['image']}"):
        print("Image not found: ", new_datapoint["image"])
        continue
    new_datapoint["conversations"] = []
    new_datapoint["conversations"].append({"from": "human", "value": "<image>\n" +instruction})
    new_datapoint["conversations"].append({"from": "gpt", "value": gpt_value})
    datapoints.append(new_datapoint)
    
with open("dataset/train/dataset.json", "w") as f:
    json.dump(datapoints, f)