import os
import json
import random




def merge_data(inter_file,output_file):
    all_data = []
    for file in os.listdir(inter_file):
        file_path = inter_file + "/" + file
        with open(file_path,"r",encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            for i in range(len(data)):
                d = data[i]
                d["type"] = 1
                all_data.append(d)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
                
if __name__ == "__main__":
    inter_files = ["Parrot_inter_data","PEGASUS_inter_data","ChatGPT_inter_data"]
    output_dir = "HC3-Para"
    ourput_files = ["Parrot.json","PEGASUS.json","ChatGPT.json"]
    for i in range(len(inter_files)):
        inter_file = inter_files[i]
        output_file = output_dir + "/" + ourput_files[i]
        merge_data(inter_file,output_file)