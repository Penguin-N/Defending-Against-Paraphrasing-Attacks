import json

from sklearn.model_selection import train_test_split

import os


def split_data(
    input_file,
    train_file,
    val_file,
    test_file,
    random_state=42,
    test_ratio=0.1,
    val_ratio=0.1,
):
    # Load data from input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Split data into training and test sets
    train_data, test_data = train_test_split(
        data, test_size=test_ratio, random_state=random_state
    )

    # Split training set into training and validation sets
    train_set, val_set = train_test_split(
        train_data, test_size=val_ratio, random_state=random_state
    )

    # Save data sets to JSON files
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_set, f, ensure_ascii=False, indent=4)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def merge_data(input_file, interdata_file, output_file, test_files, test_origin_file, test_para_file):
    # Load data from input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Create samples from data
    samples = []
    for i in range(len(data)):
        if i>=0 and i <1000: continue
        if i>=5000 and i <6000: continue
        if i>=10000 and i <11000: continue
        if i>=15000 and i <16000: continue
        if i>=20000 and i <21000: continue
        if i>=25000 and i <26000: continue
        d = data[i]
        for answer in d["human_answers"]:
            sample = {"question": d["question"], "text": answer, "type": 0}
            samples.append(sample)

        for answer in d["chatgpt_answers"]:
            sample = {"question": d["question"], "text": answer, "type": 1}
            samples.append(sample)
    
    for file in os.listdir(interdata_file):
        if file in test_files: continue
        file_path = interdata_file + "/" + file
        with open(file_path,"r",encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            for i in range(len(data)):
                d = data[i]
                samples.append(d)
    # Save samples to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Save test data to test files
    for file in test_files:
        file_path = interdata_file + "/" + file
        samples_origin = []
        samples_para = []
        with open(file_path,"r",encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            for d in data:
                samples_para.append(d)
                od = d.copy()
                od["type"] = 1
                samples_origin.append(od)
    
    with open(test_origin_file, "w", encoding="utf-8") as f:
        json.dump(samples_origin, f, ensure_ascii=False, indent=4)
    with open(test_para_file, "w", encoding="utf-8") as f:
        json.dump(samples_para, f, ensure_ascii=False, indent=4)

def main():
    input_file = "../dataset_three/HC3/all.jsonl"
    interdata_file = "../dataset_three/PEGASUS_inter_data"
    merged_file = "../dataset_three/HC3-PEGASUS/merged.jsonl"
    train_file = "../dataset_three/HC3-PEGASUS/train.json"
    val_file = "../dataset_three/HC3-PEGASUS/val.json"
    test_file = "../dataset_three/HC3-PEGASUS/test.json"
    
    # Used for evaluation
    test_files = ["0.json","5.json","10.json","15.json","20.json","25.json"]
    test_origin_file = "../dataset_three/HC3-PEGASUS/test_origin.json"
    test_para_file = "../dataset_three/HC3-PEGASUS/test_para.json"
    
    # Merge data from input file and save to output file
    merge_data(input_file, interdata_file, merged_file, test_files, test_origin_file, test_para_file)

    # # Split merged data into training, validation, and test sets and save to files
    split_data(merged_file, train_file, val_file, test_file)


if __name__ == "__main__":
    main()