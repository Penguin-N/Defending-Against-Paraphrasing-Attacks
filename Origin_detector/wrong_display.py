import json
import pdb;
class wrong_dispalyer:
    def __init__(self, wrong_file, test_file, all_file, output_file):
        self.wrong_file = wrong_file
        self.test_file = test_file
        self.all_file = all_file
        self.output_file = output_file
        self.wrong_data = self.load_wrong_data()
        self.all_data = self.load_all_data()
    
    def load_wrong_data(self):
        with open(self.wrong_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data
    
    def load_wrong_test_data(self):
        with open(self.test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        wrong_test_data = []
        for d in self.wrong_data:
            try:
                d["question"] = data[d["id"]]["question"]
                d["text"] = data[d["id"]]["text"]
                wrong_test_data.append(d)
            except:
                print(d["id"])
                break
        return wrong_test_data
    
    def load_all_data(self):
        with open(self.all_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        all_data = {}
        for d in data:
            sample = {"chatgpt_answers": d["chatgpt_answers"],"human_answers": d["human_answers"]}
            all_data[d["question"]] = sample
        return all_data
    
    def count_word(self, text):
        return len(text.split())
    
    def save_wrong_test_data(self):
        wrong_test_data = self.load_wrong_test_data()
        store_data = []
        for d in wrong_test_data:
            chat_answers = self.all_data[d["question"]]["chatgpt_answers"]
            d["chatgpt_answers"] = chat_answers[0]
            store_data.append(d)
        with open(self.output_file, "w", encoding="utf-8") as f:
            for d in store_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                
displayer = wrong_dispalyer(wrong_file="visualize_details/pegasus_true_two_id.json", test_file="/workspace/guichi/detector/Detecting-Generated-Abstract/dataset_new/HC3-PEGASUS/test_origin.json",
                            all_file="/workspace/guichi/detector/Detecting-Generated-Abstract/dataset/HC3/all.jsonl",output_file="/workspace/guichi/detector/Detecting-Generated-Abstract/Origin_detector/visualize_details/pegasus_true_two_display.json")
displayer.save_wrong_test_data()