import json

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer
)


# Definition of a custom dataset for the sequence classification task
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length=512):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Return the number of examples in the dataset
    def __len__(self):
        return len(self.inputs)

    # Return a single example and its corresponding label
    def __getitem__(self, index):
        input_ids = self.tokenizer.encode(
            self.inputs[index], add_special_tokens=True, max_length=self.max_length
        )
        label = self.labels[index]
        return input_ids, label


# Definition of a model trainer for the sequence classification task
class ModelTester:
    def __init__(
        self,
        best_model_path,
        test_files,
        model_name="roberta-base",
        batch_size=8,
        wrong_file=None,
        true_file=None,
    ):
        # Record best model path
        self.best_model_path = best_model_path
        # Record test JSON files
        self.test_files = test_files
        # Instantiate a tokenizer and a pre-trained model for sequence classification
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name).cuda()
        # Set the batch size
        self.batch_size = batch_size
        # Set the wrong file path
        self.wrong_file = wrong_file
        # Set the true file path
        self.true_file = true_file
        self.accuracy = 0.0

    # Load data from a JSON file and return a list of examples
    def load_data(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for example in data:
            text = example["text"]
            label = example["type"]
            examples.append({"text": text, "label": label})
        return examples

    # Tokenize the inputs and labels and return them as two lists
    def tokenize_inputs(self, data):
        inputs = []
        labels = []
        for example in data:
            input_ids = self.tokenizer.encode(
                example["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
            inputs.append(input_ids)
            labels.append(example["label"])
        return inputs, labels

    # Function to pad input sequences and return them in a batch
    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # Get the maximum length of the input sequences in the batch
        max_length = max(len(ids) for ids in input_ids)
        input_ids_padded = []
        attention_masks = []
        # Pad the input sequences and create attention masks
        for ids in input_ids:
            padding = [0] * (max_length - len(ids))
            input_ids_padded.append(ids + padding)
            attention_masks.append([1] * len(ids) + padding)
        # Return the inputs and labels as a dictionary and a tensor, respectively
        inputs = {
            "input_ids": torch.tensor(input_ids_padded),
            "attention_mask": torch.tensor(attention_masks),
        }
        return inputs, torch.tensor(labels)

    # Evaluate the model on a given dataloader
    def evaluate_model(self, test_loader):
        # Set the model to evaluation mode and initialize the true and predicted labels
        self.model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            # Iterate over the batches in the dataloader and get the model outputs
            for inputs, labels in test_loader:
                outputs = self.model(
                    inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                )
                # Append the true and predicted labels to their respective lists
                true_labels.extend(labels)
                predicted_labels.extend(torch.argmax(outputs.logits, dim=1).cpu())
        
        # Output the wrong predictions
        if self.wrong_file is not None:
            with open(self.wrong_file, "w", encoding="utf-8") as f:
                for i in range(len(true_labels)):
                    if true_labels[i] != predicted_labels[i]:
                        wrong_sample = {}
                        wrong_sample["id"] = i
                        wrong_sample["true_label"] = true_labels[i].item()
                        wrong_sample["predicted_label"] = predicted_labels[i].item()
                        f.write(json.dumps(wrong_sample, ensure_ascii=False) + "\n")
        
        # Output the true predictions
        if self.true_file is not None:
            with open(self.true_file, "w", encoding="utf-8") as f:
                for i in range(len(true_labels)):
                    if true_labels[i] == predicted_labels[i]:
                        true_sample = {}
                        true_sample["id"] = i
                        true_sample["true_label"] = true_labels[i].item()
                        true_sample["predicted_label"] = predicted_labels[i].item()
                        f.write(json.dumps(true_sample, ensure_ascii=False) + "\n")            
        
        # Calculate the classification report and the accuracy of the model
        report = classification_report(true_labels, predicted_labels, digits=4)
        return (
            report,
            torch.sum(torch.tensor(true_labels) == torch.tensor(predicted_labels))
            / len(true_labels),
        )

    def test(self, test_file):
        test_data = self.load_data(test_file)
        test_inputs, test_labels = self.tokenize_inputs(test_data)
        test_dataset = CustomDataset(test_inputs, test_labels, self.tokenizer)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        test_report, test_accuracy = self.evaluate_model(test_loader)
        # Print the best accuracy and the classification report for the test dataset
        
        print(f"Best accuracy: {test_accuracy:.4f}")
        print(test_report)
        self.accuracy += test_accuracy
    
    def test_all(self):
        print("Loading best model from: ", self.best_model_path)
        self.model.load_state_dict(torch.load(self.best_model_path))
        for test_file in self.test_files:
            print(f"Testing on {test_file}")
            self.test(test_file)
            print()
        print(f"Average accuracy: {self.accuracy / len(self.test_files):.4f}")

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
            if len(chat_answers[0].split(" "))<100:
                store_data.append(d)
            # if d["text"][-2:] != "..": store_data.append(d)
        with open(self.output_file, "w", encoding="utf-8") as f:
            for d in store_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    
    # test_files  = ["../dataset/HC3-Parrot/test_origin.json","../dataset/HC3-PEGASUS/test_origin.json","../dataset/HC3-ChatGPT/test_origin.json"]
    
    # test_files  = ["../dataset_new/HC3-Parrot/test_origin.json","../dataset_new/HC3-PEGASUS/test_origin.json","../dataset_new/HC3-ChatGPT/test_origin.json"]

    test_files  = ["../dataset_three/HC3/other.json","../dataset_three/HC3-Parrot/test_origin.json","../dataset_three/HC3-PEGASUS/test_origin.json","../dataset_three/HC3-ChatGPT/test_origin.json"]
    
    tester = ModelTester(
        best_model_path="training_details/best_model_three.pt", test_files=test_files
    )
    tester.test_all()



# if __name__ == "__main__":
    
#     # test_files  = ["../dataset/HC3-Parrot/test_origin.json","../dataset/HC3-PEGASUS/test_origin.json","../dataset/HC3-ChatGPT/test_origin.json"]
    
#     # test_files  = ["../dataset_new/HC3-Parrot/test_origin.json","../dataset_new/HC3-PEGASUS/test_origin.json","../dataset_new/HC3-ChatGPT/test_origin.json"]
    
#     # test_files  = ["../dataset/HC3-Para/Parrot.json","../dataset/HC3-Para/PEGASUS.json","../dataset/HC3-Para/ChatGPT.json"]
    
#     # test_files  = ["../dataset_new/HC3-Para/Parrot.json","../dataset_new/HC3-Para/PEGASUS.json","../dataset_new/HC3-Para/ChatGPT.json"]
    
#     # test_files  = ["../dataset/HC3-Para/Parrot_two.json","../dataset/HC3-Para/PEGASUS_two.json","../dataset/HC3-Para/ChatGPT_two.json"]
    
#     test_files = ["../dataset/HC3-Para/ChatGPT.json"]
    
#     # wrong_file = "visualize_details/pegasus_wrong_two_id.json"
#     # true_file = "visualize_details/pegasus_true_two_id.json"
#     # test_files  = ["../dataset/HC3-Parrot/test_origin.json","../dataset/HC3-PEGASUS/test_origin.json","../dataset/HC3-ChatGPT/test_origin.json"]

#     wrong_file = "wrongs.json"
#     true_file = None
    
#     # tester = ModelTester(
#     #     best_model_path="training_details/best_model.pt", test_files=test_files, wrong_file=wrong_file, true_file=true_file
#     # )
#     # tester.test_all()

#     all_file="/workspace/guichi/detector/Detecting-Generated-Abstract/dataset/HC3/all.jsonl"
#     output_file = wrong_file[:-8] + "_display.json"
#     displayer = wrong_dispalyer(wrong_file=wrong_file, test_file=test_files[0],
#                                 all_file=all_file,output_file=output_file)
#     displayer.save_wrong_test_data()