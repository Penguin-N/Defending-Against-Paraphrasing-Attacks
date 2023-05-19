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
        accuracy=0.0,
    ):
        # Record best model path
        self.best_model_path = best_model_path
        # Record test JSON files
        self.test_files = test_files
        # Instantiate a tokenizer and a pre-trained model for sequence classification
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()
        # Set the batch size
        self.batch_size = batch_size
        self.accuracy = accuracy

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
        
        # Save the true and predicted labels to a file
        with open("true_labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([str(label) for label in true_labels]))
        with open("predicted_labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([str(label) for label in predicted_labels]))
        
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
        print(f"Average accuracy: {self.accuracy/4:.4f}")



if __name__ == "__main__":
    
    # test_files  = ["../dataset/HC3-Parrot/test_para.json","../dataset/HC3-PEGASUS/test_para.json","../dataset/HC3-ChatGPT/test_para.json"]
    
    # test_files  = ["../dataset_new/HC3-Parrot/test_para.json","../dataset_new/HC3-PEGASUS/test_para.json","../dataset_new/HC3-ChatGPT/test_para.json"]

    test_files  = ["../dataset_three/HC3/other.json","../dataset_three/HC3-Parrot/test_para.json","../dataset_three/HC3-PEGASUS/test_para.json","../dataset_three/HC3-ChatGPT/test_para.json"]
    
    tester = ModelTester(
        best_model_path="training_details/best_model_three.pt", test_files=test_files
    )
    tester.test_all()
