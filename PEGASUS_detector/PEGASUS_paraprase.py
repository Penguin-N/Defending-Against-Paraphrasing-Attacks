import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
from tqdm import tqdm
import re


links_convert = {'\\n\\n\\n\\n\\n': '@0@', '\n\n\n\n\n\n\n\n': '@1@', '\\n\\n\\n\\n': '@2@', '\r\n\r\n\r\n\r\n': '@3@', '\\n\\n\\n': '@4@', '\r\n\r\n\r\n': '@5@', '\\n\\n': '@6@', '\n\n\n\n': '@7@', '\r\n\r\n': '@8@', '\n\n\n': '@9@', '\r\n': '@10@', '\n\n': '@11@', '\\n': '@12@', '\\': '@13@', '\n': '@14@', '\t': '@15@'}
convert_links = {convert:link for link,convert in links_convert.items()}


model_name = 'tuner007/pegasus_summarizer'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

class Phase_Paraphrase:
  def __init__(self,datapath,targetpath,model,tokenizer,torch_device):
    self.datapath = datapath
    self.targetpath = targetpath
    self.predata = self.load_data(self.datapath)
    self.model = model
    self.tokenizer = tokenizer
    self.torch_device = torch_device

  def load_data(self,datapath):
    with open(datapath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # Create samples from data
    samples = {"human":[],"chatgpt":[]}
    for d in data:
        for answer in d["chatgpt_answers"]:
            sample = {"question": d["question"], "text": answer}
            samples["chatgpt"].append(sample)
    return samples

  def text_split(self,text):
    for link,convert in links_convert.items():
        text = text.replace(link,convert)
        
    segments = re.split("@\d+@",text)
    links = re.findall("@\d+@",text)
    links = [convert_links[link] for link in links]
    return segments,links

  def text_merge(self,segments,links):
    para_phrase = ""
    for i in range(len(links)):
      para_phrase += segments[i] + links[i]
    para_phrase += segments[-1]
    return para_phrase 
  
  def get_response(self, input_text):
    batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=1024, return_tensors="pt").to(self.torch_device)
    gen_out = self.model.generate(**batch,max_length=400,num_beams=5, num_return_sequences=1, temperature=1.5)
    output_text = self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text

  def paraphrase(self):
    max_num = len(self.predata["chatgpt"]) // 1000
    num = 26
    print(f"from {num} to {max_num}")
    while(num<=max_num):
      length = 1000
      if (num==max_num): length = len(self.predata["chatgpt"])-num*1000
      inter_data = []
      print("Paraphrase Epoch ", num, end=": ")
      for i in tqdm(range(length)):
        check = True
        d = self.predata["chatgpt"][i+num*1000]
        segments,links = self.text_split(d["text"])
        for segment in segments:
          if len(segment.split(" "))>400 or "|" in segment:
            check = False
            break
        if(not check): continue
        paraphrased_texts = []
        for text in segments:
          paraphrased_texts.append(self.get_response(text)[0])
        para_phrase = self.text_merge(paraphrased_texts,links)
        sample = {"question":d["question"], "text":para_phrase, "type":2}
        inter_data.append(sample)
      output_path = self.targetpath + "/" +str(num)+".json"
      with open(output_path, "w", encoding="utf-8") as f:
        for sample in inter_data:
          f.write(json.dumps(sample, ensure_ascii=False) + "\n") 
      num+=1
    

datapath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset_new/HC3/all.jsonl"
targetpath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset_new/PEGASUS_inter_data"
paraphrase = Phase_Paraphrase(datapath,targetpath,model,tokenizer,torch_device)
paraphrase.paraphrase()

