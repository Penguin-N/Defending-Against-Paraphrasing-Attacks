import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


links_convert = {'\\n\\n\\n\\n\\n': '@0@', '\n\n\n\n\n\n\n\n': '@1@', '\\n\\n\\n\\n': '@2@', '\r\n\r\n\r\n\r\n': '@3@', '\\n\\n\\n': '@4@', '\r\n\r\n\r\n': '@5@', '\\n\\n': '@6@', '\n\n\n\n': '@7@', '\r\n\r\n': '@8@', '\n\n\n': '@9@', '\r\n': '@10@', '\n\n': '@11@', '\\n': '@12@', '\\': '@13@', '\n': '@14@', '\t': '@15@'}
convert_links = {convert:link for link,convert in links_convert.items()}

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


class ChatGPT_Paraphrase:
  def __init__(self,datapath,targetpath,model,tokenizer,device):
    self.datapath = datapath
    self.targetpath = targetpath
    self.predata = self.load_data(self.datapath)
    self.model = model
    self.tokenizer = tokenizer
    self.device = device

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
  
  def get_response(self,
      question,
      num_beams=2,
      num_beam_groups=2,
      num_return_sequences=1,
      repetition_penalty=10.0,
      diversity_penalty=3.0,
      no_repeat_ngram_size=2,
      temperature=0.7,
      max_length=400
  ):
      input_ids = tokenizer(
          f'paraphrase: {question}',
          return_tensors="pt", padding="longest",
          max_length=max_length,
          truncation=True,
      ).input_ids
      
      input_ids = input_ids.to(device)
      
      outputs = model.generate(
          input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
          num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
          num_beams=num_beams, num_beam_groups=num_beam_groups,
          max_length=max_length, diversity_penalty=diversity_penalty
      )

      res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

      return res

  def paraphrase(self):
    # max_num = 22
    max_num = len(self.predata["chatgpt"]) // 1000
    num = 23
    print(f"From {num}-{max_num}")
    while(num<=max_num):
      inter_data = []
      print("Paraphrase Epoch ", num, end=": ")
      length = 1000
      if num == max_num:
        length = len(self.predata["chatgpt"]) - num*1000
      for i in tqdm(range(length)):
        if(i+num*1000 > len(self.predata["chatgpt"])): break
        d = self.predata["chatgpt"][i+num*1000]
        text = d["text"].replace("/n/n", "/n")
        text = text.replace("/r/n/r/n", "/n")
        text = text.replace("/r", "")
        Modified_texts = text.split("/n")
        paraphrased_texts = []
        for text in Modified_texts:
          paraphrased_texts.append(self.get_response(text)[0])
        para_phrase = " \n".join(paraphrased_texts)
        sample = {"question":d["question"], "text":para_phrase, "type":2}
        inter_data.append(sample)
      output_path = self.targetpath + "/" +str(num)+".json"
      with open(output_path, "w", encoding="utf-8") as f:
        for sample in inter_data:
          f.write(json.dumps(sample, ensure_ascii=False) + "\n") 
      num+=1





  # def paraphrase(self):
  #   print("From 21-26")
  #   max_num = len(self.predata["chatgpt"]) // 1000
  #   num = 21
  #   while(num<=max_num):
  #     inter_data = []
  #     print("Paraphrase Epoch ", num, end=": ")
  #     length = 1000
  #     if num == max_num:
  #       length = len(self.predata["chatgpt"]) - num*1000
  #     for i in tqdm(range(length)):
  #       if(i+num*1000 > len(self.predata["chatgpt"])): break
  #       check = True
  #       d = self.predata["chatgpt"][i+num*1000]
  #       segments,links = self.text_split(d["text"])
  #       for segment in segments:
  #         if len(segment.split(" "))>400 or "|" in segment:
  #           check = False
  #           break
  #       if(not check): continue
  #       paraphrased_texts = []
  #       for text in segments:
  #         paraphrased_texts.append(self.get_response(text)[0])
  #       para_phrase = self.text_merge(paraphrased_texts,links)
  #       sample = {"question":d["question"], "text":para_phrase, "type":2}
  #       inter_data.append(sample)
  #     output_path = self.targetpath + "/" +str(num)+".json"
  #     with open(output_path, "w", encoding="utf-8") as f:
  #       for sample in inter_data:
  #         f.write(json.dumps(sample, ensure_ascii=False) + "\n") 
  #     num+=1

datapath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset/HC3/all.jsonl"
targetpath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset/ChatGPT_inter_data"
paraphrase = ChatGPT_Paraphrase(datapath,targetpath,model,tokenizer,device)
paraphrase.paraphrase()

