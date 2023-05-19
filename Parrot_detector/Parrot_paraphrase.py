import json
from tqdm import tqdm
from parrot import Parrot
import torch
import warnings
import re
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''


links_convert = {'\\n\\n\\n\\n\\n': '@0@', '\n\n\n\n\n\n\n\n': '@1@', '\\n\\n\\n\\n': '@2@', '\r\n\r\n\r\n\r\n': '@3@', '\\n\\n\\n': '@4@', '\r\n\r\n\r\n': '@5@', '\\n\\n': '@6@', '\n\n\n\n': '@7@', '\r\n\r\n': '@8@', '\n\n\n': '@9@', '\r\n': '@10@', '\n\n': '@11@', '\\n': '@12@', '\\': '@13@', '\n': '@14@', '\t': '@15@'}
convert_links = {convert:link for link,convert in links_convert.items()}


#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

class T5_Paraphrase:
  def __init__(self,datapath,targetpath,parrot):
    self.datapath = datapath
    self.targetpath = targetpath
    self.parrot = parrot
    self.predata = self.load_data(self.datapath)

  def load_data(self,datapath):
    with open(datapath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # Create samples from data
    samples = {"human":[],"chatgpt":[]}
    for d in data:
        for answer in d["human_answers"]:
            sample = {"question": d["question"], "text": answer}
            samples["human"].append(sample)

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

  def paraphrase(self):
    max_num = len(self.predata["chatgpt"]) // 1000
    num = 21
    max_num = 6
    num = 4
    print(f"from {num} to {max_num}")
    while(num<=max_num):
      length = 1000
      # if (num==max_num): length = len(self.predata["chatgpt"])-num*1000
      inter_data = []
      print("Paraphrase Epoch ", num, end=": ")
      for i in tqdm(range(length)):
        if(i+num*1000 > len(self.predata["chatgpt"])): break
        d = self.predata["chatgpt"][i+num*1000]
        check = True
        sections,links = self.text_split(d["text"])
        # We only want to paraphrase sections that are shorter than 400 characters
        for section in sections:
          if len(section.split(" "))>400 or "|" in section:
            check = False
            break
        if(not check): continue
        sentences = [section.split(".") for section in sections]
        paraprahse_sections_1 = []
        paraprahse_sections_2 = []
        paraprahse_sections_3 = []
        for section in sentences:
          paraphrase_sentences = []
          for sentence in section:
            if sentence == "" or sentence == '' : continue
            para_phrases = parrot.augment(input_phrase=sentence,
                                            max_length=400,
                                              use_gpu=True,
                                              do_diverse=True,             # Enable this to get more diverse paraphrases
                                              adequacy_threshold = 0.50,   # Lower this numbers if no paraphrases returned
                                              fluency_threshold = 0.80,
                                              max_return_phrases = 3)
            if(para_phrases is None and len(section)==1):
              check = False
              break
            elif(para_phrases is None or len(para_phrases)<3 ):
              continue
            else:
              intermidiate = []
              for para_phrase in para_phrases[0:3]:
                if("." in para_phrase[0] or "?" in para_phrase[0] or "!" in para_phrase[0]):
                  intermidiate.append(para_phrase[0][0].upper()+para_phrase[0][1:])
                else:
                  intermidiate.append(para_phrase[0][0].upper()+para_phrase[0][1:]+".")     
              paraphrase_sentences.append(intermidiate)
          if(check == False):
            break
          paraprahse_sections_1.append(" ".join([p[0] for p in paraphrase_sentences]))
          paraprahse_sections_2.append(" ".join([p[1] for p in paraphrase_sentences]))
          paraprahse_sections_3.append(" ".join([p[2] for p in paraphrase_sentences]))
        if(check == True):
          para_phrases = [self.text_merge(paraprahse_sections_1,links),self.text_merge(paraprahse_sections_2,links),self.text_merge(paraprahse_sections_3,links)]
        else:
          continue        
        for para_phrase in para_phrases:
          if(para_phrase == ""):
            continue
          sample = {"question":d["question"], "text":para_phrase, "type":2}
          inter_data.append(sample)
      
      output_path = self.targetpath + "/" +str(num)+".json"
      with open(output_path, "w", encoding="utf-8") as f:
        for sample in inter_data:
          f.write(json.dumps(sample, ensure_ascii=False) + "\n") 
      num+=1
    

datapath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset_new/HC3/all.jsonl"
targetpath = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset_new/Parrot_inter_data"
paraphrase = T5_Paraphrase(datapath,targetpath,parrot)
paraphrase.paraphrase()