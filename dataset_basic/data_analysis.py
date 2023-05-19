import re
import json

def count_words(text):
    return len(text.split(" "))

links = {"\r\n"}

input_file = "/workspace/guichi/detector/Detecting-Generated-Abstract/dataset/HC3/all.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

data_lens = []
for d in data:
    for answer in d["chatgpt_answers"]:
        sub_links = re.findall(r"[\r|\n|\t|\f|\\n]+",answer)
        sub_answer = re.split("[\r|\n|\t|\f]+",answer)
        for ans in sub_answer:
            if(count_words(ans) > 400):
                pass
                print({"ans":ans})
                print("x"*100)
            else:
                data_lens.append(count_words(ans))
        for link in sub_links:
            if "|" in link:
                print({"|":answer})
                break
        for link in sub_links:
            if link[0] == "r" or link[0] == "n" or link[0] == "t" or link[0] == "f": continue
            links.add(link)

list_links = []
for link in links:
    list_links.append([link,len(link)])
list_links.sort(key=lambda x:x[1],reverse=True)
links = [link for link,_ in list_links]
links_convert = {link:"@"+str(i)+"@" for i,link in enumerate(links)}
print(links_convert)

# 画出直方图
import matplotlib.pyplot as plt

print("max lens: ", max(data_lens))
  
plt.hist(data_lens, 40)
  
plt.savefig("data_lens.png")