import json
from transformers import AutoTokenizer
from tqdm import tqdm

def DataFormatting(input_path, output_path, task, tokenizer, low=0, high=16000):
    in_file = f'{input_path}{task}.jsonl'
    out_file = f'{output_path}{task}_{high // 1000}k.jsonl'
    skip = within_range = 0
    with open(out_file, "w", encoding="utf-8") as f_out:
        with open(in_file, "r", encoding="utf-8") as f_in:
            for idx, line in tqdm(enumerate(f_in)):
                data = json.loads(line)
                summary = data["general_query_list"][0]["answer"]
                conversations = ""
                for utterance in data["meeting_transcripts"]:
                    conversations += utterance["speaker"] + ":" + utterance["content"] + "\n"
                
                summary_tokens = tokenizer.encode(summary)
                conversations_tokens = tokenizer.encode(conversations)
                length = len(summary_tokens) + len(conversations_tokens)

                if length > high or length < low:
                    skip += 1
                    continue
                
                f_out.write(json.dumps({
                    "id": idx,
                    "conversations": conversations,
                    "summary": summary,
                    "length": length
                }, ensure_ascii=False) + "\n")
                within_range += 1
    print(f"within_range: {within_range}, skip: {skip}")

tokenizer = AutoTokenizer.from_pretrained("Yukang/LongAlpaca-7B")
input_path = "./original/"
output_path = "./meetpeft/"
tasks = ["train", "test", "val"]
for task in tasks:
    DataFormatting(input_path, output_path, task, tokenizer)