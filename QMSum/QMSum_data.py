import json

input_path = "./data/ALL/jsonl/"
output_path = "./MEETPEFT_data/ALL/jsonl/"
tasks = ["train", "test", "val"]

def DataFormatting(in_file, out_file):
    with open(out_file, "w", encoding="utf-8") as f_out:
        with open(in_file, "r", encoding="utf-8") as f_in:
            for idx, line in enumerate(f_in):
                data = json.loads(line)
                summary = data["general_query_list"][0]["answer"]
                conversations = ""
                for utterance in data["meeting_transcripts"]:
                    conversations += utterance["speaker"] + ":" + utterance["content"] + "\n"
                f_out.write(json.dumps({
                    "id": idx,
                    "conversations": conversations,
                    "summary": summary
                }, ensure_ascii=False) + "\n")

for task in tasks:
    in_file = input_path + task + ".jsonl"
    out_file = output_path + task + ".jsonl"
    DataFormatting(in_file, out_file)