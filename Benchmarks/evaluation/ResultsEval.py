import os
import json
import fire
import torch
from glob import glob
from tqdm import tqdm
from BleuScore import CocoScorer
# from summertime.evaluation import Rouge, RougeWe, BertScore, Bleu, Meteor
# from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.rouge.rouge import Rouge
from rouge import Rouge
from torchtext.data.metrics import bleu_score
import bert_score.score as Bert_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize

from MoverScore import MoverScore

# METRICS = [Rouge(), RougeWe(), BertScore(), Bleu(), Meteor()]
# METRICS = [Rouge(), Bleu(), Meteor()]

def load_split_data(fpath):
    """
        Load split data: train, test, dev --> List()
        Data structure:
            {id:(str) , summary:(str), source:(str),}
    """
    tmp = []
    with open(fpath, 'r') as r:
        for line in r:
            print(line)
            tmp.append(json.loads(line))
    return tmp

def run_evaluation(model_summaries, tgt, BlockList=[]):
    """
        Run all metrics for summarization results
    """

    # print("evaluating...")
    # for metric in METRICS:
    #     print(f"evaluating {metric.metric_name}...")
    #     if metric.metric_name in BlockList:
    #         continue
    #     results = metric.evaluate(model_summaries, tgt)
    #     print(f"{metric.metric_name} results: {results}")
    #     result[metric.metric_name] = results

    result = {}

    # bleu score
    scorer = CocoScorer(tgt, model_summaries)
    score = scorer.compute_scores()
    for i in range(1,len(score["Bleu"])+1):
        result[f"Bleu@{i}"] = score["Bleu"][i-1]

    # rouge score
    rouge = Rouge()
    result['rouge'] = rouge.get_scores(model_summaries, tgt, avg=True)
    # result['rouge'].append(rouge.get_scores(pred, target)[0])
    print(f"rouge results: {result['rouge']}")

    meteor_scores = []
    for target, pred in zip(tgt, model_summaries):

        # # rouge score
        # if "rouge" not in BlockList:
        #     rouge = Rouge()
        #     result['rouge'].append(rouge.get_scores(pred, target)[0])
        #     print(f"rouge results: {result['rouge']}")
        
        # bleu score
        #if "bleu" not in BlockList:
            #result['bleu'].append(bleu_score(pred, target))
            #print(f"bleu results: {result['bleu']}")
            # BleuScore module will handle the printing, 
            # no need to print here
        
        # meteor score
        meteor_scores.append(meteor_score([pred], target))
        # print(f"meteor results: {result['meteor']}")
    
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    result['meteor'] = avg_meteor

    # bert score
    P, R, F1 = Bert_score(model_summaries, tgt, lang="en", verbose=True)
    # convert from tensor to list
    P = torch.mean(P).item()
    R = torch.mean(R).item()
    F1 = torch.mean(F1).item()
    result['bertscore'] = {"P":P, "R":R, "F1":F1}
    print(f"bertscore results: {result['bertscore']}")
        
    result['MoverScore'] = MoverScore(model_summaries, tgt)
    print(f"MoverScore results: {result['MoverScore']}")
    
    # print(result)
    # write results to file
    file_path = "Benchmarks/LongAlpaca-7B/evals/eval_QM_16k.json"
    with open(file_path, "w") as w:
        json.dump(result, w)

    return result

# run evaluation by summertime
def get_tgt_pred(temp_data):
    ids = []
    tgt_list = []
    pred_list = []
    print("loading data...")
    for ins in tqdm(temp_data):
        if "id" in ins.keys():
            ids.append(ins['id'])
        else:
            ids = []
        if "target" in ins.keys():
            if not ins['target'].strip():
                continue
            elif not ins['prediction'].strip():
                continue
            tgt_list.append(ins['target'])
            pred_list.append(ins['prediction'])
        elif "summary" in ins.keys():
            tgt_list.append(ins['summary'])
            pred_list.append(ins['prediction'])
        elif "ModelPrediction" in ins.keys():
            tgt_list.append(ins["GroundTruth"])
            pred_list.append(ins["ModelPrediction"])
    return {"ids":ids, "tgt":tgt_list, "pred":pred_list}

def run_eval(fpath):
    data_name = os.path.basename(fpath).split(".")[0]
    eval_data = get_tgt_pred(load_split_data(fpath))
    return {data_name: run_evaluation(eval_data['pred'], eval_data['tgt'])}  

if __name__ == "__main__":
    # data_path = "Benchmarks/LongAlpaca-7B/output/output_MB_segment_16k_100.jsonl"
    # data_path = "Benchmarks/Llama-2-7B/output/output_MB_2k_2.jsonl"
    data_path = "Benchmarks/LongAlpaca-7B/output/output_QM_16k.jsonl"
    run_eval(data_path)

    