import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
args = parser.parse_args()

ACC=0.
F1=0.
# file=f'../ToG/phi_jsonl/ToG_{args.dataset}.jsonl'
file=f'../../ToG/qwen_jsonl/ToG_{args.dataset}.jsonl'

pred_map = {}
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        q = item["question"].strip()
        r = item["results"].strip()
        pred_map[q] = r  # 如果有重复，只保留最后一次，也可以改成列表存多个
if args.dataset=='commonsenseqa':
    datas=[]
    with open('../../data/commonsenseqa.json',encoding='utf-8') as f:
            datas = json.load(f)
    labels=["A","B","C","D","E"]
    answer_map = {
        "A": "choices_1",
        "B": "choices_2",
        "C": "choices_3",
        "D": "choices_4",
        "E": "choices_5"
                }
    for i,item in enumerate(datas):
        la=labels.index(item['answer'])
        true_text=item['choices']['text'][la]
        true_answer = answer_map[item['answer']]
        # pred_answer = pred_answers[i]  # 对应位置的预测答案
        question_text = item['question'].strip()
        pred_answer = pred_map.get(question_text, "")  # 没找到就空字符串
        if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower() or pred_answer.lower()==true_text:
            ACC+=1
            print('chengong!')
        else:
            print(f'true:{true_answer}----pred:{pred_answer}')
    print(f"Commonsenseqa_ACC:{ACC:.4f}  F1:{F1:.4f}")

if args.dataset in 'cosmosqa':
    datas=[]
    with open('../../data/cosmosqa.json',encoding='utf-8') as f:
            datas = json.load(f)
    for i,item in enumerate(datas):
        label_idx = item["label"]
        answer_key = f"answer{label_idx}"
        true_text = item[answer_key]
        true_answer = item['answer']
        question_text = item['question'].strip()
        pred_answer = pred_map.get(question_text, "")  # 没找到就空字符串
        if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower() or pred_answer==true_text:
            ACC+=1
            print('chengong!')
        else:
            print(f'true:{true_answer}----pred:{pred_answer}')
    print(f"cosmosqa_ACC:{ACC/200:.4f}  F1:{F1/200:.4f}")

if args.dataset in 'sciq':
    datas=[]
    with open('../../data/sciq.json',encoding='utf-8') as f:
            datas = json.load(f)
    for i,item in enumerate(datas):
        true_text = item[item['answer']]
        true_answer = item['answer']
        question_text = item['question'].strip()
        pred_answer = pred_map.get(question_text, "")  # 没找到就空字符串
        if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower() or pred_answer==true_text:
            ACC+=1
            print('chengong!')
        else:
            print(f'true:{true_answer}----pred:{pred_answer}')
    print(f"cosmosqa_ACC:{ACC/200:.4f}  F1:{F1/200:.4f}")


if args.dataset in 'medqa':
    datas=[]
    with open('../../data/medqa.json',encoding='utf-8') as f:
            datas = json.load(f)
    for i,item in enumerate(datas):
        idx=item['answer_idx']
        true_text = item["options"][idx]
        true_answer = item['answer']
        question_text = item['question'].strip()
        pred_answer = pred_map.get(question_text, "")  # 没找到就空字符串
        if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower() or pred_answer==true_text:
            ACC+=1
            print('chengong!')
        else:
            print(f'true:{true_answer}----pred:{pred_answer}')
    print(f"medqa_ACC:{ACC/200:.4f}  F1:{F1/200:.4f}")

if args.dataset in 'mcqa':
    datas=[]
    with open('../../data/mcqa.json',encoding='utf-8') as f:
            datas = json.load(f)
    for i,item in enumerate(datas):
        idx=item['answer']
        true_text = item["choices"][idx]
        true_answer = item['answer']
        question_text = item['question'].strip()
        pred_answer = pred_map.get(question_text, "")  # 没找到就空字符串
        if pred_answer.lower() in true_answer.lower() or true_answer.lower() in pred_answer.lower() or pred_answer==true_text:
            ACC+=1
            print('chengong!')
        else:
            print(f'true:{true_answer}----pred:{pred_answer}')
    print(f"medqa_ACC:{ACC/200:.4f}  F1:{F1/200:.4f}")