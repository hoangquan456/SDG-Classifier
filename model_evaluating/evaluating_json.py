import json
from modernBert import predict_sdg

def evaluate():
    with open('output1.json', 'r') as f:
        data = json.load(f)
    
    cnt = 0
    match = 0
    
    bert_wrong_prediction = [0] * 20
    for entry in data:
        if (entry['sdg'] == None or entry['sdg'] == []):
            continue
        cnt += 1
        predict = predict_sdg(entry['abstract_cleaned'])
        if (predict in entry['sdg']):
            match += 1
        else:
            bert_wrong_prediction[predict] += 1
            print(f"""{entry['abstract_cleaned']}""")
            print(f"""Bert prediction: {predict}""")
            print(f"""LLM prediction: {entry['sdg']}""")
            print("#######")

    print(f"""Match {match} out of {cnt}""")
    print(bert_wrong_prediction)


evaluate()