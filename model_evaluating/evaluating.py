import time, csv
# from api_call import predict_sdg_multilabel as predict_sdg
from modernBert import predict_sdg
from itertools import islice

NUM = 200000 #number of samples
filename = 'osdg-community-data-v2024-04-01.csv'

def single_label_evaluation():
    start_time = time.time()
    count_match = 0
    total = 0
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter = '\t')
        for row in reader:
            if (float(row['agreement']) < 0.6 or int(row['labels_negative']) + int(row['labels_positive']) <= 5 or int(row['labels_negative']) > int(row['labels_positive'])): 
                continue
            
            predict = predict_sdg(row['text'])
            if int(row['sdg']) == predict: 
                count_match += 1  
            #     print(predict)
            # else:
            #     print(row['text'])
            #     print(f"prediction: {predict}")
            #     print(f"actual {int(row['sdg'])}")

            total += 1
            # print(total)
            if (NUM == total):
                break
    print(f"match {count_match} out of {total}")
    end_time = time.time()
    print(f"running time: {end_time - start_time:.2f}")

def multilabel_evaluation():
    start_time = time.time()
    count_match = 0
    total = 0
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter = '\t')
        for row in reader:
            if (float(row['agreement']) < 0.65 or int(row['labels_negative']) + int(row['labels_positive']) <= 5 or int(row['labels_negative']) > int(row['labels_positive'])): 
                continue
            total += 1
            print(total)

            predict = predict_sdg(row['text'])

            while (predict != None and predict[0] == -1):
                print("temporarily time out for 10 seconds")
                time.sleep(10)
                print("continue running")
                predict = predict_sdg(row['text'])

            if int(row['sdg']) in predict: 
                count_match += 1  
                print(predict)
            else:
                print(row['text'])
                print(f"prediction: {predict}")
                print(f"actual {int(row['sdg'])}")

            if (NUM == total):
                break
            print()

    print(f"match {count_match} out of {total}")
    end_time = time.time()
    print(f"running time: {end_time - start_time:.2f}")

single_label_evaluation()