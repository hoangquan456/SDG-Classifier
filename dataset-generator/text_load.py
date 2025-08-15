import pickle

match =[]

for i in range(10, 20):
    print(i)
    with open(f"output{i}.pkl", "rb") as file:
        match.extend(pickle.load(file))

filtered_match = [entry for entry in match if len(entry) > 700]
print(len(filtered_match))
print(filtered_match[0])
print("#########")
print(filtered_match[-1])
with open("not_sdg1_pro.pkl", "wb") as file:
    pickle.dump(filtered_match, file)