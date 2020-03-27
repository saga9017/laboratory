import pickle


with open('gpt_xlnet_preds/gpt_preds_top5.pkl', 'rb') as f:
    data = pickle.load(f)

with open('gpt_xlnet_preds/xlnet_preds_top5.pkl', 'rb') as f:
    data2 = pickle.load(f)

with open('data.pickle', 'rb') as f:
    data3 = pickle.load(f)

print(data)
print(data2)
print(data3)