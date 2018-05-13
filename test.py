import json

with open('/home/retty/Datasets/caltech/data/annotations.json', 'r') as f:
    j = json.load(f)

# for key in j['set00']['V000']['frames']:
#     print(key)
    # altered log nFrame logLen frames maxObj

print(j['set00']['V000']['frames']['344'])
