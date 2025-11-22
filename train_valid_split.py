import random

valid_percent = 0.2
RANDOM_SAMPLE_FLAG = False
NUM_IMG = 700
DATA_PATH  = 'data/alldata.txt'
VALID_PATH = 'data/valid.txt'
TRAIN_PATH = 'data/train.txt'

with open(DATA_PATH, 'r') as inf:
    lines = [line.rstrip() for line in inf if len(line) > 0]

### take a random sample of NUM_IMG
if RANDOM_SAMPLE_FLAG:
    lines = random.sample(lines, NUM_IMG)    

valid = set(random.sample(lines, int(len(lines)*valid_percent)))
alldata = set(lines)
train = alldata.difference(valid)

#print(len(train))
#print(len(valid))

assert len(alldata) == len(valid) + len(train)

print('all data has   {} items'.format(len(alldata)))
print('valid data has {} items'.format(len(valid)))
print('train data has {} items'.format(len(train)))

with open(VALID_PATH, 'w') as valid_outf:
    for ln in valid:
        valid_outf.write(ln + '\n')

with open(TRAIN_PATH, 'w') as train_outf:
    for tr in train:
        train_outf.write(tr + '\n')
    
