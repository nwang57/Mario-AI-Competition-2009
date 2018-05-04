import sys
import pickle
log_file = sys.argv[1]
save_file = sys.argv[2]
parsed_lines = []

def parse_num(s, start, end):
    legal = set('1234567890.-')
    num = ''
    for c in s[start:end]:
        if c in legal:
            num += c
        else:
            if len(num) > 0:
                # num ended
                return float(num)
            else:
                # skip it
                continue
    return None

with open(log_file) as f:
    for line in f.readlines():
        inds = [line.find('len '), line.find('rewards: '), line.find('avg_reward: '), line.find('eps: ')]
        if min(inds) > -1:
            inds.append(len(line))
            pairs = zip(inds, inds[1:])
            parsed_lines.append([parse_num(line, i, j) for i,j in pairs])


print(parsed_lines)

with open(save_file, 'wb') as f:
    pickle.dump(parsed_lines, f)