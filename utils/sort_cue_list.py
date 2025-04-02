import json
import os
import itertools

cue_path1 = '../data/train/cue_json'
cue_path2 = '../data/val/cue_json'
timecode_path1 = '../data/train/timecode_json'
timecode_path2 = '../data/val/timecode_json'
all_cue_path = '../data'
timecode_output_path = '../data/processed_timecode'

cnt = 0

final_cue_list = []

for item in os.listdir(cue_path1):
    file_name = item[:-5]
    print(f"Processing: {file_name}")
    
    with open(os.path.join(cue_path1, item), 'r') as f:
        cue = json.load(f)
    
    with open(os.path.join(timecode_path1, file_name + '.json'), 'r') as f:
        timecode = json.load(f)
        timecode = timecode[0]['Subtrack'][0]['Event']

    cue_list, timecode_list = [], []
    for tc in timecode:
        if 'cue' in tc.keys():
            specific_cue = tc['cue'][2]
        for c in cue:
            if c[0]['Number']['number'] == specific_cue:
                if 'CueDatas' not in c[0].keys():
                    continue
                cue_list.append(c[0]['CueDatas'])
                
                break
    
    for item1 in cue_list:
        cnt += 1
        final_cue_item = {}
        final_cue_item['Num'] = cnt
        timecode_list.append(cnt)
        final_cue_item['CueDatas'] = item1
        final_cue_list.append(final_cue_item)

    os.makedirs(os.path.join(timecode_output_path),exist_ok=True)
    with open(os.path.join(timecode_output_path, file_name+'.json'), 'w') as output_file:
        json.dump(timecode_list, output_file, indent=4)

print(len(final_cue_list))
output_file_path = os.path.join(all_cue_path, 'cue_corpus.json')
with open(output_file_path, 'w') as output_file:
    json.dump(final_cue_list, output_file, indent=4)

print(f"Cue Corpus saved to: {output_file_path}")