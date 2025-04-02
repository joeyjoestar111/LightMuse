import json
import os

cue_path = '../data/train/cue_json'
timecode_path = '../data/train/timecode_json'
aligned_cue_path = '../data/train/cue_aligned'


for item in os.listdir(timecode_path):
    file_name = item[:-5]  # Remove the .json extension
    print(f"Processing: {file_name}")
    
    with open(os.path.join(timecode_path, item), 'r') as f:
        timecode = json.load(f)
    
    with open(os.path.join(cue_path, file_name + '.json'), 'r') as f:
        cue = json.load(f)
    
    timecode = timecode[0]['Subtrack'][0]['Event']
    
    cue_list = []
    time_list = []
    for item1 in timecode:
        if 'cue' not in item1.keys():
            continue
        cue_list.append(item1['cue'][2])
        time_list.append(item1['time'])
        
    final_cue = []
    for i in range(len(cue_list)):
        cue_num = cue_list[i]
        time_num = time_list[i]
        final_cue_item = {}
        for item1 in cue:
            for item2 in item1:
                if item2['Number']["number"] == cue_num:
                    final_cue_item["number"] = cue_num
                    final_cue_item["time"] = time_num
                    if 'CueDatas' not in item2.keys():
                        final_cue_item['CueDatas'] = 0
                    else:
                        final_cue_item['CueDatas'] = item2['CueDatas']
                    final_cue.append(final_cue_item)
                    break

    output_file_path = os.path.join(aligned_cue_path, file_name + '.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(final_cue, output_file, indent=4)

    print(f"Aligned cue saved to: {output_file_path}")