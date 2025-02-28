import xml.etree.ElementTree as ET
import os
import json

def list_to_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

def process_subtrack(cuedata):
    cuedata_dict = {}
    
    cuedata_dict['time'] = cuedata.get('time')
    cuedata_dict['command'] = cuedata.get('command')
    cuedata_dict['pressed'] = cuedata.get('pressed')
    cuedata_dict['step'] = cuedata.get('step')
    
    for item in cuedata:
        if 'Cue' in item.tag:
            nums = []
            for num in item:
                nums.append(num.text)
            cuedata_dict['cue'] = nums
    
    return cuedata_dict


file_path = 'data/timecode'


xml_files = [f for f in os.listdir(file_path) if f.endswith('.xml')]



for file in xml_files:
    cues = []
    complete_cue = []
    sinlge_cue = {}
    print('File:', file)
    filename = file[:-4]
    file = file_path + '/' + file
    tree = ET.parse(file)
    root = tree.getroot()

    
    for child in root:
        if 'Timecode' in child.tag:
            print('Timecode Num:', child.attrib['index'])
            length = child.attrib['lenght']
            complete_track = []
            for track in child:
                subtracks = []
                single_track = {}
                for item in track:
                    if 'Object' in item.tag:
                        nums = []
                        for num in item:
                            nums.append(num.text)
                        single_track['Object'] = {'name': item.attrib['name'], 'No': nums}
                    elif 'SubTrack' in item.tag:
                        subtrack = {}
                        subtrack['index'] = item.attrib['index']
                        subtrack['Event'] = []
                        for event in item:
                            subtrack['Event'].append(process_subtrack(event))
                        subtracks.append(subtrack)
                single_track['Subtrack'] = subtracks
            complete_track.append(single_track)



    pth = f'data/timecode_json/{filename}.json'
    list_to_json(complete_track, pth)
                    
                                


