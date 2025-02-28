import xml.etree.ElementTree as ET
import os
import json

def list_to_json(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

def process_cuedata(cuedata):
    cuedata_dict = {}
    
    cuedata_dict['value_multipart_index'] = cuedata.get('value_multipart_index')
    cuedata_dict['effect_multipart_index'] = cuedata.get('effect_multipart_index')
    
    for item in cuedata:
        if 'Channel' in item.tag:
            channel = item
            cuedata_dict['fixture_id'] = channel.get('fixture_id')
            cuedata_dict['attribute_name'] = channel.get('attribute_name')
        elif 'Value' in item.tag:
            cuedata_dict['value'] = item.text
    
    return cuedata_dict


file_path = 'data/cues'


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
    
    # print("Root标签: ", root.tag)
    # print("Root属性: ", root.attrib['major_vers'])
    # print("Root文本: ", root.text)
    
    for child in root:
        if 'Sequ' in child.tag:
            print('Sequence Num:', child.attrib['index'])
            index, timecode_slot = child.attrib['index'], child.attrib['timecode_slot']
            for cue in child:
                complete_cue = []
                if 'index' in cue.attrib:
                    print('Cue Num: ', cue.attrib['index'])
                    single_cue = {}
                    for cue_data in cue:
                        if 'Number' in cue_data.tag:
                            single_cue['Number'] = {'number': cue_data.attrib['number'], 'sub_number': cue_data.attrib['sub_number']}
                        elif 'CueDatas' in cue_data.tag:
                            all_cue_data = []
                            for item in cue_data:
                                all_cue_data.append(process_cuedata(item))
                            single_cue['CueDatas'] = all_cue_data
                        elif 'MibCue' in cue_data.tag:
                            single_cue['MibCue'] = {'number': cue_data.attrib['number'], 'sub_number': cue_data.attrib['sub_number']}
                        elif 'CuePart' in cue_data.tag:
                            single_cue['CuePart'] = {'index': cue_data.attrib['index']}
                        # print(single_cue)
                        complete_cue.append(single_cue)
                    cues.append(complete_cue)

    pth = f'data/cue_json/{filename}.json'
    list_to_json(cues, pth)
                    
                                


