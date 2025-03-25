import json
import os
import torch
import numpy as np

attribute_name_map = {
    "ANIMATIONINDEXROTATE": 0,
    "ANIMATIONWHEEL": 1,
    "ARTNETDMX": 2,
    "BACKGROUNDCOLOURMIX": 3,
    "COLOR1": 4,
    "COLOR2": 5,
    "COLOR3": 6,
    "COLORCOLOR": 7,
    "COLORMACRO": 8,
    "COLORMIXER": 9,
    "COLORMIXMSPEED": 10,
    "COLORRGB1": 11,
    "COLORRGB2": 12,
    "COLORRGB3": 13,
    "COLORRGB4": 14,
    "COLORRGB5": 15,
    "COLORTEMPERATURE": 16,
    "CTO": 17,
    "DIM": 18,
    "DIMMERCURVE": 19,
    "DIST": 20,
    "EFFECTINDEXROTATE": 21,
    "EFFECTMACRORATE": 22,
    "EFFECTMACROS": 23,
    "EFFECTWHEEL": 24,
    "FIXTUREGLOBALRESET": 25,
    "FLIP": 26,
    "FOCUS": 27,
    "FOCUS2": 28,
    "FOCUS3": 29,
    "FOCUSMODE": 30,
    "FOREGROUNDCOLOURMIX": 31,
    "FROST": 32,
    "GOBO1": 33,
    "GOBO1WHEELSELECTMSPEED": 34,
    "GOBO1_POS": 35,
    "GOBO2": 36,
    "GOBO2_POS": 37,
    "GOBO3": 38,
    "GOBO3_POS": 39,
    "IRIS": 40,
    "IRIS2": 41,
    "LAMPCONTROL": 42,
    "MACRO SPEED": 43,
    "MARK": 44,
    "MP_ROT_X": 45,
    "MP_ROT_Y": 46,
    "MP_ROT_Z": 47,
    "MP_SC_X": 48,
    "MP_SC_Y": 49,
    "MP_SC_Z": 50,
    "MP_TR_X": 51,
    "MP_TR_Y": 52,
    "MP_TR_Z": 53,
    "PAN": 54,
    "POSITIONMSPEED": 55,
    "PRISMA1": 56,
    "PRISMA1_POS": 57,
    "PRISMA2_POS": 58,
    "PT MACRO": 59,
    "PTSPEED": 60,
    "RESET": 61,
    "SHUTTER": 62,
    "STAGEX": 63,
    "STAGEY": 64,
    "STAGEZ": 65,
    "STROBEDURATION": 66,
    "TILT": 67,
    "VIRTUAL_POSITION_MODE": 68,
    "ZOOM": 69,
    "ZOOMMODE": 70,
    "ZOOMMSPEED": 71,
}

class Embedding():
    def __init__(self):
        pass

    def preprocess_timecode(self, data):
        features = []
        max_sequence_length = 0

        for item in data:
            for subtrack in item["Subtrack"]:
                events = subtrack["Event"]
                sequence_length = len(events)
                max_sequence_length = max(max_sequence_length, sequence_length)

                for event in events:
                    time = float(event["time"]) if event["time"] is not None else 0.0
                    command = 1 if event["command"] == "Goto" else 0
                    pressed = 1 if event["pressed"] == "true" else 0
                    step = float(event["step"]) if event["step"] is not None else 0
                    if "cue" in event and command == 1 and pressed == 1:
                        cue = int(event["cue"][2]) if event["cue"] is not None else 0
                    else:
                        continue
                    feature = [time, cue]
                    features.append(feature)

        return torch.tensor(features, dtype = torch.int)

    def get_timecode_embedding(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            features = self.preprocess_timecode(data)
            return features

    def get_cue_embedding(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            features = self.preprocess_aligned_cue(data)
            return features

    def preprocess_aligned_cue(self, data):
        final_features = []
        for item in data:
            features = []
            # 排除时间错误的情况
            if item["time"] is None:
                continue
            # 时间信息
            time = float(item["time"]) / 30 / 0.96

            # CueDatas长度信息
            if "CueDatas" in item and type(item["CueDatas"]) == list:
                cue_datas = item["CueDatas"]
                cuedata_length = len(cue_datas)
            else:
                cuedata_length = 0
                continue
            
            # CueData内具体信息
            for cue in cue_datas:
                fixture_id = int(cue["fixture_id"])
                attribute_name = cue["attribute_name"]
                if "value" in cue:
                    value = cue["value"]
                    if type(value) is not float:
                        value = 0
                    else:
                        value = float(value)
                else:
                    value = 0

                # Map attribute_name to index
                attribute_index = attribute_name_map.get(attribute_name, -1)
                if attribute_index == -1:
                    raise ValueError(f"Unknown attribute_name: {attribute_name}")

                # 特征: [fixture_id, attribute_index, value]
                feature = [fixture_id, attribute_index, value]
                features.append(feature)

        # 最终数据格式
            final_features.append([time, cuedata_length, features])


        return final_features


    def get_timelist(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        timelist = []
        for item in data:
            time = item["time"]
            if time is None:
                continue
            timelist.append(time)
        return timelist
    
    def cue_to_json(self, features, output_filename):
        data = []
        for item in features:
            time = int(item[0] * 30 * 0.96) 
            cuedata_length = item[1]
            features_list = item[2]
            
            cue_datas = []
            for feature in features_list:
                fixture_id, attribute_index, value = feature
                
                # 反向映射 attribute_index 到 attribute_name
                attribute_name = {v: k for k, v in attribute_name_map.items()}.get(attribute_index, "Unknown")

                if attribute_name == "Unknown":
                    raise ValueError(f"Unknown attribute_index: {attribute_index}")

                cue_datas.append({
                    "fixture_id": fixture_id,
                    "attribute_name": attribute_name,
                    "value": value
                })

            data.append({
                "time": time,
                "CueDatas": cue_datas
            })

        # 保存为 JSON 文件
        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"JSON file saved to {output_filename}")

        