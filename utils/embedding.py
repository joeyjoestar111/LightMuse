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

    @staticmethod
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
                    if "cue" in event:
                        cue = list(map(float, event["cue"])) if event["cue"] is not None else [0, 0, 0]

                    feature = [time, command, pressed, step] + cue
                    features.append(feature)

        # Convert to tensor
        features = np.array(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Padding missing values
        padded_features = torch.zeros((len(data), max_sequence_length, features.shape[1]))
        for i, item in enumerate(data):
            for j, subtrack in enumerate(item["Subtrack"]):
                events = subtrack["Event"]
                for k, event in enumerate(events):
                    time = float(event["time"]) if event["time"] is not None else 0.0
                    command = 1 if event["command"] == "Goto" else 0
                    pressed = 1 if event["pressed"] == "true" else 0
                    step = float(event["step"]) if event["step"] is not None else 0
                    if "cue" in event:
                        cue = list(map(float, event["cue"])) if event["cue"] else [0, 0, 0]

                    feature = [time, command, pressed, step] + cue
                    padded_features[i, k] = torch.tensor(feature, dtype=torch.float32)

        return padded_features

    @staticmethod
    def get_timecode_embedding(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            features = Embedding.preprocess_timecode(data)[0]
            print(features)
            print(features.shape)
            return features

    @staticmethod
    def preprocess_cue(self, data):
        final_features = []
        max_sequence_length = 0
        for list in data:
            features = []
            for item in list:
                if "CueDatas" in item:
                    cue_datas = item["CueDatas"]
                else:
                    continue
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

                    # Organize features: [fixture_id, attribute_index, value]
                    feature = [fixture_id, attribute_index, value]
                    features.append(feature)
                    max_sequence_length = max(max_sequence_length, len(features))

            final_features.append(features)

        for f in final_features:
            if len(f) < max_sequence_length:
                for i in range(max_sequence_length - len(f)):
                    f.append([0, 0, 0])

        # Convert to tensor
        final_features = np.array(final_features)
        final_features = torch.tensor(final_features, dtype=torch.float32)

        return final_features

    @staticmethod
    def get_cue_embedding(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            features = Embedding.preprocess_cue(data)
            print(features)
            print(features.shape)
            return features

    @staticmethod
    def preprocess_aligned_cue(self, data):
        final_features = []
        max_sequence_length = 0
        for item in data:
            features = []
            time = item["time"]
            if "CueDatas" in item:
                cue_datas = item["CueDatas"]
            else:
                continue
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

                # Organize features: [fixture_id, attribute_index, value]
                feature = [fixture_id, attribute_index, value]
                features.append(feature)
                max_sequence_length = max(max_sequence_length, len(features))

        final_features.append(features)

        for f in final_features:
            if len(f) < max_sequence_length:
                for i in range(max_sequence_length - len(f)):
                    f.append([0, 0, 0])

        # Convert to tensor
        final_features = np.array(final_features)
        final_features = torch.tensor(final_features, dtype=torch.float32)

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
        
        

# def preprocess_timecode(data):
#     features = []
#     max_sequence_length = 0

#     for item in data:
#         for subtrack in item["Subtrack"]:
#             events = subtrack["Event"]
#             sequence_length = len(events)
#             max_sequence_length = max(max_sequence_length, sequence_length)

#             for event in events:
#                 time = float(event["time"]) if event["time"] is not None else 0.0
#                 command = 1 if event["command"] == "Goto" else 0
#                 pressed = 1 if event["pressed"] == "true" else 0
#                 step = float(event["step"]) if event["step"] is not None else 0
#                 if "cue" in event:
#                     cue = list(map(float, event["cue"])) if event["cue"] is not None else [0, 0, 0]

#                 feature = [time, command, pressed, step] + cue
#                 features.append(feature)

#     # 转换为张量
#     features = np.array(features)
#     features = torch.tensor(features, dtype=torch.float32)

#     # 填充缺失值
#     padded_features = torch.zeros((len(data), max_sequence_length, features.shape[1]))
#     for i, item in enumerate(data):
#         for j, subtrack in enumerate(item["Subtrack"]):
#             events = subtrack["Event"]
#             for k, event in enumerate(events):
#                 time = float(event["time"]) if event["time"] is not None else 0.0
#                 command = 1 if event["command"] == "Goto" else 0
#                 pressed = 1 if event["pressed"] == "true" else 0
#                 step = float(event["step"]) if event["step"] is not None else 0
#                 if "cue" in event:
#                     cue = list(map(float, event["cue"])) if event["cue"] else [0, 0, 0]

#                 feature = [time, command, pressed, step] + cue
#                 padded_features[i, k] = torch.tensor(feature, dtype=torch.float32)

#     return padded_features


# def get_timecode_embedding(filename):
#     with open(filename, "r", encoding="utf-8") as file:
#         data = json.load(file)
#         # 每一行的内容依次为 [time command pressed step cue*3]
#         features = preprocess_timecode(data)[0]
#         print(features)
#         print(features.shape)
#         return features

# def preprocess_cue(data):
#     final_features = []
#     max_sequence_length = 0
#     for list in data:
#         features = []
#         for item in list:
#             if "CueDatas" in item:
#                 cue_datas = item["CueDatas"]
#             else:
#                 continue
#             for cue in cue_datas:
#                 fixture_id = int(cue["fixture_id"])
#                 attribute_name = cue["attribute_name"]
#                 if "value" in cue:
#                     value = cue["value"]
#                     if type(value) is not float:
#                         value = 0
#                     else:
#                         value = float(value)
#                 else:
#                     value = 0
#                 # 将attribute_name映射到整数
#                 attribute_index = attribute_name_map.get(attribute_name, -1)
#                 if attribute_index == -1:
#                     raise ValueError(f"Unknown attribute_name: {attribute_name}")

#                 # 组织特征：[fixture_id, attribute_index, value]
#                 feature = [fixture_id, attribute_index, value]
#                 features.append(feature)
#                 max_sequence_length = max(max_sequence_length, len(features))

#             final_features.append(features)

#     for f in final_features:
#         if len(f) < max_sequence_length:
#             for i in range(max_sequence_length - len(f)):
#                 f.append([0, 0, 0])
#     # 转换为张量
#     final_features = np.array(final_features)
#     final_features = torch.tensor(final_features, dtype=torch.float32)

#     return final_features

# def get_cue_embedding(filename):
#     with open(filename, "r", encoding="utf-8") as file:
#         data = json.load(file)
#         features = preprocess_cue(data)
#         print(features)
#         print(features.shape)
#         return features


if __name__ == "__main__":
    # for item in os.listdir('../data/timecode_json'):
    #     timecode_embedding = get_timecode_embedding('../data/timecode_json/' + item)
    for item in os.listdir('../data/cue_json'):
        print(item)
        cue_embedding = get_cue_embedding('../data/cue_json/' + item)
        exit()
