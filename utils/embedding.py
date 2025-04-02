import json
import os
import torch
import numpy as np

attribute_name_map = {
    "ANIMATIONINDEXROTATE": 0,
    "ANIMATIONWHEEL": 1,
    "ARTNETDMX": 2,
    "ASSERT": 3,
    "AUTOFOCUS": 4,
    "AUTOFOCUS_ADJ": 5,
    "AUTOFOCUS_DIST": 6,
    "BACKG B": 7,
    "BACKG G": 8,
    "BACKG R": 9,
    "BACKG W": 10,
    "BACKGROUNDCOLOURMIX": 11,
    "BAOLIU": 12,
    "BARNDOOR1": 13,
    "BARNDOOR2": 14,
    "BARNDOOR3": 15,
    "BARNDOOR4": 16,
    "BARNDOORMACROS": 17,
    "BARNDOORSASSEMBLY": 18,
    "BEAMSHAPER": 19,
    "BG DIM": 20,
    "BG STROBE": 21,
    "BJB": 22,
    "BJG": 23,
    "BJR": 24,
    "BJW": 25,
    "BLADE1A": 26,
    "BLADE1B": 27,
    "BLADE1ROT": 28,
    "BLADE2A": 29,
    "BLADE2B": 30,
    "BLADE3A": 31,
    "BLADE3B": 32,
    "BLADE4A": 33,
    "BLADE4B": 34,
    "BLUEADJUST": 35,
    "CHASE": 36,
    "CHASELEVEL": 37,
    "CHASERATE": 38,
    "CLIPSELECT": 39,
    "CMYMACRO": 40,
    "CMYSPEED": 41,
    "COLOR1": 42,
    "COLOR1SPEED": 43,
    "COLOR2": 44,
    "COLOR2SPEED": 45,
    "COLOR3": 46,
    "COLORCOLOR": 47,
    "COLORHSB1": 48,
    "COLORHSB2": 49,
    "COLORHSB3": 50,
    "COLORMACRO": 51,
    "COLORMACROS": 52,
    "COLORMIXER": 53,
    "COLORMIXER2": 54,
    "COLORMIXMSPEED": 55,
    "COLORMIXMSPEED2": 56,
    "COLORRGB1": 57,
    "COLORRGB10": 58,
    "COLORRGB11": 59,
    "COLORRGB12": 60,
    "COLORRGB13": 61,
    "COLORRGB15": 62,
    "COLORRGB19": 63,
    "COLORRGB2": 64,
    "COLORRGB23": 65,
    "COLORRGB24": 66,
    "COLORRGB25": 67,
    "COLORRGB28": 68,
    "COLORRGB3": 69,
    "COLORRGB4": 70,
    "COLORRGB5": 71,
    "COLORRGB6": 72,
    "COLORRGB7": 73,
    "COLORRGB8": 74,
    "COLORRGB9": 75,
    "COLORSPEED": 76,
    "COLORTEMPERATURE": 77,
    "COLORWHEELRESET": 78,
    "CONTROL": 79,
    "CONTROLLER": 80,
    "CONTROLRANGE": 81,
    "COOL W (1)": 82,
    "COOL W (2)": 83,
    "CRI": 84,
    "CTC": 85,
    "CTO": 86,
    "CYMMACRO": 87,
    "DIM": 88,
    "DIM2": 89,
    "DIMMER MODE": 90,
    "DIMMERCURVE": 91,
    "DIMSPEED": 92,
    "DIST": 93,
    "DOWNFOCUS": 94,
    "EFFECT": 95,
    "EFFECTINDEXROTATE": 96,
    "EFFECTINDEXROTATE2": 97,
    "EFFECTINDEXROTATE3": 98,
    "EFFECTMACRORATE": 99,
    "EFFECTMACROS": 100,
    "EFFECTMACROS2": 101,
    "EFFECTMACSPEED": 102,
    "EFFECTWHEEL": 103,
    "EFFECTWHEEL2": 104,
    "EFFECTWHEEL3": 105,
    "EFFECTWHEELRESET": 106,
    "FENG": 107,
    "FIXTUREGLOBALRESET": 108,
    "FLIP": 109,
    "FOCUS": 110,
    "FOCUS2": 111,
    "FOCUS2ROT": 112,
    "FOCUS3": 113,
    "FOCUSADJUST": 114,
    "FOCUSDISTANCE": 115,
    "FOCUSMODE": 116,
    "FOREGROUNDCOLOURMIX": 117,
    "FRAMEMACROS": 118,
    "FRAMEMSPEED": 119,
    "FROST": 120,
    "FROST2": 121,
    "FROST3": 122,
    "FROSTRESET": 123,
    "FUNCTION": 124,
    "FUNTRON": 125,
    "GOBO": 126,
    "GOBO 1": 127,
    "GOBO1": 128,
    "GOBO1OFFSET": 129,
    "GOBO1WHEELRESET": 130,
    "GOBO1WHEELSELECTMSPEED": 131,
    "GOBO1_POS": 132,
    "GOBO2": 133,
    "GOBO2MODE": 134,
    "GOBO2OFFSET": 135,
    "GOBO2WHEELOFFSET": 136,
    "GOBO2_MODE": 137,
    "GOBO2_POS": 138,
    "GOBO3": 139,
    "GOBO3OFFSET": 140,
    "GOBO3_POS": 141,
    "GOBOFADE": 142,
    "GOBOSPEED": 143,
    "GREENADJUST": 144,
    "HUE": 145,
    "INTENSITYMACRORATE": 146,
    "INTENSITYMACROS": 147,
    "INTENSITYMSPEED": 148,
    "INTENSITYRESET": 149,
    "IRIS": 150,
    "IRIS2": 151,
    "IRISMACRO": 152,
    "IRISMODE": 153,
    "LAMP": 154,
    "LAMPCONTROL": 155,
    "LENS": 156,
    "LIGHTFROST": 157,
    "LIGHTMODE": 158,
    "MACRO": 159,
    "MACRO SPEED": 160,
    "MACROS": 161,
    "MACROS2": 162,
    "MACROS3": 163,
    "MACROS4": 164,
    "MARK": 165,
    "MASTERCONTROI": 166,
    "MODE": 167,
    "MP_ROT_W": 168,
    "MP_ROT_X": 169,
    "MP_ROT_Y": 170,
    "MP_ROT_Z": 171,
    "MP_SC_X": 172,
    "MP_SC_Y": 173,
    "MP_SC_Z": 174,
    "MP_TR_X": 175,
    "MP_TR_Y": 176,
    "MP_TR_Z": 177,
    "OPEN": 178,
    "OUTPUT": 179,
    "P/T SPEED": 180,
    "P/TSPEED": 181,
    "PAGE": 182,
    "PAN": 183,
    "PAN ROTATION": 184,
    "PANEL DIM": 185,
    "PANEL SHUTTER": 186,
    "PANMODE": 187,
    "PANROT": 188,
    "PATTERN": 189,
    "PATTERNINDEXROTATION": 190,
    "POSITIONMSPEED": 191,
    "POSITIONMSPEED2": 192,
    "POSITIONRESET": 193,
    "PRISMA": 194,
    "PRISMA1": 195,
    "PRISMA1_POS": 196,
    "PRISMA2": 197,
    "PRISMA2_POS": 198,
    "PRISMA_POS": 199,
    "PRISMMACRO": 200,
    "PT MACRO": 201,
    "PTSPEED": 202,
    "PWMFREQUENCY": 203,
    "PWMFREQUENCYADJUST": 204,
    "REDADJUST": 205,
    "REFLECTORADJUST": 206,
    "RELEASE": 207,
    "RESET": 208,
    "RGBMACRO": 209,
    "RGBSTROBE": 210,
    "ROLL": 211,
    "SCANRATE": 212,
    "SCROLLER": 213,
    "SHAPER ROT": 214,
    "SHUTTER": 215,
    "SHUTTER2": 216,
    "SHUTTERRESET": 217,
    "SPEED": 218,
    "STAGEX": 219,
    "STAGEY": 220,
    "STAGEZ": 221,
    "STROBECOLORBLUE": 222,
    "STROBECOLORGREEN": 223,
    "STROBECOLORRED": 224,
    "STROBECOLORWHITE": 225,
    "STROBEDURATION": 226,
    "STROBEMODE": 227,
    "STROBE_RATIO": 228,
    "TILT": 229,
    "TILT ROTATION": 230,
    "TILTMODE": 231,
    "TILTMSPEED": 232,
    "TILTROT": 233,
    "TINT": 234,
    "UPFOCUS": 235,
    "VIDEO_MODE": 236,
    "VIRTUAL_POSITION_MODE": 237,
    "VISIBLEPOINTS": 238,
    "V_DARKNESS": 239,
    "V_DARK_IS_TRANSPAR": 240,
    "V_EFF1_P1": 241,
    "V_EFF1_P2": 242,
    "V_EFF1_P3": 243,
    "V_EFF1_T": 244,
    "V_EFF2_P1": 245,
    "V_EFF2_P2": 246,
    "V_EFF3_P1": 247,
    "V_EFF3_P2": 248,
    "V_EFF4_P1": 249,
    "V_EFF4_P2": 250,
    "V_IOFFS_X": 251,
    "V_IOFFS_Y": 252,
    "V_ISPLT_X": 253,
    "V_ISPLT_Y": 254,
    "V_O3DPOOL": 255,
    "V_OIMAGE": 256,
    "V_OIMAGE_MASK": 257,
    "V_OPOOL": 258,
    "V_OP_SCALE": 259,
    "V_OP_X": 260,
    "V_OP_Y": 261,
    "V_OP_Z": 262,
    "V_ORC_X": 263,
    "V_ORC_Y": 264,
    "V_OR_Z": 265,
    "V_OS_X": 266,
    "V_OS_Y": 267,
    "V_OTYPE": 268,
    "V_PLAYMODE": 269,
    "V_PRESET1": 270,
    "V_PRESET2": 271,
    "V_VSPEED": 272,
    "WARM W ": 273,
    "WARM W (2)": 274,
    "WHITE DIMMER": 275,
    "WHITESTROBE": 276,
    "WSHTTER": 277,
    "WSTROBE": 278,
    "WU": 279,
    "XSIZE": 280,
    "YAN": 281,
    "YSIZE": 282,
    "ZOOM": 283,
    "ZOOM2": 284,
    "ZOOMMODE": 285,
    "ZOOMMSPEED": 286,
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

    def get_cue_corpus(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            features = self.preprocess_cue_corpus(data)
            return features

    def preprocess_cue_corpus(self, data):
        final_features = []
        for item in data:
            features = []
            # 排除时间错误的情况
            if "CueDatas" in item and type(item["CueDatas"]) == list:
                cue_datas = item["CueDatas"]
            else:
                continue
            
            # CueData内具体信息
            for cue in cue_datas:
                if "fixture_id" in cue.keys() and cue["fixture_id"] != None:
                    fixture_id = int(cue["fixture_id"])
                else:
                    continue
                if "attribute_name" in cue.keys() and cue["attribute_name"] != None:
                    attribute_name = cue["attribute_name"]
                else:
                    continue
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
            final_features.append(features)

        return final_features

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

        