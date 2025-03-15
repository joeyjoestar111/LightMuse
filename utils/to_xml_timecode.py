import xml.etree.ElementTree as ET
import json
from datetime import datetime
import xml.dom.minidom

def json_to_xml(json_data, output_path):
    """
    将JSON格式的Timecode数据转换为XML格式，并美化输出。
    
    参数：
    json_data (list): JSON数据，包含Track列表
    output_path (str): 输出XML文件路径
    """
    # 创建根元素 MA
    root = ET.Element("MA", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xmlns": "http://schemas.malighting.de/grandma2/xml/MA",
        "xsi:schemaLocation": "http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd",
        "major_vers": "3",
        "minor_vers": "9",
        "stream_vers": "60"
    })
    
    # 添加 Info 元素
    info = ET.SubElement(root, "Info", {
        "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "showfile": "generated_showfile"  # 可根据需要修改
    })
    
    # 添加 Timecode 元素
    timecode = ET.SubElement(root, "Timecode", {
        "index": "0",
        "lenght": "6394"  # 默认值，可根据实际需求调整
    })
    
    # 遍历JSON数据中的每个Track
    for track_index, track in enumerate(json_data):
        track_elem = ET.SubElement(timecode, "Track", {
            "index": str(track_index),
            "active": "true",
            "expanded": "true"
        })
        
        # 创建 Object 元素
        object_elem = ET.SubElement(track_elem, "Object", {
            "name": track["Object"]["name"]
        })
        for no in track["Object"]["No"]:
            ET.SubElement(object_elem, "No").text = no
        
        # 遍历 Subtrack
        for subtrack in track["Subtrack"]:
            subtrack_elem = ET.SubElement(track_elem, "SubTrack", {
                "index": subtrack["index"]
            })
            if subtrack["index"] == "1":
                subtrack_elem.set("fader_command", "Master")
            
            # 遍历 Event
            for event_index, event in enumerate(subtrack["Event"]):
                if event["command"] is not None:
                    # 普通事件
                    event_elem = ET.SubElement(subtrack_elem, "Event", {
                        "index": str(event_index),
                        "time": event["time"],
                        "command": event["command"],
                        "pressed": event["pressed"],
                        "step": event["step"]
                    })
                    if "cue" in event and event["cue"]:
                        cue_elem = ET.SubElement(event_elem, "Cue", {
                            "name": f"Cue {event['step']}"
                        })
                        for no in event["cue"]:
                            ET.SubElement(cue_elem, "No").text = no
                else:
                    # 特殊事件（如fader_level）
                    event_elem = ET.SubElement(subtrack_elem, "Event", {
                        "index": str(event_index),
                        "time": event["time"],
                        "fader_level": "1"  # 默认值，可根据JSON调整
                    })
    
    # 生成XML字符串
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    
    # 使用minidom美化XML
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="\t")
    
    # 写入文件并添加XML声明和样式表指令
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<?xml-stylesheet type="text/xsl" href="styles/timecode@sheet.xsl"?>\n')
        f.write(pretty_xml)

# 示例用法
if __name__ == "__main__":
    input_json_path = "timecode_json/blooming_flowers.json"  # 替换为你的JSON文件路径
    output_xml_path = "output2.xml"  # 输出XML文件路径
    
    with open(input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    json_to_xml(json_data, output_xml_path)
    print(f"已生成XML文件：{output_xml_path}")