import xml.etree.ElementTree as ET
import json
from datetime import datetime
import xml.dom.minidom

def json_to_cue_xml(json_data, output_path):
    """
    将 JSON 格式的 cue 序列数据转换为 XML 格式，并美化输出。
    
    参数：
    json_data (list): JSON 数据，包含 cue 列表
    output_path (str): 输出 XML 文件路径
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
        "datetime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),  # 当前时间
        "showfile": "generated_showfile"  # 默认值，可根据需要修改
    })
    
    # 添加 Sequ 元素
    sequ = ET.SubElement(root, "Sequ", {
        "index": "0",  # 默认值
        "timecode_slot": "255",  # 默认值
        "forced_position_mode": "0"
    })
    
    # 遍历 JSON 数据中的每个 cue
    for cue_index, cue_list in enumerate(json_data):
        if not cue_list:  # 如果 cue_list 为空
            ET.SubElement(sequ, "Cue", {"xsi:nil": "true"})
        else:
            # 假设每个 cue_list 是一个列表，包含单个 cue 的字典
            cue = cue_list[0]  # 取第一个元素
            cue_elem = ET.SubElement(sequ, "Cue", {"index": str(cue_index + 1)})  # index 从 1 开始
            
            # 添加 Number 元素
            if "Number" in cue:
                number = cue["Number"]
                ET.SubElement(cue_elem, "Number", {
                    "number": number["number"],
                    "sub_number": number["sub_number"]
                })
            
            # 添加 CueDatas 元素
            if "CueDatas" in cue:
                cuedatas_elem = ET.SubElement(cue_elem, "CueDatas")
                for cuedata in cue["CueDatas"]:
                    cuedata_elem = ET.SubElement(cuedatas_elem, "CueData", {
                        "value_multipart_index": cuedata["value_multipart_index"],
                        "effect_multipart_index": cuedata["effect_multipart_index"]
                    })
                    # 添加 Channel 元素
                    ET.SubElement(cuedata_elem, "Channel", {
                        "fixture_id": cuedata["fixture_id"],
                        "attribute_name": cuedata["attribute_name"]
                    })
                    # 添加 Value 元素
                    if "value" in cuedata:
                        ET.SubElement(cuedata_elem, "Value").text = cuedata["value"]
            
            # 添加 MibCue 元素
            if "MibCue" in cue:
                mibcue = cue["MibCue"]
                ET.SubElement(cue_elem, "MibCue", {
                    "number": mibcue["number"],
                    "sub_number": mibcue["sub_number"]
                })
            
            # 添加 CuePart 元素
            if "CuePart" in cue:
                cuepart = cue["CuePart"]
                ET.SubElement(cue_elem, "CuePart", {
                    "index": cuepart["index"]
                })
    
    # 生成 XML 字符串
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    
    # 使用 minidom 美化 XML
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="\t")
    
    # 写入文件并添加固定的头部信息
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<?xml-stylesheet type="text/xsl" href="styles/sequ@html@default.xsl"?>\n')
        f.write('<?xml-stylesheet type="text/xsl" href="styles/sequ@executorsheet.xsl" alternate="yes"?>\n')
        f.write('<?xml-stylesheet type="text/xsl" href="styles/sequ@trackingsheet.xsl" alternate="yes"?>\n')
        f.write(pretty_xml)

# 示例用法
if __name__ == "__main__":
    input_json_path = "data/blooming_flowers.json"  # 替换为你的 JSON 文件路径
    output_xml_path = "output_cue.xml"  # 输出 XML 文件路径
    
    # 读取 JSON 文件
    with open(input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 转换为 XML 并保存
    json_to_cue_xml(json_data, output_xml_path)
    print(f"已生成 XML 文件：{output_xml_path}")