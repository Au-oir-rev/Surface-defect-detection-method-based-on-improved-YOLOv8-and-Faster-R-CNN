import os
import xml.etree.ElementTree as ET

# 定义类别映射表，根据需要修改
class_map = {
    "crazing": 0,  # 将类别映射为数字，从0开始
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
    # 添加其他类别，如果有的话
}

# 输入和输出路径
input_dir = r"C:\Users\88983\Desktop\NEU\NEU-DET\ANNOTATIONS"  # XML文件所在文件夹
output_dir = r"C:\Users\88983\Desktop\NEU\dataset\labels"  # TXT文件输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 遍历所有XML文件
for xml_file in os.listdir(input_dir):
    if not xml_file.endswith(".xml"):
        continue

    # 解析XML文件
    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    # 提取图像宽高
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    # 创建对应的TXT文件
    txt_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
    with open(txt_file, "w") as f:
        # 遍历每个object节点
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_map:
                print(f"Warning: Class '{class_name}' not in class_map. Skipping...")
                continue
            class_id = class_map[class_name]

            # 提取边界框信息
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # 转换为YOLO格式
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 写入TXT文件
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("转换完成！")
