import os
import xml.etree.ElementTree as ET

input_folder = '/home/lll/software/fixed/MMDetector/test_data/ShangHai_4/VOCdevkit/VOC2007/Annotations'

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.xml'):  # 仅处理以 .xml 结尾的文件
        # 构建 XML 文件的完整路径
        xml_path = os.path.join(input_folder, filename)

        # 解析 XML 文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取 <filename> 元素
        filename_element = root.find('filename')

        # 修改 <filename> 元素的属性
        filename_element.text = filename_element.text.replace('.png', '.jpg')

        # 保存修改后的 XML 文件
        tree.write(xml_path)

print("修改完成！")