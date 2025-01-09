from llama_index.core.llms import ChatMessage, MessageRole
import pandas as pd
import pymysql
import networkx as nx
import numpy as np
from pyvis.network import Network
import os
import re

def extract_image_context(md_string, n):
    lines = md_string.splitlines()

    # 定义图像行的正则表达式
    pattern = re.compile(r'^!\[\]\(([^)]*)\)$')

    # 记录所有非空、非图像行的索引和内容
    non_empty_non_image_indices = []
    # 记录所有图像行的索引和路径
    image_indices = []

    for idx, line in enumerate(lines):
        # 去除行首尾的空白字符
        stripped_line = line.strip()
        if stripped_line == '':  # 空行
            continue
        # 检查是否是图像行
        match = pattern.match(line)
        if match:
            image_indices.append((idx, match.group(1)))
        else:
            non_empty_non_image_indices.append((idx, line))

    # 现在，对于每一个图像行，找到上下n个非空、非图像行
    result = []
    text_indices = [index for index, _ in non_empty_non_image_indices]

    for img_index, img_path in image_indices:
        # 找到所有非空、非图像行的索引中，小于img_index的最大的n个索引
        lower_indices = [index for index in text_indices if index < img_index]
        lower_selected = sorted(lower_indices, reverse=True)[:n]
        lower_selected.reverse()
        # 找到所有非空、非图像行的索引中，大于img_index的最小的n个索引
        upper_indices = [index for index in text_indices if index > img_index]
        upper_selected = sorted(upper_indices)[:n]

        # 获取对应的行内容
        lower_lines = [lines[i] for i in lower_selected]
        upper_lines = [lines[i] for i in upper_selected]

        # 组合context
        context = '\n'.join(lower_lines + upper_lines)

        # 加入结果列表
        result.append((img_path, context))

    return result