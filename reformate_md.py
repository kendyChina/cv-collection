# -*- coding: utf-8 -*-

md_file = r"./README.md"

output_lines = []

with open(md_file, "r", encoding="UTF-8") as fr:
	for line in fr.readlines():
		output_lines.append(line.strip() + "  " + "\n")

with open(md_file, "w", encoding="UTF-8") as fw:
	fw.writelines(output_lines)