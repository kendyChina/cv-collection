# -*- coding: utf-8 -*-

md_file = r"./README.md"

output_lines = []

with open(md_file, "r", encoding="UTF-8") as fr:
	for line in fr.readlines():
		if "[TOC]" in line or "[toc]" in line: continue
		if "---" in line:
			output_lines.append("\n" + line + "\n")
			continue
		output_lines.append(line.rstrip() + "  " + "\n")

with open(md_file, "w", encoding="UTF-8") as fw:
	fw.writelines(output_lines)