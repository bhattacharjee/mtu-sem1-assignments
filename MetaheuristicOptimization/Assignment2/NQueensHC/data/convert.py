#!/usr/bin/env python3

import sys

filename = sys.argv[1]
out_filename = filename[:-3] + "csv"

with open(filename, "r", encoding='utf-16le') as inputFile:
    with open(out_filename, "w") as outputFile:
        lines = [line.strip() for line in inputFile.readlines()]
        lines = [line[2:] for line in lines if line.startswith('=')]
        final_lines = []
        header_line = None
        for line in lines:
            if line.startswith("Run"):
                header_line = line
            else:
                final_lines.append(line)
        sys.stdout = outputFile
        print(header_line)
        [print(line) for line in final_lines]
inputFile.close()
outputFile.close()
