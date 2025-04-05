import re
import sys


def get_header(filepath: str):
    with open(filepath, mode='r') as fp:
        return fp.readline().strip()


if __name__ == '__main__':
    header = get_header(sys.argv[1:][0])
    #print(header)
    clean_header = re.sub(r"[^a-zA-Z0-9,_]", "", header)
    print(clean_header.lower())
