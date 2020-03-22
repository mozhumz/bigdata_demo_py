# coding=utf-8
import sys
import re


def dealWord(word):
    p = re.compile(r'\w+')
    re_word = p.findall(word)
    if len(re_word) == 0:
        return None
    return re_word[0].lower()


for line in sys.stdin:
    if line:
        wd_list = line.split(" ")
        for wd in wd_list:
            wd = dealWord(wd)
            if wd:
                print( "\t".join([wd, '1']))
