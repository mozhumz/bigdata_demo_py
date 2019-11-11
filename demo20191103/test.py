# coding=utf-8
import re

def get_word(word):
    p = re.compile(r'\w+')
    re_word=p.findall(word)
    if len(re_word) == 0:
        return None
    return re_word[0].lower()

def validateTitle( title):
    """ 将 title 名字 规则化
    :param title: title name 字符串
    :return: 文件命名支持的字符串
    """
    rstr = r"[\"\“\”=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"  # '= ( ) ， / \ : * ? " < > |  '   还有空格
    new_title = re.sub(rstr, "", title)  # 替换为空
    return new_title

# str='ssSS2   ,”'
# newStr=get_word(str)
# print newStr
#
# dict = {}
# a=1
# a_v="haha"
#
# dict[a]=a_v
#
# print dict
# rec_data="+MIPLOBSERVE:0,68220,1,3303,0,-1"
# msgidRegex = re.compile(r',(\d)+,')
# mo = msgidRegex.findall(rec_data)
# print mo

s=""
if s:
    print str(s!="")
else:
    print "jj"