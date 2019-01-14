# -*- coding: utf-8 -*-
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:    
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248

        #rstring += unichr(inside_code)
        rstring += chr(inside_code)

    return rstring
    
def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring



b = strQ2B("ｍｎ123abc博客园")                           
print(b)

c = strB2Q("ｍｎ123abc博客园")                           
print(c)
