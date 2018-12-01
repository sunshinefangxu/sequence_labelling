# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 下午8:37
# @Author  : lfx
# @FileName: prf.py
# @Software: PyCharm

def tool(ref):
    res = []
    for item in ref:
        ref_new = []
        c_list = []
        for c in item:
            if c == "E" or c == "S":
                c_list.append(c)
                ref_new.append("".join(c_list))
                c_list = []
            else:
                c_list.append(c)
        res.append(ref_new)
    return [' '.join(item) for item in res]

def getSegPRF(ref,result):
    ref = tool(ref)
    result = tool(result)
    RefWordNum = 0
    CorrectNum = 0
    ErrorNum = 0
    if len(ref) != len(result):
        print 'file error'
        raise Exception
    for i in xrange(len(ref)):
        ref_sentence = ref[i].strip().decode('utf-8')
        result_sentence = result[i].strip().decode('utf-8')
        #ref seg position
        ref_seg = []
        b = 0
        e = 0
        for ch in ref_sentence:
            if ch == ' ':
                ref_seg.append((b,e))
                b = e
            else:
                e += 1
        ref_seg.append((b,e))
        #result seg position
        result_seg = []
        b = 0
        e = 0
        for ch in result_sentence:
            if ch == ' ':
                result_seg.append((b,e))
                b = e
            else:
                e += 1
        result_seg.append((b,e))
        #count N,c,e
        RefWordNum += len(ref_seg)
        ref_wp = 0
        result_wp = 0
        while ref_wp < len(ref_seg) and result_wp < len(result_seg):
            if ref_seg[ref_wp][0] == result_seg[result_wp][0]:
                if ref_seg[ref_wp][1] == result_seg[result_wp][1]:
                    CorrectNum += 1
                    ref_wp += 1
                    result_wp += 1
                elif ref_seg[ref_wp][1] > result_seg[result_wp][1]:
                    result_wp += 1
                    ErrorNum += 1
                else:
                    ref_wp += 1
            else:
                if ref_seg[ref_wp][1] == result_seg[result_wp][1]:
                    ErrorNum += 1
                    ref_wp += 1
                    result_wp += 1
                elif ref_seg[ref_wp][1] > result_seg[result_wp][1]:
                    ErrorNum += 1
                    result_wp += 1
                else:
                    ref_wp += 1

    p = float(CorrectNum)/float(CorrectNum+ErrorNum)
    r = float(CorrectNum)/float(RefWordNum)
    f = float(2*p*r)/float(p+r)

    return f

