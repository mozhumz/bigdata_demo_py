import sys

sum_count = 0
cu_wd = None
for line in sys.stdin:
    if line:
        wd_list = line.strip().split("\t")

        if len(wd_list) != 2:
            continue
        word, count = wd_list
        if cu_wd is None:
            cu_wd = word
        if word != cu_wd:
            print '\t'.join([cu_wd, str(sum_count)])
            sum_count = 0
            cu_wd = word

        sum_count += int(count)
if cu_wd:
    print '\t'.join([cu_wd, str(sum_count)])
