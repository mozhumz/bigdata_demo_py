import demoDay21_recsys_music.hyj.config_hyj as conf
import common.common_util as util
import math
# util.mkdirs('G:\\idea_workspace\\bigdata\\bigdata_demo_py\\demoDay21_recsys_music\\data\\music_mid_data_hyj2')
#
# res_file=conf.res_file
# with open(res_file,mode='r',encoding='utf-8') as f:
#     res_sort_list=eval(f.read())
#
# # topN
# filter_lst=res_sort_list[:5]
#
#
# res=['=>'.join([item_id,item_name,str(final_score),explain]) for item_id,item_name,final_score,explain in filter_lst]
# print(res)

p1=1/3
p2=1/3
p3=1/3
def getI(p1,p2):
    return -(math.log2(p1)*p1+math.log2(p2)*p2)

print(getI(p1,p2))
print(getI(p1,p2)-(math.log2(p3)*p3))