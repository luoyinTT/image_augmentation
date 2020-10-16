#!/usr/bin/env python
# coding: utf-8

# In[5]:

# from pyecharts.globals import CurrentConfig, NotebookType, OnlineHostType
# CurrentConfig.ONLINE_HOST = "https://assets.pyecharts.org/assets/"
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity='all'
from pyecharts.charts import Bar,Grid,Pie,Line
from pyecharts import options as opts
import os
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import matplotlib.pyplot as plt
import datetime
import csv
import sympy
from data_aug import horizone_flip_enhance, vertical_flip_enhance, luminance_enhance, translation_enhance,                      rotate_enhance, gaussian_enhance, crop_enhance


# In[ ]:


def directory_check(xml_path, image_path, xml_source_path, image_source_path):
    if os.path.isdir(xml_path):
        shutil.rmtree(xml_path)
        print("clear the exist xml_path !")
    if os.path.isdir(image_path):
        shutil.rmtree(image_path)
        print("clear the exist image_path !")
    shutil.copytree(xml_source_path, xml_path)
    shutil.copytree(image_source_path, image_path)


# In[ ]:


def parse_obj(xml_path, filename):
    tree = ET.parse(xml_path + filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        objects.append(obj_struct)
    return objects


# In[ ]:


def get_files(xml_path):
    """
    获取文件夹中所有文件的名称
    :param xml_path:
    :return filenames: filenames is the list of all images
    """
    file_list = os.listdir(xml_path)
    filenames = []
    for name in file_list:
        name = name.replace('.xml', '')
        filenames.append(name)
    return filenames


# In[ ]:


def class_percent(num_objs):
    """
    计算每个类别的百分比average_list，以及平均分布时应该有的百分比average
    :param num_objs:
    :return average_list, average:
    """
    average_list = []
    list_sum = sum(num_objs.values())
    for obj in list(num_objs.values()):
        percent = obj / list_sum
        average_list.append(percent)
    average = 1 / len(num_objs)
#     average_list.sort()
    # print("average:", average)
#     print("average_list:", average_list)
    return average, average_list


# In[ ]:


def draw(source_num_objs, num_objs,xml_path):
    """
    画图
    :param source_num_objs:
    :param num_objs:
    :return:
    """
    plt.figure(1)
    plt.subplot(1, 2, 1)
    x = list(num_objs.keys())
    plt.plot(x, list(source_num_objs.values()), color="b", linestyle="-", marker="v", linewidth=1)
    plt.title("Original data")
    max_class = (max(list(num_objs.values())) // 100 + 2) * 100
    my_y_ticks = np.arange(0, max_class, 100)
    plt.yticks(my_y_ticks)
    plt.subplot(1, 2, 2)
    plt.plot(x, list(num_objs.values()), color="r", linestyle="-", marker="^", linewidth=1)
    plt.title("After augmentation")
    plt.yticks(my_y_ticks)
    plt.savefig(os.path.join(xml_path, os.path.pardir+"/"+"final.png"))
    # plt.show()

# In[ ]:


def draw_show(source_num_objs, num_objs):
    """
    画图
    :param source_num_objs:
    :param num_objs:
    :return:
    """
    plt.figure(1)
    plt.subplot(1, 2, 1)
    x = list(num_objs.keys())
    plt.plot(x, list(source_num_objs.values()), color="b", linestyle="-", marker="v", linewidth=1)
    plt.title("Original data")
    max_class = (max(list(num_objs.values())) // 100 + 2) * 100
    my_y_ticks = np.arange(0, max_class, 100)
    plt.yticks(my_y_ticks)
    plt.subplot(1, 2, 2)
    plt.plot(x, list(num_objs.values()), color="r", linestyle="-", marker="^", linewidth=1)
    plt.title("After augmentation")
    plt.yticks(my_y_ticks)
    plt.show()

# In[ ]:


def log_file(times, class_names, num_objs, log_path):
    """
    把每一轮增强后的数据量存入CSV文件中
    :param times:
    :param class_names:
    :param num_objs:
    :return:
    """
    if times == 0:
        # the first time write log
        csv_head = ["times"]
        with open(log_path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_head.extend(class_names)
            csv_write.writerow(csv_head)
    else:
        # write data into csv
        data_row = [times]
        with open(log_path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            for obj in class_names:
                data_row.append(num_objs[obj])
            csv_write.writerow(data_row)
        print("已写入日志文件")

        
        
# In[ ]:


def class_num(filenames,xml_path):
    """
    计算每个类别的数量，并且统计有几个类别
    :param filenames: 
    :return: class_names,num_objs,recs
    """
    recs = {}
    class_names = []
    num_objs = {}
    for i, name in enumerate(filenames):
        recs[name] = parse_obj(xml_path, name + '.xml')
    # print(recs)
    for name in filenames:
        for object in recs[name]:
            if object['name'] not in num_objs.keys():
                num_objs[object['name']] = 1
            else:
                num_objs[object['name']] += 1
            if object['name'] not in class_names:
                class_names.append(object['name'])
    # print("class_names:", class_names)
    # print("num_objs:", num_objs)
    return class_names, num_objs, recs


# In[ ]:


def if_balanced(num_objs, times, source_list, PERCENT):
    """
    判断类别量是否已经均衡
    :param num_objs, times:
    :return:
    """
    num_list = list(num_objs.values())
    num_list.sort()
    average, average_class = class_percent(num_objs)
    a = 1
    banlance_control = times//200
    percent = PERCENT + banlance_control*0.005
    for obj in average_class:
        if obj <= average+percent and obj >= average-percent:
            continue
        else:
            a = 0
            break
    low_class = [a for a in average_class if a < average]
    if a == 1:
        print("all class have meet the requirement")
        return 1
    elif len([b for b in low_class if b >= average - percent]) == len(low_class):
        print("low class meet the requirement")
        return 1
    else:
        return 0


# In[ ]:


def class_sort(num_objs):
    """
    给每个类别评分
    :param num_objs:
    :return class_list:
    """
    average, average_list = class_percent(num_objs)
    average_list.sort()
#     print(average_list)
    # class_list : give each class a number according to the quantity
    class_list = {}
    # from small to big
    class_order = sorted(num_objs.items(), key=lambda x: x[1], reverse=False)
#     print(class_order)
    total_length = len(num_objs)
    i = 0
    for obj in class_order:
        class_list[obj[0]] = [total_length, average_list[i]]
        total_length = total_length - 1
        i = i + 1
#     print("class_list:", class_list)
    return class_list


# In[ ]:


def image_sort(class_list,source_source_list):
    """
    给每张图片进行所含类别进行标注，返回的文件列表来源为原始图片组成的列表
    :param class_list:
    :param source_source_list:
    :return image_class: [each image's score list]
    """
    image_list = {}
    for key, values in source_source_list.items():
        # one_image_class : the class of one picture include
        one_image_class = []
        for obj in values:
            one_image_class.append(list(obj.values())[0])
        # save every picture's objects. image_class: [one picture_score list]
        score_list = []
        for obj in one_image_class:
            score = class_list[obj][0]
            score_list.append(score)
        image_list[key] = score_list
    image_list = sorted(image_list.items(), key=lambda x: x[1], reverse=True)
#     print("image_list:", image_list)
    return image_list


# In[ ]:


def get_aug_list(image_list, class_list, AUG_NUMBER):
    """
    得到最终需要增强的数据列表
    :param image_list: 带有标签的文件名列表
    :param class_list: 带有评分的类别列表
    :return aug_image_list: 最终进行增强的文件名列表
    """
    average = 1/len(class_list)
    # get the class over and under average
    aug_list = []
    not_aug_list = []
    for key, values in class_list.items():
        if values[1] < average:
            aug_list.append(values[0])
        else:
            not_aug_list.append(values[0])
    # print(aug_list, not_aug_list)
    # get the aug_image_list witch include AUG_NUMBER class in aug_list and not in not_aug_list
    aug_image_list = []
    for obj in image_list:
        list_b = [b for b in obj[1] if b in not_aug_list]
        list_c = [c for c in obj[1] if c in aug_list[0:AUG_NUMBER]]
        if len(list_b) == 0 and len(list_c) > 0:
            aug_image_list.append(obj[0])
        else:
            continue
    if not aug_image_list:
        for obj in image_list:
            list_d = [d for d in obj[1] if d in aug_list[0:AUG_NUMBER]]
            if len(list_d) > 0:
                aug_image_list.append(obj[0])

#     print("aug_image_list:", aug_image_list)
    return aug_image_list


# In[ ]:


def augmentation(aug_image_list, times, source_file_name, xml_path, image_path):
    """
    对列出的aug_image_list 中的图片进行增强
    :param aug_image_list:需要进行增强的文件列表
    :param times:增强的轮数
    :param source_file_name:带有类别标注的文件名字典
    :return none:
    """
    enhance = [
        horizone_flip_enhance,
        vertical_flip_enhance,
        luminance_enhance,
        translation_enhance,
        rotate_enhance,
        gaussian_enhance,
        crop_enhance
    ]
    for name in aug_image_list:
        # if times <= 5 ,use the original augmentation method
        if times <= 7:
            method = enhance[0:(times+len(enhance)) % len(enhance)+1]
            for method in enhance:
                test_enhance = method
                test_enhance(name, times, xml_path, image_path)
        # if times > 5 ,use the multiple augmentation method
        else:
            img = name
            img_list = []
            for i in range(np.random.randint(2, 5)):
                img = enhance[np.random.randint(0, 4)](img, times, xml_path, image_path)
                img_list.append(img)
            # delete the picture and xml files before the final augmentation in the multiple augmentation
            img_list = img_list[:-1]
            img_list = list(set(img_list))
#             print(img_list)
            for obj in img_list:
                if obj not in source_file_name:
                    os.remove(os.path.join(image_path, obj + ".jpg"))
                    os.remove(os.path.join(xml_path, obj + ".xml"))
                    
                    
# In[ ]:

def goal_class(goal_obj_num, class_list):
    """
    from input to get the final goal class
    :param goal_obj_num : the input number dic
    :param class_list:
    :return:
    """
    final_goal_obj_num = {}
    no_goal_class = []
    for key, value in class_list.items():
        if goal_obj_num[key]:
            final_goal_obj_num[value[0]] = [key, goal_obj_num[key]]
        else:
            no_goal_class.append(key)
    # print(final_goal_obj_num)
    return final_goal_obj_num,no_goal_class


# In[ ]:

def find_class(class_name,image_list, no_goal_class):
    """
    find all image have the goal class [[image have one class],[image have two class],[image have over two class]]
    :param class_name:
    :param image_list:
    :return:
    """
    list1 = []
    list2 = []
    list3 = []
    list2_pack = []
    list3_pack = []
    for obj in image_list:
        if class_name in obj[1]:
            if len(obj[1]) == 1:
                list1.append(obj[0])
            elif len(obj[1]) == 2:
                if len([a for a in obj[1] if a in no_goal_class]) == 0:
                    list2.append(obj[0])
                else:
                    list2_pack.append(obj[0])
            else:
                if len([a for a in obj[1] if a in no_goal_class]) == 0:
                    list3.append(obj[0])
                else:
                    list3_pack.append(obj[0])
    if len(list2) == 0:
        list2 = list2_pack
    if len(list3) == 0:
        list3 = list3_pack
    image_class = [list1, list2, list3]
    return image_class


# In[ ]:

def if_satisfy(value, obj_num):
    # value = [name, goal]
    if obj_num[value[0]] >= int(value[1]):
        print("finish one class")
        return 1
    else:
        return 0
    
# In[ ]:

def round_time(list1, value):
    """
    计算三种含有不同目标物品熟练的图片应该增强的轮数
    """
    one = int(len(list1[0]))
    print(one)
    two = int(len(list1[1]))
    print(two)
    three = int(len(list1[2]))
    print(three)
    total = int(value[1])
    x, y, z = sympy.symbols('x y z')
    dic_round = sympy.solve([one * x + two * y + three * z - total, 3 * x - 7 * y, 3 * y - 7 * z], [x, y, z])
    print(dic_round)
    list_round = []
    for key, values in dic_round.items():
        num = int(values) + 1
        list_round.append(num)
    print(list_round)
    return list_round

def pyecharts_line_bar(x_data, v1, v2, num_objs):
    line = (
    Line()
    .add_xaxis(x_data)
    .add_yaxis(
        "百分比",
        v2,
        yaxis_index=1,    #删了一个Y轴，Y轴索引由2改为1
        color="#d14a61",
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(color="#5793f3")
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            type_="value",
            name="百分比",
            min_=0,
            max_=100,
            position="left",
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color="#5793f3")
            ),
            axislabel_opts=opts.LabelOpts(formatter="{value} %"),
            splitline_opts=opts.SplitLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)
            ),
        )
    )
    .set_global_opts(
        yaxis_opts=opts.AxisOpts(
            name="数量",
            min_=0,
            max_=max(num_objs.values())*1.5,
            position="right",
            offset=0,    #这里是Y轴间距，由80改为0即两个Y轴重合，当然我们已经删除了原来的一个Y轴，所以相当于把第二Y轴左移
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color="#d14a61")
            ),
            axislabel_opts=opts.LabelOpts(formatter="{value} "),
        ),
        title_opts=opts.TitleOpts(title="物品数量百分比展示"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="none"),
        datazoom_opts= [opts.DataZoomOpts(range_start=0, range_end=100,is_zoom_lock=False)]
        )
    
    )

    bar = (
        Bar()
        .add_xaxis(x_data)
        .add_yaxis(
            "物品数量",
            v1,
            yaxis_index=0,
            color="#d14a61",
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position='top'),
                        markpoint_opts=opts.MarkPointOpts(
                    data=[
                        opts.MarkPointItem(type_="max", name="最大值"),
                        opts.MarkPointItem(type_="min", name="最小值"),
                    ]
                ))

    )

    line.overlap(bar)
    grid = Grid()
    grid.add(line, opts.GridOpts(pos_left="5%", pos_right="20%"), is_control_axis_index=True)
#     grid.render_notebook()

    
    return grid


def pyecharts_pie(x_data, v1):
    data_pair = list(zip(x_data,v1))
    pie = (
        Pie()
        .add("", data_pair,center=["40%","50%"])
        .set_global_opts(title_opts=opts.TitleOpts(title="每类物品比例展示"),
                         legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"))
        .set_series_opts(series_name="原始数据单类别占比",label_opts=opts.LabelOpts(formatter="{b}: {c}"))

        #.render("pie_set_color.html")
     )
    return pie


def pyecharts_bar_stack(x_data, source_num_objs, num_objs):
    v1=list(source_num_objs.values())
    v2=list(num_objs.values())
    bar = (
            Bar(opts.InitOpts(width = '1000px',height = '500px'))
            .add_xaxis(x_data)
            .add_yaxis("原始数据", v1, stack="stack1")
            .add_yaxis("增强后数据", v2, stack="stack1")
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="原始数据与增强后数据展示"),
                            datazoom_opts= [opts.DataZoomOpts(range_start=0, range_end=100,is_zoom_lock=False)],
                             legend_opts=opts.LegendOpts(type_="scroll", pos_left="center", orient="horizontal"))
        )
    return bar
    