
# coding: utf-8

# In[ ]:


# coding:utf-8

import cv2
import math
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choice
import random
import shutil

# translation augmentation
TRANSLATION_WIDTH = 0.05
TRANSLATION_HEIGHT = 0.05


# In[ ]:


def horizone_flip_enhance(img_name, times, xml_path, image_path):
    """
    水平翻转
    :param img_name: demo:1
    """
    times = str(times)
    def xmlAnnochange(name, img_width, times):
        tree = ET.parse(os.path.join(xml_path, name + '.xml'))
        root = tree.getroot()
        for objs in root.findall('object'):
            location = objs.find('bndbox')
            xmin = location.find('xmin')
            xmax = location.find('xmax')
            temp = xmin.text
            xmin.text = str(img_width - int(xmax.text))
            xmax.text = str(img_width - int(temp))
        tree.write(os.path.join(xml_path, name + '_hor_flip' + '.xml'))

    # print(img_name)
    # TODO: check img exists
    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    _, img_width, _ = img.shape
    hor_img = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(image_path, img_name + "_hor_flip" + ".jpg"), hor_img)
    xmlAnnochange(img_name, img_width, times)
    # print(os.path.join(image_path, img_name + "_hor_flip" + ".jpg"))
    return img_name + "_hor_flip"


def vertical_flip_enhance(img_name, times, xml_path, image_path):
    """
    垂直翻转
    :param img_name: demo:1
    """
    # TODO:
    times = str(times)
    def xmlAnnochange(name, img_height, times):
        tree = ET.parse(os.path.join(xml_path, name + '.xml'))
        root = tree.getroot()
        for objs in root.findall('object'):
            location = objs.find('bndbox')
            ymin = location.find('ymin')
            ymax = location.find('ymax')
            temp = ymin.text
            ymin.text = str(img_height - int(ymax.text))
            ymax.text = str(img_height - int(temp))
        tree.write(os.path.join(xml_path, name + '_ver_flip' + '.xml'))

    # print(img_name)
    # TODO: check img exists
    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    img_height, _ , _ = img.shape
    hor_img = cv2.flip(img, 0)
    cv2.imwrite(os.path.join(image_path, img_name + "_ver_flip" + ".jpg"), hor_img)
    xmlAnnochange(img_name, img_height, times)
    # print(os.path.join(image_path, img_name + "_ver_flip" + ".jpg"))
    return img_name + "_ver_flip"


def luminance_enhance(img_name, times, xml_path, image_path):
    """
    亮度增强 13,  0.9,
    """
    LUMINANCE_ALPHA_LIST = [0.5, 0.6, 0.7, 0.8, 0.9]
    LUMINANCE_BETA_LIST = [7, 8, 9, 10, 11, 12, 13, 14]
    LUMINANCE_ALPHA = choice(LUMINANCE_ALPHA_LIST)
    LUMINANCE_BETA = choice(LUMINANCE_BETA_LIST)
#     print(LUMINANCE_BETA)
#     print(LUMINANCE_ALPHA)
    times = str(times)
    def xmlAnnochange(name, times):
        tree = ET.parse(os.path.join(xml_path, name + '.xml'))
        tree.write(os.path.join(xml_path, name + '_luminance' + '.xml'))

    # print(img_name)
    # TODO: check img exists
    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_HSV)
    v = (v * LUMINANCE_ALPHA + LUMINANCE_BETA).astype(np.uint8)
    lumin_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(image_path, img_name + "_luminance" + ".jpg"), lumin_img)
    xmlAnnochange(img_name, times)
    return img_name + "_luminance"

def translation_enhance(img_name, times, xml_path, image_path):
    """
    平移增强
    """
    times = str(times)
    def xmlAnnochange(name, height, width, times):
        # TODO: finish
        corner_x = int(TRANSLATION_WIDTH * width)
        corner_y = int(TRANSLATION_HEIGHT * height)
        tree = ET.parse(os.path.join(xml_path, name + ".xml"))
        root = tree.getroot()
        for objs in root.findall("object"):
            location = objs.find("bndbox")
            xmin = location.find("xmin")
            xmax = location.find("xmax")
            ymin = location.find("ymin")
            ymax = location.find("ymax")
            corner_xmin = int(xmin.text) + corner_x
            corner_ymin = int(ymin.text) + corner_y
            corner_xmax = int(xmax.text) + corner_x
            corner_ymax = int(ymax.text) + corner_y
            if (corner_xmax > width or corner_ymax > height):
                return False
            else:
                xmin.text = str(corner_xmin)
                xmax.text = str(corner_xmax)
                ymin.text = str(corner_ymin)
                ymax.text = str(corner_ymax)
        tree.write(os.path.join(xml_path, name + "_translation" + ".xml"))
        return True

    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    height, width = img.shape[:2]
    translation_matrix = np.float32([[1, 0, np.int32(width * TRANSLATION_WIDTH)],
                                     [0, 1, np.int32(height * TRANSLATION_HEIGHT)]])
    shifted_img = cv2.warpAffine(
        img, translation_matrix, (img.shape[1], img.shape[0]))
    check = xmlAnnochange(img_name, height, width, times)
    if check == True:
        # save
        cv2.imwrite(os.path.join(image_path, img_name + "_translation" + ".jpg"), shifted_img)
        return img_name + "_translation"
    else:
        print(img_name, "out of picture")
        return img_name

def rotate_enhance(img_name, times, xml_path, image_path):
    """
    旋转增强 
    """
    times = str(times)
    ROTATE_ANGLE_LIST = [-60, -30, -10, 10, 30, 60, 100, 120]
    ROTATE_ANGLE = choice(ROTATE_ANGLE_LIST)
    def bbox_location(xmin, xmax, ymin, ymax, rot_mat):
        point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
        point3 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
        concat = np.vstack((point1, point2, point3, point4)).astype(np.int32)
        # 旋转后坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        return rx, ry, rx + rw, ry + rh

    def xmlAnnochange(name, rotate_matrix, times):
        tree = ET.parse(os.path.join(xml_path, name + ".xml"))
        root = tree.getroot()
        for objs in root.findall("object"):
            location = objs.find("bndbox")
            xmin_label = location.find("xmin")
            xmax_label = location.find("xmax")
            ymin_label = location.find("ymin")
            ymax_label = location.find("ymax")
            # 计算翻转后的推荐框位置
            rx_min, ry_min, rx_max, ry_max =                 bbox_location(int(xmin_label.text), int(xmax_label.text),
                              int(ymin_label.text), int(ymax_label.text), rotate_matrix)
            xmin_label.text = str(rx_min)
            xmax_label.text = str(rx_max)
            ymin_label.text = str(ry_min)
            ymax_label.text = str(ry_max)
        tree.write(os.path.join(xml_path, name + "_rotate" + ".xml"))

    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    theta = ROTATE_ANGLE * np.pi / 180
    width, height = img.shape[1], img.shape[0]
    # 旋转后图片长宽计算
    new_width = abs(np.sin(theta) * height) + abs(np.cos(theta) * width)
    new_height = abs(np.cos(theta) * height) + abs(np.sin(theta) * width)

    # 构建旋转矩阵，并完成图像翻转
    rotate_matrix = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5),
                                            ROTATE_ANGLE, scale=1)
    rotate_move = np.dot(rotate_matrix,
                         np.array([(new_width - width) * 0.5, (new_height - height) * 0.5, 0]))
    rotate_matrix[0, 2] += rotate_move[0]
    rotate_matrix[1, 2] += rotate_move[1]
    rotate_img = cv2.warpAffine(img, rotate_matrix,
                                (int(new_width), int(new_height)))
    cv2.imwrite(os.path.join(image_path, img_name + "_rotate" + ".jpg"), rotate_img)
    xmlAnnochange(img_name, rotate_matrix, times)
    return img_name + "_rotate"


def gaussian_enhance(img_name, times, xml_path, image_path):
    """
    添加高斯噪声
    """
    noise_sigma = 25
    times = str(times)

    def xmlAnnochange(name, times):
        tree = ET.parse(os.path.join(xml_path, name + '.xml'))
        tree.write(os.path.join(xml_path, name + '_gaussian' + '.xml'))

    def add_gaussian_noise(image_in, noise_sigma):
        """
        给图片添加高斯噪声
        image_in:输入图片
        noise_sigma：
        """
        temp_image = np.float64(np.copy(image_in))
        h, w, _ = temp_image.shape
        # 标准正态分布*noise_sigma
        noise = np.random.randn(h, w) * noise_sigma
        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

        return noisy_image

    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    noise_img = add_gaussian_noise(img, noise_sigma=noise_sigma)
    cv2.imwrite(os.path.join(image_path, img_name + "_gaussian" + ".jpg"), noise_img)
    xmlAnnochange(img_name, times)
    return img_name + "_gaussian"


def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text

        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
        result.append(222)

        results.append(result)
    return results


def crop_enhance(img_name, times, xml_path, image_path):
    """
    随机裁剪
    """
    def xmlAnnochange(name, bboxes):
        tree = ET.parse(os.path.join(xml_path, name + '.xml'))
        root = tree.getroot()
        no = 0
        for objs in root.findall('object'):
            location = objs.find('bndbox')
            xmin = location.find('xmin')
            ymin = location.find('ymin')
            xmax = location.find('xmax')
            ymax = location.find('ymax')
            xmin.text = str(bboxes[no][0])
            ymin.text = str(bboxes[no][1])
            xmax.text = str(bboxes[no][2])
            ymax.text = str(bboxes[no][3])
            no += 1
        tree.write(os.path.join(xml_path, name + '_crop' + '.xml'))


    def random_crop(img, bboxes, p=0.5):
        # 随机裁剪
        if random.random() < p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes

    img = cv2.imread(os.path.join(image_path, img_name + ".jpg"), 1)
    bboxes = readAnnotations(os.path.join(xml_path, img_name + ".xml"))
#     print(bboxes)
    crop_img, bboxes = random_crop(img, np.array(bboxes), 1)
    cv2.imwrite(os.path.join(image_path, img_name + "_crop" + ".jpg"), crop_img)
    xmlAnnochange(img_name, bboxes)
    return img_name + "_crop"

