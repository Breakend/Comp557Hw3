# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:32:06 2015

@author: Fahim
"""
import os, sys
import fnmatch
import numpy as np
from PIL import Image

DEFAULT_IM_DIR1 = './images/'
DEFAULT_IM_DIR2 = './images_sol/'

def cmp_images(im1, im2):
    ''' compare two image represented using numpy array'''
    im_diff = np.abs(im1 - im2)
    im_L1 = np.sum(im_diff)
    im_L2 = np.sum(im_diff**2)
    
    return im_L1, im_L2, im_diff
    
def cmp_image_files(im_file1, im_file2):
    ''' compare two images provided by filename '''
    im1 = np.array(Image.open(im_file1))
    im2 = np.array(Image.open(im_file2))
    if im1.shape != im2.shape:
        raise Exception('image sizes are different', im1.shape, im2.shape)
    return cmp_images(im1, im2)

def cmp_image_dirs(im_dir1, im_dir2, filterStr = '*.png'):
    ''' compare images from two directories if their names are the same'''
    print('comparing directories ' + im_dir1 + ' and ' + im_dir2)
    im_diffs_found = []
    im_compared = []
    for root, subDirs, files in os.walk(im_dir1):
        img_files = fnmatch.filter(os.listdir(root), filterStr)
        for img_file in img_files:
            im1_file_path = os.path.join(root, img_file)
            im2_file_path = os.path.join(im_dir2, img_file)
            
            #print(im1_file_path, im2_file_path, os.path.exists(im2_file_path))
            if os.path.exists(im2_file_path):
                try:
                    result = cmp_image_files(im1_file_path, im2_file_path)
                    print('comparing %s, L1 : %f, L2 : %f' % (img_file, result[0], result[1]))
                    im_compared.append((img_file, result[0], result[1]))
                    if result[0] > 1e-6:
                        im_diffs_found.append((img_file, result[0], result[1]))
                except Exception as e:
                    print('unable to compare %s:' % img_file)
                    print(e)
                    im_diffs_found.append((img_file, e))
            else:
                print('skipping ' + img_file)

    print('-'*40)
    print('Images that differ:')
    for elem in im_diffs_found:
        print elem
    print('-'*40)
    print('Total number of images compared : %d' % len(im_compared))
    
def main():
    #res = cmp_image_files('./test/scene4_rot.png', './test/scene4_rot_cp.png')
    #print(res[0], res[1])
    im_dir1 = DEFAULT_IM_DIR1
    im_dir2 = DEFAULT_IM_DIR2
    if len(sys.argv) == 3:
        im_dir1 = sys.argv[1]
        im_dir2 = sys.argv[2]
    else:
        print('Usage: python %s dir1-path dir2-path' % sys.argv[0])
        raw_input('Press Enter to compare default dirs %s and %s ' %(im_dir1, im_dir2))
    
    cmp_image_dirs(im_dir1, im_dir2)

if __name__ == '__main__':
    main()
     
