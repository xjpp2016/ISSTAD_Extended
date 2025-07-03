# coding=utf-8
import subprocess
import os
import shutil
from PIL import Image

if __name__ == '__main__':
    
    objects_list = ['bottle', 'cable', 'capsule',  'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    data_dir = './data/visa/VisA/'
    maps_dir = './result/maps/VisA/' 

    for object_name in objects_list:

        print(object_name)

        train_data_dir = data_dir + object_name + '/train'  
        test_data_dir = data_dir + object_name + '/test'
        test_label_dir = data_dir + object_name + '/ground_truth'
        object_maps_dir = maps_dir + object_name
             

        #step2 sub-thread
        python_script = './get_result/visa_result.py' 
        process = subprocess.Popen(['python', python_script, 
                                   '--object_name', object_name, 
                                   '--train_data_dir', train_data_dir, 
                                   '--test_data_dir', test_data_dir,
                                   '--test_label_dir', test_label_dir,
                                   '--maps_dir', object_maps_dir])
        process.wait()   