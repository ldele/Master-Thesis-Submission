import os
import sys
import glob
import tkinter.filedialog as fdialog #window to select directories
from tqdm import tqdm #progress bars

from datetime import datetime

import shutil
import time
import cv2
from PIL import Image

import re
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cc3d

from skimage import io, morphology, measure, filters
from skimage.measure import block_reduce
import dask_image.imread

from scipy import ndimage

####################################################################################
# Utilities
####################################################################################

def print_allocation_size(object_):
    print('Object allocation size:', round(sys.getsizeof(object_)/1073741824, 3),'GB')

####################################################################################
# Access to data
####################################################################################

def get_dir(path):
    tiffs = [os.path.join(path, f) for f in os.listdir(path) if f[0] != '.']
    return sorted(tiffs)

"""
def from_folder_to_stack(input_folder):
    img_list = get_dir(input_folder)
    img_stack = []
    for count, item in enumerate(img_list):
        img_stack.append(cv2.imread(item, -1))
    return np.array(img_stack)
"""

def from_folder_to_stack(input_folder, ext=".tif"):
    return dask_image.imread.imread(input_folder + "/*"+ ext).compute() #a bit faster, multithreading

def from_stack_to_folder(tiff_stack_file, output_folder, filename_tmp, extension=r'.tif'):
    #Can be extremely RAM intensive
    if os.path.exists(output_folder):
        print(output_folder + " already exists. Will be overwritten.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    image_stack = io.imread(tiff_stack_file)
    print("Stack dimensions:", image_stack.shape)
    for count, img in enumerate(image_stack):
        pil_image = Image.fromarray(img)    
        pil_image.save(f'{output_folder}/{filename_tmp}_{100000 + count}{extension}')
        
def from_vol_to_folder(vol, stack_name, output_folder, overwrite=False):
    
    if overwrite:
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten.")
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        else:
            os.makedirs(output_folder)
            print("New directory created at:", output_folder)
    else: 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print("New directory created at:", output_folder)
            
    print("Stack dimensions:", vol.shape)
    if os.path.exists(output_folder):
        print(output_folder + " already exists. Will be overwritten.")
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    for count, img in enumerate(vol):
        pil_image = Image.fromarray(img)    
        pil_image.save(output_folder + "/" + str(100000 + count) + "_" + os.path.basename(stack_name))

def read_tiff_stack(path):
    # From TRAILMAP/utilities
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)

    return np.array(images)

def write_tiff_stack(vol, fname):
    # From TRAILMAP/utilities
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)
        
        
####################################################################################
# Main functions
####################################################################################

def downscale_imgs(in_folder, out_folder, factor=(10,10,1)):
    images = get_dir(in_folder)
    
    for _,item in enumerate(tqdm(images)):
        img = cv2.imread(item, -1) #black/white
        out_img = block_reduce(img, block_size=factor, func=np.mean)
        out_img = Image.fromarray(out_img)
        out_img.save(out_folder + "//" + "ds_" + os.path.basename(item))
        
        
def downscale_imgs_allen(in_folder, out_folder, output_filename, factor=(5/25,5.3/25,5.3/25)):
    img_stack = dask_image.imread.imread(in_folder + "/*.tif").compute()
    img_stack = ndimage.zoom(img_stack, zoom=factor)
    print(f'Saving downscaled stack at {output_filename}...')
    write_tiff_stack(img_stack, output_filename)
    print(f'Saving done !')
    return img_stack

            
def skeletons_3d(input_folder, output_folder_loc, ext=r'.tif'):
    
    skel_count = 0
    folder_list = []
    thresh_list = 0.1*np.array(range(2,10,1))
            
    output_name = "skel-" + os.path.basename(input_folder)
    output_folder = os.path.join(output_folder_loc, output_name)
    if os.path.exists(output_folder):
        print(output_folder + " already exists. Will be overwritten.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder) 
    
    for thresh in thresh_list:
        dask_stack = dask_image.imread.imread(input_folder + "/*" + ext)
        dask_stack[dask_stack>=thresh] = 1
        dask_stack[dask_stack<thresh] = 0
        img_stack = dask_stack.astype('uint8').compute()
        img_stack = morphology.skeletonize_3d(img_stack) # max output from skeletonize is 255 (0->255), uint8 data type
        print("Skeleton for threshold", thresh, "computed !")
        folder_save = output_folder + '/skel_thresh_' + str(skel_count)
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
            print(folder_save, "directory created !")
        for count in range(dask_stack.shape[0]):
            pil_image = Image.fromarray(img_stack[count])
            pil_image.save(folder_save + '/' + os.path.basename("skeleton_image_" + str(100000 + count) + ".tif"))
        print("Images for threshold", thresh, "saved !")
        skel_count += 1
        folder_list.append(folder_save)
        
    list_files=get_dir(folder_list[0])
    folder_save = output_folder + "/weighted_sum_skeleton"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        print(folder_save, "directory created !")
    for i in range(len(list_files)):
        img_f = np.multiply(thresh_list[0],cv2.imread(folder_list[0] + "/" + os.path.basename(list_files[i]), cv2.COLOR_BGR2GRAY))
        for count, folder in enumerate(folder_list[1:]):
            img_f = np.add(img_f, np.multiply(thresh_list[count+1],cv2.imread(folder + "/" + os.path.basename(list_files[i]), cv2.COLOR_BGR2GRAY))).astype('float32')
            pil_img = Image.fromarray(img_f)
            pil_img.save(folder_save + '/def-' + os.path.basename(list_files[i]))
        
    return folder_save


def skeletons_3d_bis(input_folder, output_folder_loc, ext=r'.tif'):
    
    skel_count = 0
    folder_list = []
    thresh_list = np.array(0.1*np.array(range(2,10,1))).astype('float32')
            
    output_name = "skel-" + os.path.basename(input_folder)
    output_folder = os.path.join(output_folder_loc, output_name)
    if os.path.exists(output_folder):
        print(output_folder + " already exists. Will be overwritten.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder) 
    
    for thresh in thresh_list:
        dask_stack = dask_image.imread.imread(input_folder + "/*" + ext)
        dask_stack[dask_stack>=thresh] = 1
        dask_stack[dask_stack<thresh] = 0
        img_stack = dask_stack.astype('uint8').compute()
        img_stack = morphology.skeletonize_3d(img_stack) # max output from skeletonize is 255 (0->255), uint8 data type
        print("Skeleton for threshold", thresh, "computed !")
        folder_save = output_folder + '/skel_thresh_' + str(skel_count)
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
            print(folder_save, "directory created !")
        for count in range(dask_stack.shape[0]):
            pil_image = Image.fromarray(img_stack[count])
            pil_image.save(folder_save + '/' + os.path.basename("skeleton_image_" + str(100000 + count) + ".tif"))
        print("Images for threshold", thresh, "saved !")
        skel_count += 1
        folder_list.append(folder_save)
        
    list_files=get_dir(folder_list[0])
    folder_save = output_folder + "/weighted_sum_skeleton"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        print(folder_save, "directory created !")
    dask_stacks = []
    for _, folder in enumerate(folder_list):
        dask_stacks.append(dask_image.imread.imread(folder + "/*" + ext))
    for i in range(len(list_files)):
        img_f = np.multiply(thresh_list[0],dask_stacks[0][i].compute())
        for count, folder in enumerate(folder_list[1:]):
            img_f = np.add(img_f, np.multiply(thresh_list[count+1],dask_stacks[count+1][i].compute()))
            pil_img = Image.fromarray(img_f)
            pil_img.save(folder_save + '/def-' + os.path.basename(list_files[i]))
        
    return folder_save

def skeletons_3d_sep(input_folder, output_folder_loc, ext=r'.tif'):
    
    skel_count = 0
    folder_list = []
    thresh_list = 0.1*np.array(range(2,10,1))
            
    output_name = "skel-" + os.path.basename(input_folder)
    output_dir = os.path.dirname(output_folder_loc)
    output_folder = os.path.join(output_dir, output_name)
    if os.path.exists(output_folder):
        print(output_folder + " already exists. Will be overwritten.")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder) 
    
    for thresh in thresh_list:
        dask_stack = dask_image.imread.imread(input_folder + "/*" + ext)
        dask_stack[dask_stack>=thresh] = 1
        dask_stack[dask_stack<thresh] = 0
        img_stack = dask_stack.astype('uint8').compute()
        img_stack = morphology.skeletonize_3d(img_stack) # max output from skeletonize is 255 (0->255), uint8 data type
        print("Skeleton for threshold", thresh, "computed !")
        folder_save = output_folder + '/skel_thresh_' + str(skel_count)
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
            print(folder_save, "directory created !")
        for count in range(dask_stack.shape[0]):
            pil_image = Image.fromarray(img_stack[count])
            pil_image.save(folder_save + '/' + os.path.basename("skeleton_image_" + str(count) + ".tif"))
        print("Images for threshold", thresh, "saved !")
        skel_count += 1
        folder_list.append(folder_save)
        
    return folder_list

def weighted_sum_skeleton(folder_list, thresh_list, output_folder, ext=r'.tif'):
    if len(folder_list) != len(thresh_list):
        print('Error ! Input of different sizes.)')
        return
    list_files=get_dir(folder_list[0])
    folder_save = output_folder + "/weighted_sum_skeleton"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        print(folder_save, "directory created !")
    dask_stacks = []
    for _, folder in enumerate(folder_list):
        dask_stacks.append(dask_image.imread.imread(folder + "/*" + ext))
    for i in range(len(list_files)):
        img_f = np.multiply(thresh_list[0],dask_stacks[0][i].compute())
        for count, folder in enumerate(folder_list[1:]):
            img_f = np.add(img_f, np.multiply(thresh_list[count+1],dask_stacks[count+1][i].compute()))
            pil_img = Image.fromarray(img_f)
            pil_img.save(folder_save + '/def-' + os.path.basename(list_files[i]))
            

def skeleton_3d_one_threshold(input_file_names, output_folder, threshold):
    img_stack = [None]*len(input_file_names)
    for count, item in enumerate(tqdm(input_file_names)):
        img = cv2.imread(item, cv2.COLOR_BGR2GRAY)
        img[img >= threshold] = 1
        img[img < threshold] = 0
        img_stack[count] = img
        
    img_stack = morphology.skeletonize_3d(np.array(img_stack))
    for count, item in enumerate(input_file_names):
        pil_image = Image.fromarray(img_stack[count])
        pil_image.save(output_folder + "/skel_" + os.path.basename(item))
    
    return img_stack

####################################################################################
# Trimming and looking at skeletons
####################################################################################

def find_biggest_elements(stack_filename, min_threshold):
    """
    Find the Connected-Components of an image stack 
        that have more voxels that the specified threshold.

    Parameters
    ----------
    first : string
        name of the folder containing image sequence
    second : integer
        threshold. minimal size of C-Cs

    Returns
    -------
    python list
        list containing the indexes of connected components (their labels)
        first detected C-C has a label 1 (bg has 0), etc...
    propsa skimage type object
        an object that contains information about connected-components
    """
    label_stack = from_folder_to_stack(stack_filename) # loading stack a first time for connected-components labeling
    label_stack = label_stack.view(dtype=np.float32) # forces a specific size (should be native size after segmentation) (check if worth doing)
    label_stack[label_stack > 0] = 1 # binarization, checking below 0 should be unnecessary (if proper workflow order)
    label_stack, N = cc3d.connected_components(label_stack, out_dtype=np.uint32, return_N=True) # better than skimage ?
    #label_stack = measure.label(label_stack, background=0) #computationally expensive
    print('Individual "blobs":', N)
    
    propsa = measure.regionprops(label_stack)
    del label_stack #liberate memory
    
    index_list = []
    for count, item in enumerate(propsa):
        if len(item.coords) >= min_threshold:
            index_list.append(count)
            
    print(f'Blobs of size superior to {min_threshold}:', len(index_list))
                        
    return index_list, propsa


def print_biggest_connected_components(indexes, propsa):
    max_size = 0
    for index in indexes:
        ccompo_size = len(propsa[index].coords)
        print("Index number:", index, "size:", ccompo_size)


####################################################################################
# Registration and Transformation
####################################################################################
import itk

def Registration(tempdir, fixed_img_name, moving_img_name, parameter_filename_1, parameter_filename_2):
    """
    """    
    fixed_image = itk.imread(fixed_img_name, itk.F) # itk.F is float32 (itk.D for float64/Double)
    moving_image = itk.imread(moving_img_name, itk.F) # code does not work if you don't specifiy data type
    
    parameter_object = itk.ParameterObject.New() # create a parameter object to load parameter maps
    parameter_object.AddParameterFile(parameter_filename_1)
    parameter_object.AddParameterFile(parameter_filename_2)
    
    elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_image, moving_image) # Elastix filter object
    elastixImageFilter.SetOutputDirectory(tempdir)
    elastixImageFilter.SetParameterObject(parameter_object)
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.UpdateLargestPossibleRegion() # performs registration
    
    result_image = elastixImageFilter.GetOutput()
    result_transform_parameters = elastixImageFilter.GetTransformParameterObject()
    
    itk.imwrite(result_image, tempdir + f'/registered_img.mhd')
    for index in range(result_transform_parameters.GetNumberOfParameterMaps()):
        parameter_map = result_transform_parameters.GetParameterMap(index)
        result_transform_parameters.WriteParameterFile(parameter_map, tempdir + f"/Parameters.{index}.txt")
        
    print('Elastix done !')
            
    
def Transformation(tempdir, fixed_img_name, moving_img_name, parameter_filename_1, parameter_filename_2, t0_filename, points_filename):
       
    fixed_image = itk.imread(fixed_img_name, itk.F) # cropped atlas

    parameter_object = itk.ParameterObject.New() # parameter object to load parameter maps
    parameter_object.AddParameterFile(parameter_filename_1)
    parameter_object.AddParameterFile(parameter_filename_2)
    
    elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_image, fixed_image) # Elastix filter object
    elastixImageFilter.SetOutputDirectory(tempdir)
    elastixImageFilter.SetParameterObject(parameter_object)
    elastixImageFilter.SetInitialTransformParameterFileName(t0_filename)
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.UpdateLargestPossibleRegion() # performs registration
    
    inverse_transform_parameters = elastixImageFilter.GetTransformParameterObject()
    inverse_transform_parameters.SetParameter(0, "InitialTransformParametersFileName", "NoInitialTransform") # Otherwise infinite loop
            
    
    moving_image_transformix = itk.imread(moving_img_name, itk.F) # Don't know if it is still required but ignored: would be the downsampled brain
    
    transformixImageFilter = itk.TransformixFilter.New(moving_image_transformix) # Still need to give the moving image but only to get some parameters such as image dimensions.
    transformixImageFilter.SetFixedPointSetFileName(points_filename)
    transformixImageFilter.SetTransformParameterObject(inverse_transform_parameters)
    #transformixImageFilter.SetComputeDeformationField(True)
    transformixImageFilter.SetOutputDirectory(tempdir)
    transformixImageFilter.UpdateLargestPossibleRegion()
    
    print('Transformix done !')
    
    
def Inverse_Transformation(tempdir, fixed_img_name, parameter_filename_1, parameter_filename_2, t0_filename):
        
    fixed_image = itk.imread(fixed_img_name, itk.F)

    parameter_object = itk.ParameterObject.New() # parameter object to load parameter maps
    parameter_object.AddParameterFile(parameter_filename_1)
    parameter_object.AddParameterFile(parameter_filename_2)
    
    elastixImageFilter = itk.ElastixRegistrationMethod.New(fixed_image, fixed_image) # Elastix filter object
    elastixImageFilter.SetOutputDirectory(tempdir)
    elastixImageFilter.SetParameterObject(parameter_object)
    elastixImageFilter.SetInitialTransformParameterFileName(t0_filename)
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.UpdateLargestPossibleRegion() # performs registration
    
    inverse_transform_parameters = elastixImageFilter.GetTransformParameterObject()
    inverse_transform_parameters.SetParameter(0, "InitialTransformParametersFileName", "NoInitialTransform") # Otherwise infinite loop
    
    for index in range(inverse_transform_parameters.GetNumberOfParameterMaps()):
        parameter_map = inverse_transform_parameters.GetParameterMap(index)
        inverse_transform_parameters.WriteParameterFile(parameter_map, tempdir + f"/Parameters_inverseTransfo_.{index}.txt")
       
    print('Inverse transform successfully generated !')
    
    
def Transformation_noInverse(tempdir, points_filename, moving_img_name, parameter_filename_1, parameter_filename_2):
        
    moving_image_transformix = itk.imread(moving_img_name, itk.F) # Don't know if it is still required
    
    parameter_object = itk.ParameterObject.New() # parameter object to load parameter maps
    parameter_object.AddParameterFile(parameter_filename_1)
    parameter_object.AddParameterFile(parameter_filename_2)
    
    parameter_object.SetParameter(0, "InitialTransformParametersFileName", "NoInitialTransform") # Otherwise infinite loop
    
    transformixImageFilter = itk.TransformixFilter.New(moving_image_transformix) # Still need to give the moving image but only to get some parameters such as image dimensions.
    transformixImageFilter.SetFixedPointSetFileName(points_filename)
    transformixImageFilter.SetTransformParameterObject(parameter_object)
    #transformixImageFilter.SetComputeDeformationField(True)
    transformixImageFilter.SetOutputDirectory(tempdir_transformix)
    transformixImageFilter.UpdateLargestPossibleRegion()
    
    print('Transformix done !')
    

def compute_transformation(output_directory, moving_img_autof, points_filename):
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string
    """
    fixed_img_cropped_refatlas = r"/home/lucasdelez/Documents/AllenBrainCCFv3/template_65.tif"
    param1 = r'/home/lucasdelez/Documents/master_project/elastix_params/clearmap_params/align_affine.txt'
    param2 = r'/home/lucasdelez/Documents/master_project/elastix_params/clearmap_params/align_bspline.txt'
    Registration(output_directory, fixed_img_cropped_refatlas, moving_img_autof, param1, param2)
    
    tfx_output_directory = output_directory + r'/Transformation'
    if not os.path.exists(tfx_output_directory):
        os.makedirs(tfx_output_directory)
        print(tfx_output_directory, "directory created !")
    t0_filename = output_directory + r'/TransformParameters.1.txt'
    Transformation(tfx_output_directory, fixed_img_cropped_refatlas, moving_img_autof, param1, param2, t0_filename, points_filename)
    

def compute_transformation_wo_registration(output_directory, moving_img_autof, points_filename):
    """
    My numpydoc description of a kind
    of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string
    """
    fixed_img_cropped_refatlas = r"/home/lucasdelez/Documents/AllenBrainCCFv3/template_65.tif"
    param1 = r'/home/lucasdelez/Documents/master_project/elastix_params/clearmap_params/align_affine.txt'
    param2 = r'/home/lucasdelez/Documents/master_project/elastix_params/clearmap_params/align_bspline.txt'
    
    tfx_output_directory = output_directory + r'/Transformation'
    if not os.path.exists(tfx_output_directory):
        os.makedirs(tfx_output_directory)
        print(tfx_output_directory, "directory created !")
    t0_filename = output_directory + r'/TransformParameters.1.txt'
    Transformation(tfx_output_directory, fixed_img_cropped_refatlas, moving_img_autof, param1, param2, t0_filename, points_filename)
    

####################################################################################
# Demo
####################################################################################

def disc(r):
    d = 2*r + 1
    rx, ry = d/2, d/2
    x, y = np.indices((d, d))
    return ((np.hypot(rx - x, ry - y)-r) < 0.5).astype(int)

def demo_skel_circle(r):
    circle = disc(r)
    skeleton = morphology.skeletonize(circle).astype(int)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plt.imshow(circle,cmap='gray')
    plt.subplot(122)
    plt.imshow(skeleton,cmap='gray')
    plt.tight_layout()
    plt.show()

def demo_skel_blob():
    n = 12
    l = 256
    np.random.seed(12)
    im = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(int), (points[1]).astype(int)] = 1
    im = filters.gaussian(im, sigma= l / (4. * n))
    blobs = im > 0.7 * im.mean()
    
    skeletal_blobs = morphology.skeletonize(blobs).astype(int)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plt.imshow(blobs, cmap='gray')
    plt.subplot(122)
    plt.imshow(skeletal_blobs, cmap='gray')

    plt.tight_layout()
    plt.show()