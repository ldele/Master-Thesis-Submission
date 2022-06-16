import pandas as pd
import numpy as np
import re

import SimpleITK as sitk # To read .mhd files
from tqdm import tqdm # Progress bars
import dask_image.imread # Better reading performance of image folder
from skimage import io

####################################################################################
# Seg2Pts
####################################################################################

#This will run for at least 15 minutes for a half brain, should be faster on computer with an optic fiber
def return_3d_coordinates_from_seg(imseq):
    # Initialize 3 list to store the xyz coordinates 
    x, y, z = [], [], []
    
    coordinates = pd.DataFrame(columns=['x','y','z'])
    img_stack = dask_image.imread.imread(imseq + "/*.tif")

    #Loops through all images
    for i in range(img_stack.shape[0]):
        img = img_stack[i].compute() # reads image
        this_plane=np.argwhere(img>0) # get xy indices of any that is 1

        for j in this_plane:
            #append the xy indices to list, append the z to list
            # note that the shape of the image is rows, columns (hence y, x)
            x.append(j[1])
            y.append(j[0])
            z.append(i)
            
    coordinates['x']=x
    coordinates['y']=y
    coordinates['z']=z
    
    return coordinates

#change output_name
def write_atlas_file_from_segdf(df, outdir, currentxyz, goalxyz, mousename):

    currentxyz=5
    goalxyz=25
    ratio= goalxyz/currentxyz
    # Now, we want to reslice this horizontal view to coronal view (so that matches the orientation of our transformation and atlas)
    ds_x=pd.to_numeric(df['x'])//ratio
    ds_y=pd.to_numeric(df['y'])//ratio
    ds_z=pd.to_numeric(df['z'])//ratio
    
    q = [' '.join(i) for i in zip(ds_x.astype(str),ds_y.astype(str),ds_z.astype(str))]
    out_name= f'{mousename}_{goalxyz}um_points.txt'

    num_row=df.shape[0]
    f=open(outdir+'/'+out_name,'w+')
    f.write('point'+'\n')
    f.write(str(num_row)+'\n')
    for lines in q:
        f.write(lines+'\n')
    f.close()

    print(f'saved at {outdir}/{out_name}')


####################################################################################
# vimage - correct and extract
####################################################################################

#Requires the output files from "Seg2Pts" -> files containing 3d coordinates xyz

def make_tif_1(all_points, atlas, outname):
    ''' Project downsampled points on to a tiff stack, useful for overlaping with brain or template (ie, in imageJ) 
    Feb 2022
    Modified for viral images where points are indices after trailmap segmentation with probability 
    Since some noise will be picked up as axons and cross probability threshold, it is possible for transformed points to have negative value or outside of the brain
    points outside of the brain are ignored
    
    Aug 2021
    Slightly modified version where each pixel will have a value of 1, instead of actual number of points.
    Useful for visualization (ie. max projection) so that all pixel axon or dendrite is equally strong in intensity
       
    input: downsampled points in a list containing x y z ordinates as int, directory containing it (this is also the output directory) and whether annotation is axon or not (default True)
    example: [[12, 13, 25],
             [13, 14, 25],...]
    
    output: a tiff stack with the same dimensions of the brain/template/atlas mhd files with downsampled points only
    each point has a value of the number of occurences (since downsampling combines multiple points as one)
    '''
        
    print('Starting to saving tif files..')
    
    atlas_size=atlas.shape
    svolume=np.zeros(atlas_size)
    #columns, rows, planes
    
    zplanes=[]
    for i in all_points:
        zplanes.append( i[2])
    zplanes=np.unique(zplanes)
    temp=np.zeros(atlas_size[0:2])
    thepoints=np.asarray(all_points)

    for i in tqdm(zplanes):
        index= thepoints[:,2]==i
        uindex,counts=np.unique(thepoints[index],return_counts=True, axis=0)
        for lines in uindex:
            coord1,coord2=lines[0:2]
            temp[coord1][coord2]= 1
        svolume[:,:,i]=temp #write this in 
        temp=np.zeros(atlas_size[0:2]) #reset the empty plane after each z
        
    
    coronal_planetmp= np.swapaxes(np.int16(svolume),0,2)
    #for some reason, if just save stuff as tiff, it will save x planes of yz view
    #here we shift the 3rd dimension with the first dimension to obtain xy view
    
    out_name= outname + '.tif'

    io.imsave(out_name, coronal_planetmp)
    return 

def check_points(points_in_atlas):
    '''Checks whether all your points' ID is within the atlas labels
    Input: matching ID of the points (this is the second output from na.make_pd)
    '''
    id_inatlas=[]
    for x in atlas_labels['region_id']:
        intID = int(x)
        id_inatlas.append(intID)

    # need to format this first ourselves,otherwise problematic for 0 and very large numbers (idk why)    
    num_of_zeros = [i for i, x in enumerate(points_in_atlas) if x == 0]
    # find the indices for which carries an id =0
    
    unique_id=set(points_in_atlas)
    
    for id_inbrain in unique_id:
        if id_inbrain not in id_inatlas:
            if id_inbrain==0:
                print(f'There are {len(num_of_zeros)} points with ID= {id_inbrain}, this index is outside of the brain, consider possible suboptimal image registration')
            else: 
                print(id_inbrain,'this index does not exist in allen reference atlas, see https://github.com/ChristophKirst/ClearMap/issues/37')
            warnings.warn('Some points do not have corresponding labels')
    return

####################################################################################
# vimage - results and plots
####################################################################################

def get_pt_natlas(dspoint_name, to_add):
    '''Read downsampled points after Transformix transformation onto
        Allen Atlas domain
    Feb 2022
        Modified for viral images where points are indices after trailmap segmentation with probability 
        Since some noise will be picked up as axons and cross probability threshold, it is possible for transformed points to have negative value or outside of the brain
        points with negative values any one dimensions are ignored
    Apr 2022
        Modified to not include atlas name
        Prefer to select atlas directly
    '''
    with open(dspoint_name,'r') as output:
        outputpoint= output.readlines()
    
    all_points=[]

    for lines in tqdm(outputpoint):
        m=re.search("(?:OutputIndexFixed = \[ )([0-9]+ [0-9]+ [0-9]+)", lines)
        if not m:
            print('negative number in one of the dimension, skipped')
            print(f'{lines}')
            pass
        else:
            m=m.groups(0)
            this_line= str(m[0]).split(' ')
            mypoints= [int(stuff) for stuff in this_line]
            mypoints[1]= mypoints[1]+to_add
            all_points.append(mypoints)
    
    return all_points


def make_tif(all_points, atlas_shape, outname, axon=1):
    ''' Project downsampled points on to a tiff stack, useful for overlaping with brain or template (ie, in imageJ)
    input: downsampled points in a list containing x y z ordinates as int, directory containing it (this is also the output directory) and whether annotation is axon or not (default True)
    example: [[12, 13, 25],
             [13, 14, 25],...]
    
    output: a tiff stack with the same dimensions of the brain/template/atlas mhd files with downsampled points only
    each point has a value of the number of occurences (since downsampling combines multiple points as one)
    ---
    From coordinate system: x, y, z in point list has to be reordered into z, y, x
    '''
        
    print('Starting to saving tif files..')    
    # ----
    
    stack = np.zeros((atlas_size[0],atlas_size[2],atlas_size[1])) #initialize a stack volume the size of the atlas which is in the right orientation
    my_points=np.asarray(all_points)
    zplanes = np.unique(my_points[:,2]) # get all z with a point
    
    for z in zplanes:
        xy_plane = my_points[my_points[:,2]==z] # Take all points on a xy_plane
        unique_points, counts = np.unique(xy_plane, return_counts=True, axis=0) # Due to downsampling, some points appear multiple times
        temp_plane = np.zeros((atlas.shape[2],atlas.shape[1]))
        for xy, u_point in enumerate(unique_points):
            temp_plane[u_point[0]][u_point[1]] = counts[xy]
        stack[z,:,:] = temp_plane
        
    stack=np.swapaxes(np.int16(stack),1,2)
        
    if axon==1:
        out_name=outname + '_axons.tif'
    else:
        out_name=outname + '_dendrites.tif'

    io.imsave(out_name, stack) 
    

def find_points_id(points):
    points_in_atlas=[int(annot_h[i[0], i[1],i[2]]) for i in points]
    
    points_in_atlas= np.where(points_in_atlas==0, 981, points_in_atlas) 
    # replace id= 0 with 981 (ssp-bfd layer1)

    points_in_atlas= np.where(points_in_atlas==484682520, 484682528 , points_in_atlas) 
    points_in_atlas= np.where(points_in_atlas==484682524, 484682528 , points_in_atlas)
    
    # replace id= 484682520 (optic radiation)  and id= 484682524 (auditory radiation) with 484682258, stc(a subregion of fiber bundle)
    # these are intrinsic issue of the allen atlas, the labels for these regions are wrong
    
    points_in_atlas= np.where(points_in_atlas==382, 484682528 , points_in_atlas)
    # replace id= 382, hippocampus CA1 with 484682258, stc(a subregion of fiber bundle), to correct for slight misalignment with atlas
    
    points_in_atlas= np.where(points_in_atlas==81 , 672 , points_in_atlas) 
    points_in_atlas= np.where(points_in_atlas==98 , 672 , points_in_atlas)
    # replace id= 98 (subependymal zone) and 81 (lateral ventricle) with 215(caudoputamen), to correct for slight misalignment with atlas
    
    points_in_atlas= np.where(points_in_atlas==0, 981, points_in_atlas) 
    # somehow this line needs to be ran twice??
    return points_in_atlas

def regions_csv(points_in_atlas, out_name):

    unique_id, counts = np.unique(points_in_atlas, return_counts=True)
    id_withcounts=list(zip(unique_id, counts))

    our_regions=na.atlas_labels.loc[na.atlas_labels['id'].isin (unique_id)]

    new_df= pd.DataFrame(id_withcounts, columns=['id', 'counts'])
    our_regionWcounts=pd.merge(na.atlas_labels, new_df)
    
    our_regionWcounts.to_csv(out_name + 'region_with_counts.csv', index=None, header=True)

    return our_regionWcounts

def parent_df(df):
    # group dataframe by parent id structure
    grouped_pd=df.groupby(['parent_structure_id'],as_index=False).sum()
    d= {'id': grouped_pd.parent_structure_id.astype(int), 'counts': grouped_pd.counts}
    grouped_pd2= pd.DataFrame(data=d)
    result = pd.merge(grouped_pd2, na.atlas_labels, on=["id"])
    result.sort_values(['counts'], ascending=True, inplace=True)
    # result is the final pd

    return result

####################################################################################
# vimage - asada
####################################################################################

def arrange_parent_subregion(parentdf,subregiondf):
    new_order=parentdf.id.to_numpy()
    
    old_order= subregiondf.parent_structure_id.to_numpy()   
    
    print('Old parent region id order is: ', old_order)
    print('New parent region id order is: ', new_order)
    
    
    new_array= np.zeros_like(old_order)
    for i, j in enumerate(new_order):
        new_array[old_order==j]=i
        
    print('Re-arranged subregion order based on parent region that has the greatest number of axon: ', new_array)
    
    subregiondf['new_order']= new_array
    subregiondf.sort_values('new_order', inplace=True)
    
    subregiondf.sort_values(by=['new_order', 'graph_order'], ascending=[True, True] ,inplace=True)

    return subregiondf

def plot_hist(pd_axonL,pd_axonR, out_name):
    ''' 
    Plot horizontal histogram of all points and ending points of axons and dendrites
    Input: pandas dataframe of axon, pandas dataframe of dendrite, mousename
    '''

    fig = make_subplots(
        shared_yaxes=True,
        rows=2, cols=1,
        row_heights=[0.6, 0.5],
        row_titles=['Left axons', 'Right axons', 'Dendrites']
    )
    
    fig.add_trace(
        go.Bar(
        y=pd_axonL['acronym'], x=pd_axonL['counts']/1000, # units now in milimeters
        marker_color='red', #for future, pd_axon['region_id'],
        name='',
        text=pd_axonL['name'],
        hovertemplate=
            '<i>%{x}</i>, '+
            '<b>%{text}</b>',
        orientation='h'),
        row=1,col=1
    )
    
    fig.add_trace(
        go.Bar(
        y=pd_axonR['acronym'], x=pd_axonR['counts']/1000, # units now in milimeters
        marker_color='magenta', #for future, pd_axon['region_id'],
        name='',
        text=pd_axonR['name'],
        hovertemplate=
            '<i>%{x}</i>, '+
            '<b>%{text}</b>',
        orientation='h'),
        row=2,col=1
    )
    

   
    fig.update_layout(yaxis={'categoryorder':'trace'}, 
                      width=2000,
                      height=1000, # 1500 for AL066 since too many items
                      showlegend= False,
                      paper_bgcolor='rgba(0,0,0,0)', # transparent background
                      plot_bgcolor='rgba(0,0,0,0)' # transparent background
                     )
    
    fig.update_xaxes(gridcolor='gold')
    
    fig.show()

    fig.write_image(f"{out_name}.svg")
    fig.write_html(f"{out_name}.html")