################################################################
# Libraries
################################################################
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
################################################################



################################################################
class EcoLoc():
    '''
    Main Class to define cropped and cleaned ecographic images.
    It requires the input location of the eco to crop and analyze.
    '''
    
    ############################################################
    # __init__ class:
    def __init__(self, 
                 input_local_path, 
                 no_watermark_local_path = '1_NoWatermark', 
                 cropped_local_path = '2_CroppedOutput', 
                 squared_local_path = '3_SquaredOutput'
                 ):
            '''
            Parameters
            ----------
            input_local_path : str
                Local path to the input folder containing the eco images.
            no_watermark_local_path : str, optional
                    Local path to the output folder containing the eco images 
                    without watermarks    
            cropped_local_path : str, optional
                Local path to the output folder containing the eco images 
                cropped
            squared_local_path : str, optional
                Local path to the output folder containing the eco images 
                cropped and trasformed to be without black borders.
            '''

            # Define the class attributes
            self.input_path = os.path.join(os.getcwd(), str(input_local_path))
            self.no_watermark_path = os.path.join(os.getcwd(),'Outputs', 
                                                  str(no_watermark_local_path))
            self.cropped_path = os.path.join(os.getcwd(),'Outputs', 
                                             str(cropped_local_path))
            self.cropped_path_res = os.path.join(os.getcwd(),'Outputs', 
                                            str(cropped_local_path)+"_squared")
            self.squared_path = os.path.join(os.getcwd(), 'Outputs', 
                                             str(squared_local_path))
            
            # Generate the folder (if not existent):
            if os.path.isdir(self.input_path) == False:
                print("Warning, not existent input directory. \
                                              Check the spelling.")
            
            # Make the output folder
            if os.path.isdir(os.path.join(os.getcwd(), 'Outputs')) == False:
                os.mkdir(os.path.join(os.getcwd(), 'Outputs'))

            # Generate the list of candidate images:
            self.input_image_list = os.listdir(self.input_path)
            
            try:
                self.input_image_list = [x for x in self.input_image_list if x.split('.')[-1]  
                                                         in ['jpeg', 'png', 'jpg']]
            except:
                pass

            
            # Check dimensions
            if len(self.input_image_list ) == 0:
                print("Warning: The number of echo contained in the input \
                                                     folder seems equal to 0.")
    ############################################################



    ############################################################
    def pre_remove(self, 
                   path_pre_remove = 'ImageToPreProcess',
                   black_threshold = 5,
                   reconstruct = True, 
                   color_threshold = 5):
        '''
        Method for the preliminary removal of colour from images 
        (characterised in that the watermarks overlap the ecographic cone).
        
        
        Parameters
        ----------
        path_pre_remove : string, optional
            Folder containing the images to pre_remove.
        reconstruct : boolean, optional
            If True, it tries to reconstruct removed pixels 
            from the surrounding
        black_threshold : int, optional
            Convert each pixel with value [0,black_threshold] to 0.
        color_threshold : int, optional
            Threshold used to define when a pixel is a grayscale value or RGB one.
        '''
        
        # Initialize the path
        self.path_pre_remove = os.path.join(os.getcwd(), str(path_pre_remove))
        
        list_images =os.listdir(os.path.join(os.getcwd(),self.path_pre_remove))
        
        try:
            list_images = [x for x in list_images if x.split('.')[-1]
                                                     in ['jpeg', 'png', 'jpg']]
        except:
            pass
        
        if os.path.isdir('InputEco')==False:
            os.mkdir('InputEco')
            
        for temp_image_name in tqdm(list_images):
            try:
                im_temp =  Image.open(os.path.join(self.path_pre_remove, 
                                                              temp_image_name)).convert('RGB')

                # Operations:
                im_temp_np = np.array(im_temp)      # Original image
                im_mask = np.array(im_temp)[:,:,0]  # Mask to find colors
                
                
                # Aux function to find color and gray pixels
                def test_color(a,b,c,threshold=color_threshold):
                    if max( abs(round(a)-round(b)),abs(round(a)-round(c)), abs(round(b)-round(c)))>threshold:
                        return True
                    else:
                        return False
                    
                # Aux function to find the first original pixel
                def close_pixel(i, j, im_temp_np, im_mask, direction):
                    i_selected = i
                    j_selected = j
                    
                    
                    found = False
                    try:
                        if direction == 'bottom':
                            # Bottom
                            while found == False and i_selected<np.shape(im_temp_np)[0]:
                                i_selected += 1
                                if im_mask[i_selected, j_selected] ==0:
                                    found = True
                        elif direction == 'top':
                            # top
                            while found == False and i_selected>0:
                                i_selected -= 1
                                if im_mask[i_selected, j_selected] ==0:
                                    found = True
                        elif direction == 'left':
                            # lef
                            while found == False and j_selected>0:
                                j_selected -= 1
                                if im_mask[i_selected, j_selected] ==0:
                                    found = True
                        elif direction == 'right':
                            # right
                            while found == False and j_selected<np.shape(im_temp_np)[1]:
                                j_selected += 1
                                if im_mask[i_selected, j_selected] ==0:
                                    found = True
                    except:
                        found = False
                        
                        
                    
                    if found:    
                        return np.array(im_temp_np[i_selected, j_selected,:])
                    else:
                        return np.array([0,0,0])
                
            
                # Populate the mask
                for i_temp in range(np.shape(im_mask)[0]):
                    for j_temp in range(np.shape(im_mask)[1]):
                        
                        # If color:
                        if test_color(im_temp_np[i_temp,j_temp,0],im_temp_np[i_temp,j_temp,1],im_temp_np[i_temp,j_temp,2]):
                            im_mask[i_temp,j_temp] = 255
                        else:
                            im_mask[i_temp,j_temp] = 0
            
                
            
                # Use the mask to filter the original image
                for i_temp in range(np.shape(im_temp_np)[0]):
                    for j_temp in range(np.shape(im_temp_np)[1]):                
                        if im_mask[i_temp,j_temp] == 255:
                            
                            if reconstruct:
                                # Find the closest 4 points without the null mask
                                im_temp_np[i_temp,j_temp,:] = (close_pixel(i_temp, j_temp, im_temp_np, im_mask, 'right')+\
                                                                close_pixel(i_temp, j_temp, im_temp_np, im_mask, 'left')+\
                                                                close_pixel(i_temp, j_temp, im_temp_np, im_mask, 'top')+\
                                                                close_pixel(i_temp, j_temp, im_temp_np, im_mask, 'bottom'))/4
                            else:
                                im_temp_np[i_temp,j_temp,:] = [0,0,0]
                
              
                # Convert grayscale:
                im_temp_np = im_temp_np[:,:,0]    
                im_temp_np[im_temp_np<black_threshold] = 0
                
                # Reconvert to image
                im_temp = Image.fromarray(np.uint8(im_temp_np)).convert('RGB')
                
                # Save image in the input Folder:
                im_temp.save(os.path.join(os.getcwd(), 'InputEco',\
                                          temp_image_name))
                # Generate the list of candidate images:
                self.input_image_list = os.listdir(self.input_path)
            except:
                print(temp_image_name)
    ############################################################

     

    ############################################################
    def no_watermark(self, threshold = 0, iteration = 3, 
                     max_rec = 500, how_many_points = 10): 
        '''
        Method which takes the raw images inside the InputEco folder and 
        removes every disconnected component but the central ecographic image
        using a WaterShed algorithm. 
        
        Parameters
        ----------
        threshold : int, optional
            Numerical threshold of grayscale values. Pixels lower than this 
            value will be set to 0. 
            
        iteration : int, optional
            Number of iterative steps (i.e., restart the method from pixels
            reached by the previous step once the maximum recursion is reached)
            Increase lineary the cost but increase precision too.
            
        max_rec = 500 : int, optional
            Maximum number of recursive call for each starting pixel.
        
        how_many_points = 10 : int, optional
            Size of the square of non-null pixels included to start the 
            watershed algorithm.
    
        '''     
        
        if os.path.isdir(self.no_watermark_path) == False:
            # Generate the folder:
            os.mkdir(self.no_watermark_path)

        # The list of images is input_image_list:
        for temp_image_name in tqdm(self.input_image_list):
            
            # Import image
            temp_image = Image.open(os.path.join(self.input_path, 
                                                 temp_image_name))
            
            # Convert to numpy greyscale
            temp_image_np = np.asarray(temp_image)[:,:,0]       
            
            # Watershed on segmentation
            # Create a mask  (an empty array to be filled)
            global mask_array_np
            mask_array_np = np.zeros((np.shape(temp_image_np)[0], 
                                      np.shape(temp_image_np)[1]))
            
            # Find a candidate center of the image:
            center_coordinates_x = int(np.shape(temp_image_np)[0]/2)
            center_coordinates_y = int(np.shape(temp_image_np)[1]/2)
                
            global global_last_seeds_list
            global_last_seeds_list = []  # List of the last positions 
            
            # Set the center value (center and close (values) as non null in the mask:
            for i_x in range(-how_many_points, how_many_points+1):
                for j_y in range(-how_many_points, how_many_points+1):
                    mask_array_np[center_coordinates_x+i_x, 
                                  center_coordinates_y+j_y] = temp_image_np[
                                      center_coordinates_x+i_x, 
                                      center_coordinates_y+j_y]
                    global_last_seeds_list.append([center_coordinates_x+i_x, \
                                                  center_coordinates_y+j_y])
            # Define an iterative function: 
            global visited_mask
            visited_mask = np.zeros((np.shape(temp_image_np)[0], 
                                     np.shape(temp_image_np)[1])) 

            def fun_iterative_worm(starting_x, starting_y, start = True, 
                                   threshold = threshold, actual_rec=0):
                actual_rec += 1
                # Define a maximum number of internal recursion
                if actual_rec < max_rec:
                    if start == True:
                        mask_array_np[starting_x, starting_y] = \
                                        temp_image_np[starting_x, starting_y]
                        # If the value is true, it starts from there:
                        visited_mask[starting_x, starting_y] = 1
                        
                        # Go bottom:
                        if visited_mask[starting_x+1, starting_y]==0 and \
                                (starting_x+1 < np.shape(temp_image_np)[0]-1):
                            fun_iterative_worm(starting_x+1, starting_y, 
                                        start = False, actual_rec = actual_rec)
                            
                        # Go top
                        if visited_mask[starting_x-1, starting_y]==0 and \
                                                            (starting_x-1 >0):
                            fun_iterative_worm(starting_x-1, starting_y, 
                                        start = False, actual_rec = actual_rec)
                            
                        # Go right:
                        if visited_mask[starting_x, starting_y+1]==0 and  \
                                (starting_y+1 < np.shape(temp_image_np)[1]-1):
                            fun_iterative_worm(starting_x, starting_y+1, 
                                        start = False, actual_rec = actual_rec)
                            
                        # Go left:
                        if visited_mask[starting_x, starting_y-1]==0 and \
                                                            (starting_y-1 >0):
                            fun_iterative_worm(starting_x, starting_y-1, 
                                        start = False, actual_rec = actual_rec)
   
                    elif start == False:
                        # This is a candidate, sign it as visited to speed up
                        visited_mask[starting_x, starting_y] = 1
                        
                        # If it is over the threshold, save it and iterate:
                        if temp_image_np[starting_x, starting_y]>threshold:
                            mask_array_np[starting_x, starting_y] =  \
                                          temp_image_np[starting_x, starting_y]
                        
                            # Go bottom:
                            if visited_mask[starting_x+1, starting_y]==0 and \
                                 (starting_x+1 < np.shape(temp_image_np)[0]-1):
                                fun_iterative_worm(starting_x+1, starting_y, 
                                        start = False, actual_rec = actual_rec)    
                                
                            # Go top
                            if visited_mask[starting_x-1, starting_y]==0 and \
                                                             (starting_x-1 >0):
                                fun_iterative_worm(starting_x-1, starting_y, 
                                        start = False, actual_rec = actual_rec)    
                                
                            # Go right:
                            if visited_mask[starting_x, starting_y+1]==0 and \
                                 (starting_y+1 < np.shape(temp_image_np)[1]-1):
                                fun_iterative_worm(starting_x, starting_y+1, 
                                        start = False, actual_rec = actual_rec)                        
                                
                            # Go left:
                            if visited_mask[starting_x, starting_y-1]==0 and \
                                                             (starting_y-1 >0):
                                fun_iterative_worm(starting_x, starting_y-1, 
                                        start = False, actual_rec = actual_rec)
                else:
                    # Append the new starting seeds:
                    global_last_seeds_list.append([starting_x, starting_y])

            # First worm iteration:    
            fun_iterative_worm(center_coordinates_x, center_coordinates_y)
            
            # Next worm iterations
            for temp_it in range(iteration):
                # Check over the seeds
                temp_list = global_last_seeds_list
                
                for seeds in temp_list:
                    fun_iterative_worm(seeds[0], seeds[1])

            # Convert back to image:
            watershed_image = Image.fromarray(np.uint8(mask_array_np))
                        
            # Save the clean image
            watershed_image.save(os.path.join(self.no_watermark_path, 
                                                              temp_image_name))    
    ############################################################


          
    ############################################################        
    def standard_crops(self, 
                       threshold = 0, 
                       path_input = None, 
                       resolution_output = None,
                       correct_center=10):
        '''
        Method to crop the input image (with one single connected component) 
        keeping a percentage (threshold) of the non-null values to provide
        an ending image without noisy/pixellated contours.
        
        Parameters
        ----------
        threshold : int, optional
            Value in [0,100]. Describes the percentile of not 0 image 
                                                  that it is admissible to cut.
                                                  
        path_input : str, optional
            If given it is the absolute path of images to be cropped. 
            Otherwise it takes the no_watermark folder as inputs.
            It expects vertical eco without watermarks.
            
        resolution: int, optional
            The resolution of the squared image, if None: no rescaling is 
            applied.
        
        correct_center: int, optional
            The amount of pixels around the center to accept the new center
        '''
        
        if os.path.isdir(self.cropped_path) == False:
            # Generate the folder:
            os.mkdir(self.cropped_path)

        if path_input != None:
            # Overwrite the starting path
            self.no_watermark_path = str(path_input)   
        
        # Define the list of images to be analyzed:
        input_image_list = os.listdir(self.no_watermark_path)   
        
        try:
            input_image_list = [x for x in input_image_list if x.split('.')[-1]
                                                     in ['jpeg', 'png', 'jpg']]
        except:
            pass
        
        
        
        # As a first step it defines the matrix of 1/0. 
        # 1 for each pixel which is higher than threshold.
        
        # This matrix is used to calculate two vectors
        for temp_image_name in tqdm(input_image_list):
            # Import image
            temp_image = Image.open(os.path.join(self.no_watermark_path, 
                                                 temp_image_name))
            
            # Convert to grey if RGB image
            try:
                temp_image_np = np.asarray(temp_image)[:,:,0]       
            except: 
                temp_image_np = np.asarray(temp_image)[:,:] 
                
            vertical_np = np.zeros(np.shape(temp_image_np)[0])
            for i_row in range(np.shape(temp_image_np)[0]):
                for j_col in range(np.shape(temp_image_np)[1]):
                    if temp_image_np[i_row, j_col]>threshold:
                        vertical_np[i_row] += 1   
                    
            # We use the plot to find the minimum values:
            # First reduce only vertically
            min_top = 0 # 0 is top
            
            found = False
            try:
                while found == False:
                    if vertical_np[min_top+1]>0:
                        found = True
                    else:
                        min_top += 1 
                        if min_top > np.shape(temp_image_np)[0]-1:
                            found = True
                            min_top = 0
                            print("namefile mintop: ", temp_image_name)
            except:
                print("problems namefile mintop: ", temp_image_name)
            
            max_bottom = np.shape(temp_image_np)[0]
            
            found = False
            try:
                while found == False:
                    if vertical_np[max_bottom-1]>0:
                        found = True
                    else:
                        max_bottom -= 1   
                        if max_bottom < 1:
                            found = True
                            max_bottom = np.shape(temp_image_np)[0]
                            print("namefile maxbottom: ", temp_image_name)
            except:
                print("problems namefile mintop: ", temp_image_name)

            temp_image_np = temp_image_np[min_top:max_bottom, :]

            # On the reduced image we have to found the minimum x and y 
            number_of_points_to_check = 5
            
            # Find the left:
            center_left = 0
            
            # See from left to half
            i_temp = 0
            found = False
            while i_temp <int(np.shape(temp_image_np)[1]/2) and found == False:
                center_left = i_temp
                # Check for the next value for several lines:
                for j_temp in range(number_of_points_to_check):
                    if temp_image_np[j_temp, i_temp+1]>0:
                        found = True
                i_temp += 1
          
            # Find the right:
            center_right = np.shape(temp_image_np)[1]
           
            # See from right to half
            i_temp = 0
            found = False
            while i_temp <int(np.shape(temp_image_np)[1]/2) and found == False:
                center_right = np.shape(temp_image_np)[1] -i_temp
                # Check for the next value for several lines:   
                for j_temp in range(number_of_points_to_check):
                    if temp_image_np[j_temp, \
                                        np.shape(temp_image_np)[1]-i_temp-1]>0:
                        found = True
                i_temp += 1
            
            # Check on additional vertical points
            temp_number_of_points_to_check= number_of_points_to_check
            while center_right <= int(np.shape(temp_image_np)[1]/2)+correct_center  or \
               center_left >= int(np.shape(temp_image_np)[1]/2)-correct_center:
                temp_number_of_points_to_check=2*temp_number_of_points_to_check
                
                # Find the left:
                center_left = 0
            
                # See from left to half
                i_temp = 0
                found = False
                while i_temp < int(np.shape(temp_image_np)[1]/2) \
                                                            and found == False:
                    center_left = i_temp
                    # Check for the next value for several lines:
                    for j_temp in range(temp_number_of_points_to_check):
                        if temp_image_np[j_temp, i_temp+1]>0:
                            found = True
                    i_temp += 1

                # Find the right:
                center_right = np.shape(temp_image_np)[1]

                # See from right to half
                i_temp = 0
                found = False
                while i_temp < int(np.shape(temp_image_np)[1]/2) and \
                                                                found == False:
                    #print("i_temp", i_temp)
                    center_right = np.shape(temp_image_np)[1] -i_temp
                    # Check for the next value for several lines:   
                    for j_temp in range(temp_number_of_points_to_check):
                        if temp_image_np[j_temp, 
                                        np.shape(temp_image_np)[1]-i_temp-1]>0:
                            found = True
                    i_temp += 1
                
            # Second step: Calculate horizontal distribution vector:
            horizontal_np = np.zeros(np.shape(temp_image_np)[1])
            for j_col in range(np.shape(temp_image_np)[1]):
                for i_row in range(np.shape(temp_image_np)[0]):
                    if temp_image_np[i_row, j_col]>threshold:
                        horizontal_np[j_col] += 1
                        
            min_left = 0
            
            found = False
            while found == False:
                if horizontal_np[min_left+1]>0:
                    found = True
                else:
                    min_left += 1
            
            max_right =np.shape(temp_image_np)[1]
            
            found = False
            while found == False:
                if horizontal_np[max_right-1]>0:
                    found = True
                else:
                    max_right -= 1

            # We choose if we want to keep them or not:
            # We have to calculate the two distances:
            left_distance = abs(min_left-center_left)
            right_distance = abs(max_right-center_right)
            best_distance = max(left_distance, right_distance)
            min_left = max(min(min_left, int(center_left-best_distance)),0)
            max_right = min(max(max_right, int(center_right+best_distance)), 
                            np.shape(temp_image_np)[1]-1)
            
            # These values are used to downsample the image:
            temp_reduced_image_np = temp_image_np[:, min_left:max_right]
            
           
            # Convert back to image:
            cropped_image = Image.fromarray(np.uint8(temp_reduced_image_np))
              
            # Apply a different resolution:
            if resolution_output != None:
                cropped_image_res = cropped_image.resize((resolution_output, 
                                                          resolution_output))
                cropped_image_res.save(os.path.join(self.cropped_path, 
                                                    temp_image_name))           
                
            # Save the clean image
            try:
                cropped_image_res.save(os.path.join(self.cropped_path, temp_image_name))           
            except:
                print("filename: ", temp_image_name, "dimension cropped: ", 
                      np.shape(temp_reduced_image_np))
    ############################################################     
    
    

    ############################################################     
    def square_crops(self, 
                     delta_pixel_radius = 5, 
                     path_input = None, 
                     number_of_points_to_check = 10, 
                     resolution_input_x = None, 
                     resolution_input_y = None, 
                     reduced_resolution_x = 0.7, 
                     reduced_resolution_y = 0.9, 
                     resolution_output = None):
        
        '''
        Method to convert in polar coordinates the cropped images.
        
        Parameters
        ----------
        
        delta_pixel_radius : int, optional
            The delta x of the radius to check to speed up computation. 
            The optimal is 1 but is slower.
            
        path_input : str, optional
            Alternative global input path.
            
        resolution_x : int, optional
            The number of pixel of the ending image (on x)
            
        resolution_y : int, optional
            The number of pixel of the ending image (on y)
            
        reduced_resolution_x/y : int, optional
            Amount of resolution lost (to improve quality of transform)
        '''

        # Generate the folder:
        if os.path.isdir(self.squared_path) == False:            
            os.mkdir(self.squared_path)

        # Overwrite the starting path
        if path_input != None:
            self.cropped_path = path_input
                
        # If resolution is given, resize the image:
        if resolution_input_x != None:
            resolution_x = resolution_input_x
        
        if resolution_input_y != None:
            resolution_y = resolution_input_y

        # Define the list of images to be analyzed:
        input_image_list = os.listdir(self.cropped_path)    
        
        try:
            input_image_list = [x for x in input_image_list if x.split('.')[-1]  
                                                     in ['jpeg', 'png', 'jpg']]
        except:
            pass
        
        for temp_image_name in tqdm(input_image_list):
            # Import image
            temp_image = Image.open(os.path.join(self.cropped_path, temp_image_name))
            
            # Convert to grey
            try:
                temp_image_np = np.asarray(temp_image)[:,:,0]       
            except: 
                temp_image_np = np.asarray(temp_image)[:,:]
        
          
            try:
                # If the resolution is not set, the method defines it as:
                if resolution_input_x == None:
                    resolution_x = \
                        int(np.shape(temp_image_np)[1]*reduced_resolution_x)

                if resolution_input_y == None:
                    resolution_y = \
                        int(np.shape(temp_image_np)[0]*reduced_resolution_y)   


                # We assume that the two starting point has y value (row) 
                # equal to 0 (it is oriented to the base).
                # We find the value of columns:
                # There is the risk that the first row is uncorrect, 
                # so we check on a few pixels number_of_points_to_check

                # Find the left:
                center_left = 0
                y_center_left = 0

                # See from left to half (to define x1_l and y1_l)
                i_temp = 0
                found = False
                while i_temp < int(np.shape(temp_image_np)[1]/2) and\
                                                                found == False:
                    center_left = i_temp
                    # Check for the next value for several lines:
                    for j_temp in range(number_of_points_to_check):
                        if temp_image_np[j_temp, i_temp+1]>0:
                            if found != True:
                                y_center_left = j_temp
                            found = True
                    i_temp += 1

                # Find the right:
                center_right = np.shape(temp_image_np)[1]
                y_center_right = 0
                
                # Look from right to half
                i_temp = 0
                found = False
                while i_temp < int(np.shape(temp_image_np)[1]/2) and\
                                                                found == False:
                    
                    center_right = np.shape(temp_image_np)[1] -i_temp
                    
                    # Check for the next value for several lines:
                    for j_temp in range(number_of_points_to_check):
                        if temp_image_np[j_temp, 
                                        np.shape(temp_image_np)[1]-i_temp-1]>0:
                            if found != True:
                                y_center_right = j_temp
                            found = True
                    i_temp += 1

                # See from top to bottom Right vertical point border
                i_temp = 0
                found = False
                while i_temp < int(np.shape(temp_image_np)[0])-1\
                                 -number_of_points_to_check and found == False:
                    top_vert_r = i_temp
                    # Check for the next value for several lines:
                    for j_temp in range(number_of_points_to_check):
                        if temp_image_np[i_temp+1,
                                  int(np.shape(temp_image_np)[1])-1- j_temp]>0:
                            found = True
                    i_temp += 1

                # Given the two center point (o, center_left) and 
                # (0, center_right) we have to choose the optimal candidate for 
                # the radius. The "radius" is a value from 0 to maximum the 
                # length of the image. We use bisection to speed up:
                min_radius_temp = delta_pixel_radius
                max_radius_temp = np.shape(temp_image_np)[1]

                n_iter = 0
                should_I_stop = False
                # Iterate bisection until maximum number or found:
                while n_iter < 10 and should_I_stop == False:
                    candidate_radius = abs(max_radius_temp+min_radius_temp)/2

                    n_iter += 1 
                    how_many_found = 0

                    x1 = center_left
                    y1 = 0
                    x2 = center_left - candidate_radius
                    y2 = int(np.shape(temp_image_np)[0])

                    x1_r = center_right
                    y1_r = 0
                    x2_r = center_right + candidate_radius
                    y2_r = int(np.shape(temp_image_np)[0])

                    # Check how many points are inside the cone
                    for y_row in range(int(np.shape(temp_image_np)[0])):
                        for x_col in range(int(np.shape(temp_image_np)[1])):
                            if temp_image_np[y_row, x_col]>0:

                                if (y_row < (x_col-x1)*(y2-y1)/(x2-x1)+y1) or\
                           (y_row < (x_col-x1_r)*(y2_r-y1_r)/(x2_r-x1_r)+y1_r):
                                    how_many_found += 1

                    # If how_many_found = 0 -> TOO wide:
                    # If we did not find any -> The radius is too big
                    if how_many_found == 0: 
                        max_radius_temp = candidate_radius
                    # Found something, we have to increase the range:
                    elif how_many_found > 0:    
                        min_radius_temp = candidate_radius

                    if abs(max_radius_temp-min_radius_temp)<2:
                        should_I_stop = True
                        
                # Define the: center, inner radius, outher radius and angle.

                # Two bottom points:
                x2_r = int(np.shape(temp_image_np)[1])
                y2_r = top_vert_r#int(np.shape(temp_image_np)[0])

                # Median point
                x_m = int((center_left+center_right)/2)
                y_m = int(np.shape(temp_image_np)[0])
                
                # Two top points:
                x1_r = max(center_right, 2*x_m -center_left) 
                y1_r = y_center_right
                
                x1_l = min(center_left, 2*x_m -center_right) 
                y1_l = y_center_left

                # We calculate the center of the cone, the x value is easy, 
                # the y is the intersection:
                x_c = x_m
                y_c = (x_m-x1_r)*(y2_r-y1_r)/(x2_r-x1_r) +y1_r

                # Here we have to modifiy the image to add black pixels 
                # on the bottom:
                # Candidates radia are the distances among the center and the 
                # various pixels:
 
                # We calculate the maximum radius:
                R_max= np.sqrt((x_c-x_m)**2 +(y_c-y_m)**2)

                # We calculate the minimum radius:
                R_min= np.sqrt((x_c-x1_r)**2 +(y_c-y1_r)**2)  # QUI ASSUME CHE y1 R sia su 0, pericoloso!  

                # We calculate the radius (Radiant):
                hypothenuse = np.sqrt((x_c-x1_r)**2 +(y_c-y1_r)**2)
                alpha_top_right = abs(np.arccos(abs(y1_r-y_c)/hypothenuse)) 

                hypothenuse = np.sqrt((x_c-x1_l)**2 +(y_c-y1_l)**2)
                alpha_top_left = abs(np.arccos(abs(y1_l-y_c)/hypothenuse))
                alpha = min(alpha_top_right, alpha_top_left)

                # We can now define an empty matrix of given resolution:
                new_squared_image = np.zeros((resolution_y, resolution_x))

                # For each pixel of this square image we find the original 
                # value (the closest) in the polar space:
                for y_row in range(resolution_y):
                    for x_col in range(resolution_x):
                        corresponding_radius = \
                                       (R_max-R_min)*(y_row/resolution_y)+R_min
                        corresponding_angle = \
                                           (x_col/resolution_x)*2*alpha - alpha
                        
                        if (int(corresponding_radius*\
                                np.cos(corresponding_angle)+y_c)>0) and\
                                    (int(corresponding_radius\
                                     *np.cos(corresponding_angle)+y_c)<\
                                     np.shape(temp_image_np)[0]) and\
                                    (int(x_m+corresponding_radius\
                                    *np.sin(corresponding_angle)) > 0) and \
                                    (int(x_m+corresponding_radius\
                                    *np.sin(corresponding_angle)) < \
                                    np.shape(temp_image_np)[1]):
                            y_row_polar = min(max(int(corresponding_radius\
                                        *np.cos(corresponding_angle)+y_c),0),  
                                        np.shape(temp_image_np)[0]-1)
                            x_col_polar = min(max(int(x_m+corresponding_radius\
                                        *np.sin(corresponding_angle)),0), \
                                        np.shape(temp_image_np)[1]-1)
                        else:
                            y_row_polar = 0
                            x_col_polar = 0
                        new_squared_image[y_row, x_col] = \
                                        temp_image_np[y_row_polar, x_col_polar]

                # Convert back to image:
                polar_image = Image.fromarray(np.uint8(new_squared_image))

                # Set up squared resolution:
                if resolution_output != None:
                    polar_image = polar_image.resize\
                                       ((resolution_output, resolution_output))

                # Save the clean image
                polar_image.save(os.path.join(self.squared_path, \
                                                              temp_image_name))           
            except:
                print("problematic images: ", temp_image_name)
    ############################################################     