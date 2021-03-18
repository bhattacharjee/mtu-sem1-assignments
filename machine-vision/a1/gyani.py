import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import argmax, shape

KERNEL_SIZE = 3
KEY_POINT_THRESHOLD = 10

"""
 - Load input image
 - covert to gray image
 - convert to float32
"""
def load_image(path):
    raw_image = cv2.imread(path)
    gray_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
    gray_image_float = np.float32(gray_image)
    return raw_image,gray_image_float
"""
 - Create Gaussian kernel based on sigma value. The window size is multiple of sigma
"""
def create_gaussian_kernel(sigma):
    point_x, point_y = np.meshgrid(np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma),
                       np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma))
    return np.exp(-(point_x**2 + point_y**2)/(2*sigma**2))/(2*np.pi*sigma**2)    

"""
 - Get 12 Gaussian kernel as per sigma value
"""
def get_gaussian_kernel():
    sigma_list = []
    gaussian_kernel_list = []

    for index in range(0,12):
        sigma = 2**(index/2)        
        kernel = create_gaussian_kernel(sigma)
        sigma_list.append(sigma)
        gaussian_kernel_list.append(kernel)

    return sigma_list,gaussian_kernel_list

def plot_kernels(kernel_list,sigma_list):
    fig = plt.figure(figsize=(10,10))     
    fig_count = len(kernel_list)
    for index in range(0,fig_count):
        #3 figure per row    
        plt.subplot(math.ceil(fig_count/3),3,index+1)
        plt.imshow(kernel_list[index])
        plt.title("Gaussian Kernel sigma = %s"%round(sigma_list[index],2)) 

def generate_scale_space_image(input_image,kernel_list):
    scale_space_image_list = []
    for index in range(0,len(kernel_list)):
        scale_space_image = cv2.filter2D(input_image,-1,kernel_list[index])
        scale_space_image_list.append(scale_space_image)
    return scale_space_image_list

def plot_image_list(image_list,sigma_list):
    fig = plt.figure(figsize=(10,10))     
    fig_count = len(image_list)
    for index in range(0,fig_count):
        #3 figure per row    
        plt.subplot(math.ceil(fig_count/3),3,index+1)
        plt.imshow(image_list[index],cmap="gray")
        plt.title("sigma = %s"%round(sigma_list[index],2)) 

def generate_display_scale_space_image(input_image,kernel_list,sigma_list):
    scale_space_image_list = generate_scale_space_image(input_image,kernel_list)
    plot_image_list(scale_space_image_list,sigma_list)
    return scale_space_image_list

# Copied from gyani
def get_gaussian_smoothing_kernel_internal(size:int, sigma:float):
    point_x, point_y = np.meshgrid(np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma),
                       np.arange(-KERNEL_SIZE*sigma, KERNEL_SIZE*sigma))
    return np.exp(-(point_x**2 + point_y**2)/(2*sigma**2))/(2*np.pi*sigma**2)    

def get_gaussian_smoothing_kernel(sigma:float)->np.ndarray:
    return get_gaussian_smoothing_kernel_internal(math.ceil(6 * sigma), sigma)

#def get_smoothed_images(image, sigma_values, kernel_size=100)->list:
def generate_display_scale_space_image(image,kernel_list,sigma_values):
    print("Rajbir's override")
    smoothed_images = []
    for sigma in sigma_values:
        g = get_gaussian_smoothing_kernel(sigma)
        im = cv2.filter2D(image.copy(), -1, g)
        smoothed_images.append(im)
    plot_image_list(smoothed_images, sigma_values)
    return smoothed_images
        

def get_difference_of_gaussian(scale_space_image_list):
    dog_image_list = []
    for index in range(1,len(scale_space_image_list)):
        dog_image_list.append(scale_space_image_list[index]-scale_space_image_list[index-1])
    return dog_image_list

def plot_dog_image_list(dog_image_list,sigma_list):
    plot_image_list(dog_image_list,sigma_list)

def get_dog_and_plot(scale_space_image_list,sigma_list):
    dog_image_list = get_difference_of_gaussian(scale_space_image_list)
    plot_dog_image_list(dog_image_list,sigma_list)
    return dog_image_list

def get_key_points(dog_image_list,sigma_list):
    key_points = []

    for index in range(1,len(dog_image_list)-1):
        
        level_dog = dog_image_list[index]
        lower_dog = dog_image_list[index-1]
        upper_dog = dog_image_list[index+1]
        level_sigma = sigma_list[index]        

        for x in range(1,len(level_dog)-1):
            for y in range(1,len(level_dog[0])-1):
                pix_val = level_dog[x][y]
                if pix_val>KEY_POINT_THRESHOLD and pix_val>level_dog[x-1][y-1] and pix_val>level_dog[x-1][y] and pix_val>level_dog[x-1][y+1] and pix_val>level_dog[x][y-1] and \
                                pix_val>level_dog[x][y+1] and pix_val>level_dog[x+1][y+1] and pix_val>level_dog[x+1][y] and pix_val>level_dog[x+1][y-1] and \
                                pix_val>lower_dog[x-1][y-1] and pix_val>lower_dog[x-1][y] and pix_val>lower_dog[x-1][y+1] and pix_val>lower_dog[x][y-1] and \
                                pix_val>lower_dog[x][y+1] and pix_val>lower_dog[x+1][y+1] and pix_val>lower_dog[x+1][y] and pix_val>lower_dog[x+1][y-1] and pix_val>lower_dog[x][y] and \
                                pix_val>upper_dog[x-1][y-1] and pix_val>upper_dog[x-1][y] and pix_val>upper_dog[x-1][y+1] and pix_val>upper_dog[x][y-1] and \
                                pix_val>upper_dog[x][y+1] and pix_val>upper_dog[x+1][y+1] and pix_val>upper_dog[x+1][y] and pix_val>upper_dog[x+1][y-1] and pix_val>upper_dog[x][y]:
                                      
                    key_points.append((x,y,level_sigma))

    print("Total %d key points"%len(key_points))    
    return key_points


def visulaize_key_point(input_image,key_points):
    for point in key_points:        
        marked_image = cv2.circle(input_image, (point[1],point[0]), round(3*point[2]), (0,255,0), 1)    
        cv2.imshow("KEY POINT CIRCLED IMAGE", marked_image)  

def visualize_key_point_orientation(input_image,key_point_theeta):
    
    circle_image = input_image
    for point in key_point_theeta:
        x1,y1,sigma,theeta = point[0],point[1],point[2],point[3]
        x2 = round(x1+ round(3*sigma)*math.cos(theeta*180/np.pi))
        y2 = round(y1+ round(3*sigma)*math.sin(theeta*180/np.pi)) 
        marked_image = cv2.circle(input_image, (y1,x1), round(3*sigma), (0,255,0), thickness = 2)   
        #circle_image = cv2.circle(circle_image,(x1,y1), round(3*sigma),color = (0,0,255),thickness=2)
        circle_image = cv2.line(circle_image,(y1,x1),(y2,x2),(0,0,255), thickness=1,lineType=4)        
        
    cv2.imshow("result_1", circle_image)     
    cv2.waitKey(0)


def get_derivative_kernel():
    d_x = np.array([[1, 0, -1]])
    return d_x,d_x.T

def calculate_scale_space_image_drivative(scale_space_image_list):

    derivative_image_list = []
    dx,dy = get_derivative_kernel()    
    for image in scale_space_image_list:
        derivative_image_x = cv2.filter2D(image,-1,dx)
        derivative_image_y = cv2.filter2D(image,-1,dy)
        derivative_image_list.append((derivative_image_x,derivative_image_y))

    return derivative_image_list

def plot_scale_space_derivative_image(scale_space_derivative_image_list,sigma_list):
    plot_image_list([image[0] for image in scale_space_derivative_image_list],sigma_list)
    plot_image_list([image[1] for image in scale_space_derivative_image_list],sigma_list)

def get_mag_direction(key_points,sigma_list,derivative_image_list):
    key_points_grid = []    
    for points in key_points:
        sigma = points[2]
        x_pos = points[0]
        y_pos = points[1]
        index = sigma_list.index(sigma)
        derivative_x,derivative_y= derivative_image_list[index]
        try:
            grid_magnitutede = np.zeros((7,7),dtype=float)
            grid_direction = np.zeros((7,7),dtype=float)
            grid_Weight = np.zeros((7,7),dtype=float)
            for x in range(-3,4):
                for y in range(-3,4):
                    q = int((3/2)*x*sigma)
                    r = int((3/2)*y*sigma)
                    grid_x = q+x_pos
                    grid_y = r+y_pos
                    x_pixel = derivative_x[grid_x][grid_y]
                    y_pixel = derivative_y[grid_x][grid_y]

                    grid_magnitutede[x+3][y+3]= math.sqrt(x_pixel**2+y_pixel**2)
                    grid_direction[x+3][y+3]= (np.pi + math.atan2(y_pixel,x_pixel))*180/np.pi
                    grid_Weight[x+3][y+3] = np.exp(-(q**2+r**2)/(9*sigma**2/2))/(9*np.pi*sigma**2/2)

            grid_bin = np.zeros((36,1),dtype=float)            
            for x in range(0,7):
                for y in range(0,7):
                    angle = grid_direction[x][y]
                    bin_index = int(angle/10)
                    grid_bin[bin_index] = grid_bin[bin_index] + grid_magnitutede[x][y]*grid_Weight[x][y]
            theeta = (2*np.pi/36)*(grid_bin[np.argmax(grid_bin)]+1/2)
            #key_points_grid.append((x_pos,y_pos,sigma,grid_magnitutede,grid_direction,grid_Weight,grid_bin))            
            key_points_grid.append((x_pos,y_pos,sigma,theeta))
        except:
            pass
    print("Total grid points:%d"%len(key_points_grid))    
    return  key_points_grid   

def get_feature_descriptor(key_points_theeta,sigma_list,derivative_image_list):

    key_point_grid_theeta = []
    for points in key_points_theeta:        
        
        x_pos = points[0]
        y_pos = points[1]
        sigma = points[2]
        theta = points[3]
        index = sigma_list.index(sigma)
        derivative_x,derivative_y= derivative_image_list[index]
        
        grid_magnitude = np.zeros((256),dtype=float)
        grid_magnitude = np.reshape(grid_magnitude,(4,4,4,4))
        grid_direction = np.zeros((256),dtype=float)
        grid_direction = np.reshape(grid_direction,(4,4,4,4))
        grid_Weight = np.zeros((256),dtype=float)
        grid_Weight = np.reshape(grid_Weight,(4,4,4,4))
        
        try:
            for i in range(-2,2):
                for j in range(-2,2):
                    k_i = [4*i,4*i+1,4*i+2,4*i+3]
                    k_j = [4*j,4*j+1,4*j+2,4*j+3]   
                    for ii in range(0,4):
                        for  jj in range(0,4):
                            s=9/16*(4*i+ii+1/2)*sigma
                            t=9/16*(4*j+jj+1/2)*sigma

                            grid_x = int(s)+x_pos
                            grid_y = int(t)+y_pos
                            x_pixel = derivative_x[grid_x][grid_y]
                            y_pixel = derivative_y[grid_x][grid_y]
                            grid_magnitude[i+2][j+2][ii][jj]= math.sqrt(x_pixel**2+y_pixel**2)
                            grid_direction[i+2][j+2][ii][jj]= math.atan2(y_pixel,x_pixel) - theta
                            grid_Weight[i+2][j+2][ii][jj] = np.exp(-(s**2+t**2)/(81*sigma**2/2))/(81*np.pi*sigma**2)/2
            
            grid_bin = np.zeros(128,dtype=float)  
            grid_bin = np.reshape(grid_bin,(4,4,8))
            for i in range(0,4):
                for j in range(0,4):
                    grid_m = grid_magnitude[i][j]
                    grid_d = grid_direction[i][j]
                    grid_w = grid_Weight[i][j]
                    for x in range(0,4):
                        for y in range(0,4):
                            angle = (grid_d[x][y]+np.pi)*180/np.pi
                            index = int(angle/8)
                            grid_bin[x][y] =  grid_bin[x][y] + grid_m[x][y]*grid_w[x][y]
            

            key_point_grid_theeta.append((points,grid_bin.flatten()))
        except:
            pass
    print("Total feature points %d"%len(key_point_grid_theeta))
    return  key_point_grid_theeta   



def plot_gausian():
    sigma_list = []
    for k in range(0,12):
        sigma_list.append(2**(k/2))
    kernel_list = []
    for sigma in sigma_list:

        x, y = np.meshgrid(np.arange(-3*sigma, 3*sigma),
                       np.arange(-3*sigma, 3*sigma))
        
        kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*np.pi*sigma**2)    
        kernel_list.append(kernel)
    fig = plt.figure(figsize=(10,10))     
    
    for index in range(0,12):    
        plt.subplot(4,3,index+1)
        plt.imshow(kernel_list[index])
        plt.title("Gaussian Kernel sigma = %s"%round(sigma_list[index],3))   
    
    input_image = cv2.imread('C:\\Gyani\\Vision\\Assignment-1\\Assignment_MV_1_image.png')    
    gray_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    image_list = []
    for index in range(0,12):
        image = cv2.filter2D(gray_image,-1,kernel_list[index])
        image_list.append(image)
    
    fig1 = plt.figure(figsize=(10,10))     
    for index in range(0,12):    
        plt.subplot(4,3,index+1)
        plt.imshow(image_list[index],cmap="gray")
        plt.title("Sigma =  %s"%round(sigma_list[index],3))   

    diff_of_gaussian = []

    for index in range(1,12):
        dog = image_list[index] - image_list[index-1]
        diff_of_gaussian.append((dog,sigma_list[index-1]))
    
    fig1 = plt.figure(figsize=(10,10))     
    for index in range(0,11):    
        plt.subplot(4,3,index+1)
        plt.imshow(diff_of_gaussian[index][0],cmap="gray")
        plt.title("Sigma =  %s"%round(diff_of_gaussian[index][1],4))   

    key_points = []

    for z in range(1,10):
        (level_dog,le_sigma)=diff_of_gaussian[z]
        (lower_dog,lo_sigma)=diff_of_gaussian[z-1]
        (upper_dog,up_sigma)=diff_of_gaussian[z+1]
        for x in range(1,len(level_dog)-1):
            for y in range(1,len(level_dog[0])-1):
                pix_val = level_dog[x][y]
                if(pix_val>10):
                    Level_condition = pix_val>level_dog[x-1][y-1] and pix_val>level_dog[x-1][y] and pix_val>level_dog[x-1][y+1] and pix_val>level_dog[x][y-1] and \
                                pix_val>level_dog[x][y+1] and pix_val>level_dog[x+1][y+1] and pix_val>level_dog[x+1][y] and pix_val>level_dog[x+1][y-1] and \
                                pix_val>lower_dog[x-1][y-1] and pix_val>lower_dog[x-1][y] and pix_val>lower_dog[x-1][y+1] and pix_val>lower_dog[x][y-1] and \
                                pix_val>lower_dog[x][y+1] and pix_val>lower_dog[x+1][y+1] and pix_val>lower_dog[x+1][y] and pix_val>lower_dog[x+1][y-1] and pix_val>lower_dog[x][y] and \
                                pix_val>upper_dog[x-1][y-1] and pix_val>upper_dog[x-1][y] and pix_val>upper_dog[x-1][y+1] and pix_val>upper_dog[x][y-1] and \
                                pix_val>upper_dog[x][y+1] and pix_val>upper_dog[x+1][y+1] and pix_val>upper_dog[x+1][y] and pix_val>upper_dog[x+1][y-1] and pix_val>upper_dog[x][y]
                  
                    if Level_condition == True:
                        key_points.append((x,y,le_sigma))

        for point in key_points:        
            marked_image = cv2.circle(input_image, (point[1],point[0]), round(3*point[2]), (0,255,0), 2)    
            cv2.imshow("result", marked_image)                        
    
    print("Total %d crossing point found"%len(key_points))
    print(key_points)
    d_x = np.array([[1, 0, -1]])
    d_y = np.array([[1, 0, -1]])

    dvx_image_list=[]
    for index in range(0,11):
        d_x_image = cv2.filter2D(diff_of_gaussian[index][0],-1,d_x)
        d_y_image = cv2.filter2D(diff_of_gaussian[index][0],-1,d_y.T)
        dvx_image_list.append((d_x_image,d_y_image,sigma_list[index]))

    #print image

    fig2 = plt.figure(figsize=(10,10))     
    for index in range(0,11):    
        plt.subplot(4,3,index+1)
        plt.imshow(dvx_image_list[index][0],cmap="gray")
        plt.title("Sigma =  %s"%round(dvx_image_list[index][2],4))   

    fig3 = plt.figure(figsize=(10,10))     
    for index in range(0,11):    
        plt.subplot(4,3,index+1)
        plt.imshow(dvx_image_list[index][1],cmap="gray")
        plt.title("Sigma =  %s"%round(dvx_image_list[index][2],4))   

    mqr_list=[]
    for point in key_points:
        x,y,sigma = point[0],point[1],point[2]
        mqr_Mag_entry = np.zeros((7,7),dtype=float)
        mqr_dir_entry = np.zeros((7,7),dtype=float)
        mqr_wt_entry = np.zeros((7,7),dtype=float)
        for q in range(-3,4):
            for r in range(-3,4):
                q1 = np.int(q*sigma*(3/2))
                r1 = np.int(r*sigma*(3/2))
                d_index = sigma_list.index(sigma)
                x11 = dvx_image_list[d_index][0]
                y11 = dvx_image_list[d_index][1] 
                if( x+q1<1080 and y+r1 <1920 ):
                    x_component = x11[x+q1][y+r1]
                    y_component = y11[x+q1][y+r1]
                    magnitutude = np.sqrt(x_component**2+y_component**2)
                    direction = math.atan2(y_component,x_component)
                    mqr_Mag_entry[q+3][r+3] = magnitutude
                    mqr_dir_entry[q+3][r+3] = direction    
                    
                    mqr_wt_entry[q+3][r+3] = np.exp(-(q1**2+r1**2)/(9*sigma**2/2))/(9*np.pi*sigma**2)/2

        mqr_list.append((point,mqr_Mag_entry,mqr_dir_entry,mqr_wt_entry))

    final_point_list = []
    for entry in mqr_list:
        bin = np.zeros((36,),dtype=float)
        point = entry[0]
        mag = entry[1]
        dir = entry[2]
        wt = entry[3]
        for x in range(0,7):
            for y in range(0,7):
                theeta = dir[x][y]*36/2*np.pi
                coeff = int(theeta/10)
                rem = theeta%10
                bin_index = coeff -1
                if rem>0 or bin_index == -1:
                    bin_index = bin_index+1
                bin[bin_index]= bin[bin_index] + mag[x][y]*wt[x][y]
        h1 = bin[np.argmax(bin)]       
        theeta1 = 2*np.pi*(1/2 + h1)*1/36
        final_point_list.append((point[0],point[1],point[2],theeta1))

    for point in final_point_list:       
        x1 = point[0]
        y1 = point[1] 
        sigma = point[2]
        angle = point[3]
        marked_image = cv2.circle(input_image, (x1,y1), round(3*sigma), (0,255,0), 2)    
        x2 = round(x1+ round(3*sigma)*math.cos(angle*np.pi*1/180 - np.pi))
        y2 = round(y1+ round(3*sigma)*math.sin(angle*np.pi*1/180 - np.pi))
        cv2.line(marked_image,(x1,y1),(x2,y2),(255,0,0), 2)
        
        cv2.imshow("result_1", marked_image)     

    plt.tight_layout()
    plt.show()


def main():
    
    assignment_image,float_image = load_image("./Assignment_MV_1_image.png")
    sigm_list,gaussian_kernel_list = get_gaussian_kernel()
    plot_kernels(gaussian_kernel_list,sigm_list)
    scale_space_image_list = generate_display_scale_space_image(float_image,gaussian_kernel_list,sigm_list)
    dog_image_list = get_dog_and_plot(scale_space_image_list,sigm_list)
    key_points = get_key_points(dog_image_list,sigm_list)
    
    visulaize_key_point(assignment_image,key_points)
    derivative_image_list = calculate_scale_space_image_drivative(scale_space_image_list)
    plot_scale_space_derivative_image(derivative_image_list,sigm_list)
    key_points_theeta = get_mag_direction(key_points,sigm_list,derivative_image_list)
    visualize_key_point_orientation(assignment_image,key_points_theeta)
    plt.show()

    key_point_feature_grid = get_feature_descriptor(key_points_theeta,sigm_list,derivative_image_list)

main()
