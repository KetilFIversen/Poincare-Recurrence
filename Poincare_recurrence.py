import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import imageio
from tqdm import trange

def crop_center(image, w, h):
    y, x, _ = image.shape
    startx = x//2 - w//2
    starty = y//2 - h//2    
    return image[starty:starty+h, startx:startx+w, :]

class poincare_recurrence:
    
    def __init__(self, image, dir_name, name, transform_list=[0], figsize=(9,9), dpi=25, filetype='.jpg'):
        
        self.figsize  = figsize
        self.dpi      = dpi
        self.dir_name = dir_name
        self.name     = name
        self.filetype = filetype
        self.transform_list = transform_list
        
        w = image.shape[0]
        h = image.shape[1]

        self.x = np.linspace(1,w,w)
        self.y = np.linspace(1,h,h)

        [X,Y] = np.meshgrid(self.x,self.y)
        [self.XX, self.YY] = self.transformation(X,Y,w,h)
        
        self.c = ['r','g','b']
        self.set_image(image)
        
    def set_dataset(self): 
        self.ds = xr.Dataset(data_vars   = {'F'  : (['y','x','c'], self.F)},
                             coords      = {'x'  : (['x'],     self.x),
                                            'y'  : (['y'],     self.y),
                                            'c'  : (['c'],     self.c),
                                            'XX' : (['y','x'], self.XX),
                                            'YY' : (['y','x'], self.YY)})
    def set_image(self, image):
        F_r = image[:,:,0]
        F_g = image[:,:,1]
        F_b = image[:,:,2]
        self.F     = np.stack([F_r,F_g,F_b], axis=-1)
        self.image = image
        self.set_dataset()
        
        
    def run(self, N=1000):
        F = self.ds['F'].data 
        original_image = F
        print('Generating images...')
        for self.n in trange(N):
            
            self.plot(F)
            F = self.ds['F'].data[self.ds['YY'].data, self.ds['XX'].data,:]
            self.ds['F'] = (['y','x','c'], F)

            if ((original_image.flatten() - F.flatten()) == 0).all():
                self.plot(F)
                break
    
    
    def make_movie(self, duration=0.025, dynamic=True):
        images = []
        print('Generating movie...')
        for filename in sorted(glob.glob(self.dir_name + '/' + self.name + '*' + self.filetype), 
                               key=lambda f: int(''.join(filter(str.isdigit, f)))):
                images.append(imageio.imread(filename))
        n_images = len(images)
        duration = [(1.0-duration)*(np.exp(-ii/2)+np.exp((ii-n_images)/2)) + duration for ii in range(n_images)] if dynamic else duration
        imageio.mimsave(self.dir_name + '/' + self.name + 'movie.gif', images, duration=duration)
        print('Done!')
    
    
    def plot(self, F):
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(F)
        plt.axis('off')
        plt.savefig(self.dir_name+self.name+str(self.n)+'.jpg', bbox_inches='tight')
        plt.close('all')
    
    
    def newcoords_1(self, X, Y, w, h):
    
        XX = -X-(w/h)*Y+3*w/2
        YY = -(h/w)*X+h

        for j in range(h):
            for k in range(w):
                if XX[j,k] >= w:
                    XX[j,k] = XX[j,k] - w
                    YY[j,k] = YY[j,k]
                if XX[j,k] < 0:
                    XX[j,k] = XX[j,k] + w
                    YY[j,k] = YY[j,k]

        return XX.astype(int), YY.astype(int)
    
    
    def newcoords_2(self, X, Y, w, h):

        XX = -X -(w/h)*Y + 4*w/2
        YY = -(h/w)*X + h

        for j in range(h):
            for k in range(w):
                if XX[j,k] >= w:
                    XX[j,k] = XX[j,k] - w
                    YY[j,k] = YY[j,k]
              
                if XX[j,k] < 0:
                    XX[j,k] = XX[j,k] + w
                    YY[j,k] = YY[j,k]
                      
        return XX.astype(int), YY.astype(int)
    
    
    def newcoords_3(self, X, Y, w, h):

        XX = X+1
        YY = Y+1

        for j in range(h):
            for k in range(w):
                XX[j,k] = X[2*(j+1) - (h+1)*np.floor((2*(j+1)-1)/h).astype(int)-1, k] - 1
                YY[j,k] = Y[2*(j+1) - (h+1)*np.floor((2*(j+1)-1)/h).astype(int)-1, k] - 1

        return XX.astype(int), YY.astype(int)
    
    
    def newcoords_4(self, X, Y, w, h):
        
        XX = X+1 #Remove +1 for cool fractal
        YY = Y+1 #Remove +1 for cool fractal

        for j in range(h):
            for k in range(w):
                XX[j,k] = X[2*(j+1) - (h+1)*np.floor((2*(j+1)-1)/h).astype(int)-1, 2*(k+1) - (w+1)*np.floor((2*(k+1)-1)/w).astype(int)-1] - 1
                YY[j,k] = Y[2*(j+1) - (h+1)*np.floor((2*(j+1)-1)/h).astype(int)-1, 2*(k+1) - (w+1)*np.floor((2*(k+1)-1)/w).astype(int)-1] - 1  
        
        return XX.astype(int), YY.astype(int)
    
    
    def newcoords_5(self, X, Y, w, h):
        XX = X
        YY = Y
        n = 0;
        for j in range(h):
            for k in range(w):
                XX[j,k] = X[j, np.mod((k+1)+n-1,w)] - 1
                YY[j,k] = Y[j, np.mod((k+1)+n-1,w)] - 1   
            n = n + 1
        
        return XX.astype(int), YY.astype(int)

    # BROKEN
    def newcoords_6(self, X, Y, w, h):
        
        XX = X
        YY = Y

        for j in range(h):
            for k in range(w):
                XX[j,k] = X[j,-k] 
                YY[j,k] = Y[-j,k]

        return XX.astype(int), YY.astype(int)

    # BROKEN
    def newcoords_7(self, X, Y, w, h):

        XX = X
        YY = Y

        for j in range(h):
            for k in range(w):
                XX[j,k] = Y[j,k] - 1
                YY[j,k] = X[j,k] - 1  
        
        return XX.astype(int), YY.astype(int)

    # BROKEN
    def newcoords_8(self, X, Y, w, h):

        XX = X + w/2
        YY = Y + h/2

        for j in range(h):
            for k in range(w):

                if XX[j,k] > w and YY[j,k] <= h:
                    XX[j,k] = XX[j,k] - w
                    YY[j,k] = YY[j,k]

                if YY[j,k] > h and XX[j,k] <= w:
                    XX[j,k] = XX[j,k]
                    YY[j,k] = YY[j,k] - h

                if YY[j,k] > h and XX[j,k] > w:
                    XX[j,k] = XX[j,k] - w
                    YY[j,k] = YY[j,k] - h

        XX = XX - 1
        YY = YY - 1

        return XX.astype(int), YY.astype(int)


    def newcoords_9(self, X, Y, w, h):

        XX = X + w/2
        YY = Y + h/2

        for j in range(h):
            for k in range(w):

                if XX[j,k] > w and YY[j,k] <= h:
                    XX[j,k] = 3*w/2 - XX[j,k] + 1
                    YY[j,k] = YY[j,k]
                    
                if YY[j,k] > h and XX[j,k] <= w:
                    XX[j,k] = XX[j,k]
                    YY[j,k] = 3*h/2 - YY[j,k] + 1

                if YY[j,k] > h and XX[j,k] > w:
                    XX[j,k] = 3*w/2 - XX[j,k] + 1
                    YY[j,k] = 3*h/2 - YY[j,k] + 1

        XX = XX - 1
        YY = YY - 1
        
        return XX.astype(int), YY.astype(int)


    def newcoords_10(self, X, Y, w, h):

        XX = X + w/2
        YY = Y + h/2

        for j in range(h):
            for k in range(w):

                if XX[j,k] > w and YY[j,k] <= h:
                    XX[j,k] = 3*w/2 - XX[j,k] + 1
                    YY[j,k] = YY[j,k]

                if YY[j,k] > h and XX[j,k] <= w:
                    XX[j,k] = XX[j,k]
                    YY[j,k] = 3*h/2 - YY[j,k] + 1

                if YY[j,k] > h and XX[j,k] > w:
                    XX[j,k] = 3*w/2 - XX[j,k] + 1
                    YY[j,k] = 3*h/2 - YY[j,k] + 1

        XX = XX - 1
        YY = YY - 1

        return XX.astype(int), YY.astype(int)
    
                      
    def transformation(self, X, Y, w, h):
        XX = X
        YY = Y
        newcoords_list = [self.newcoords_1, self.newcoords_2, self.newcoords_3, self.newcoords_4,
                          self.newcoords_5, self.newcoords_6, self.newcoords_7, self.newcoords_8,
                          self.newcoords_9, self.newcoords_10]
        
        for ii, transform in enumerate(self.transform_list):
            if ii == 0:
                XX, YY = newcoords_list[transform-1](XX, YY, w, h)
            else:
                XX, YY = newcoords_list[transform-1](XX+1, YY+1, w, h)
        return XX.astype(int), YY.astype(int)



if __name__ == '__main__':

    parent_dir = os.getcwd()
    in_dir     = '\\Indata_images\\'
    out_dir    = '\\Images\\'
    image_file = 'Poinman.jpg'
    N = 1000
    transform_list = [1]
    duration = 0.1
    dynamic  = True
    figsize  = (9,9)
    dpi      = 40

    # Remove previous images if any.
    file_names=os.listdir(parent_dir+out_dir)
    for file in file_names:
        os.remove(parent_dir+out_dir+file)

    print('Reading and cropping image...')
    image = mpimg.imread(parent_dir+in_dir+image_file)
    image = crop_center(image=image,w=np.min(image.shape[0:2]),h=np.min(image.shape[0:2]))

    poin_rec = poincare_recurrence(image=image, dir_name=parent_dir+out_dir, name='img_',
                                transform_list=transform_list, figsize=figsize, dpi=dpi)
    poin_rec.run(N=N)
    poin_rec.make_movie(duration=duration, dynamic=dynamic)