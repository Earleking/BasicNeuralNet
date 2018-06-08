import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import numbers
import random
import time
class cNN:
    def __init__(self):
        print("hello world")
    def apply_convolution(self, img, filter, step):
        # Pad the image properly
        #get excess cells
        remainderY = len(img) % len(filter)
        remainderX = len(img[0]) % len(filter[0])
        #create new sized image
        if(isinstance(img[0][0], numbers.Number)):
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX])
        else:
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX, 4])
        start_x = math.floor(remainderX / 2)
        start_y = math.floor(remainderY / 2)
        padded_img[start_y:start_y + len(img), start_x:start_x + len(img[0])] = img
        img = padded_img


        new_array = []
        for y_index in range(0, len(img) - len(filter), step):
            new_array.append([])
            for x_index in range(0, len(img[0]) - len(filter[0]), step):
                #select area that will be convulted
                img_target_area = img[y_index:y_index + len(filter), x_index: x_index + len(filter[0])]
                temp = []

                if(isinstance(img_target_area[0][0], numbers.Number)):
                    temp = np.sum(np.multiply(filter, img_target_area[:,:]))
                else:
                    #create temp object to store each layer
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,0]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,1]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,2]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,3]))))

                #add temp object to the new image array
                new_array[int(y_index/step)].append(temp)

        return np.asarray(new_array)

    def apply_pooling(self, img, size, step):
        # Pad the image properly
        #get excess cells
        remainderY = len(img) % size
        remainderX = len(img[0]) % size
        #create new sized image
        if(isinstance(img[0][0], numbers.Number)):
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX])
        else:
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX, 4])

        start_x = math.floor(remainderX / 2)
        start_y = math.floor(remainderY / 2)
        padded_img[start_y:start_y + len(img), start_x:start_x + len(img[0])] = img
        img = padded_img

        new_array = []
        for y_index in range(0, len(img) - size, step):
            new_array.append([])
            for x_index in range(0, len(img[0]) - size, step):
                temp = []

                if(isinstance(img[0][0], numbers.Number)):
                    temp = np.max(img[y_index:y_index + size, x_index:x_index + size])
                else:
                    temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 0]))
                    temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 1]))
                    temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 2]))
                    temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 3]))

                new_array[int(y_index/step)].append(temp)
        return np.asarray(new_array)

class convolution_layer:
    def __init__(self, filter_size, depth, step):
        self.filters = []
        self.weights = []
        self.step = step
        for i in range(depth):
            self.filters.append([[(random.random() * 2) - 1 for y in range(int(filter_size[1]))] for z in range(int(filter_size[0]))])
            self.weights.append(random.random() - 0.5)
        # print(self.filters)
    def run(self, img):
        # use the hard-coded version for testing. All it should do is blur the image
        filter = self.filters[0]
        # filter = [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]]
        # Pad the image properly
        #get excess cells
        remainderY = len(img) % len(filter)
        remainderX = len(img[0]) % len(filter[0])
        #create new sized image
        if(isinstance(img[0][0], numbers.Number)):
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX])
        else:
            padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX, 4])
        start_x = math.floor(remainderX / 2)
        start_y = math.floor(remainderY / 2)
        padded_img[start_y:start_y + len(img), start_x:start_x + len(img[0])] = img
        img = padded_img


        new_array = []
        for y_index in range(0, len(img) - len(filter), self.step):
            new_array.append([])
            for x_index in range(0, len(img[0]) - len(filter[0]), self.step):
                #select area that will be convulted
                img_target_area = img[y_index:y_index + len(filter), x_index: x_index + len(filter[0])]
                #create temp object that will store stuff that is to be added to the image array later
                temp = []
                #check if its a multi or single channel image
                if(isinstance(img_target_area[0][0], numbers.Number)):
                    temp = np.sum(np.multiply(filter, img_target_area[:,:]))
                else:
                    #sum up the matrix created from each layer in the image
                    runningSum = 0
                    for i in range(len(img_target_area[0][0])):
                        runningSum += np.sum(np.multiply(filter, img_target_area[:,:,i]))
                    temp = runningSum + self.weights[0] #change 0 to a var. Same as at the top
                    temp = max(0, temp)
                    # or keep the color?
                    # for i in range(len(img_target_area[0][0])):
                    #     temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,i]))))
                
                #add temp object to the new image array
                new_array[int(y_index/self.step)].append(temp)

        return np.asarray(new_array)

img = mpimg.imread("data_files/icon.png")
net = cNN()
# test = net.apply_convolution(img[:,:,2], [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]], 1)
# test = net.apply_pooling(test, 2, 2)
# test = net.apply_convolution(test, [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]], 3)
# test = net.apply_pooling(test, 2, 2)
t0 = time.time()
conv = convolution_layer([3,3], 3, 4)
test = conv.run(img)
print(time.time() - t0)
# test = net.apply_convolution(img, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 2)
# test = test[:,:,1]
# test[10][20] = [0, 0, 0, 1]
# print(test)
# test = [[1, 0, 1], [0, 1, 0]]
plt.imshow(test)
plt.show()

# test = [[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [1, 4, 3]]]
# test[:][:] = [3, 2]    
# print(test[:][:])