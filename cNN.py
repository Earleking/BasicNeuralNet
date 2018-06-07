import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
class cNN:
    def __init__(self):
        print("hello world")
    def create_convolution_layer(self):
        print("ccl")
    def apply_convolution(self, img, filter, step):
        # Pad the image properly
        #get excess cells
        remainderY = len(img) % len(filter)
        remainderX = len(img[0]) % len(filter[0])
        #create new sized image
        padded_img = np.zeros([len(img) + remainderY, len(img[0]) + remainderX, 4])
        start_x = math.floor(remainderX / 2)
        start_y = math.floor(remainderY / 2)
        padded_img[start_y:start_y + len(img), start_x:start_x + len(img[0])] = img
        img = padded_img


        new_array = []
        for y_index in range(0, len(img) - len(filter), step):
            new_array.append([])
            for x_index in range(0, len(img[0]) - len(filter[0]), step):
                try:
                    #select area that will be convulted
                    img_target_area = img[y_index:y_index + len(filter), x_index: x_index + len(filter[0])]

                    #create temp object to store each layer
                    temp = []
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,0]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,1]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,2]))))
                    temp.append(max(0, np.sum(np.multiply(filter, img_target_area[:,:,3]))))

                    #add temp object to the new image array
                    new_array[int(y_index/step)].append(temp)
                    
                except ValueError as e:
                    print(e)    
                    quit()
        return np.asarray(new_array)

    def apply_pooling(self, img, size, step):
        # Pad the image properly
        #get excess cells
        remainderY = len(img) % size
        remainderX = len(img[0]) % size
        #create new sized image
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
                temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 0]))
                temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 1]))
                temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 2]))
                temp.append(np.max(img[y_index:y_index + size, x_index:x_index + size, 3]))

                new_array[int(y_index/step)].append(temp)
        return np.asarray(new_array)


img = mpimg.imread("data_files/icon.png")
net = cNN()
test = net.apply_convolution(img, [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]], 1)
test = net.apply_pooling(test, 2, 2)
test = net.apply_convolution(test, [[.1, .1, .1], [.1, .1, .1], [.1, .1, .1]], 3)
test = net.apply_pooling(test, 2, 2)
#test = net.apply_convolution(img, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 2)
test = test[:,:,0]
# test[10][20] = [0, 0, 0, 1]
# print(test)
plt.imshow(test)
plt.show()

# test = [[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [1, 4, 3]]]
# test[:][:] = [3, 2]    
# print(test[:][:])