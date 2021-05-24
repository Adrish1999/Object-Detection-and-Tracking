import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from filterpy.discrete_bayes import normalize
from filterpy.discrete_bayes import update
from filterpy.discrete_bayes import predict

# Basic Utility Functions
def create_ground_truth(img):
    '''
    Create the background image by setting the objects location pixals as white
    '''
    img = np.array(img)
    img[0:200, 0: 200 , :] = np.ones((200, 200, 3))*255
    return img

def diff_mask(image, background):
    ''' 
    Subtract the current image (with the object) with the background image to get a blob mask 
    '''
    diff = cv2.subtract(background, image)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, res = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
    return res

def find_region(mask):
    ''' 
    Use the blob mask to find the bouding box of the image 
    '''
    rows = np.where(np.any(mask!=0, axis=1))[0]
    columns = np.where(np.any(mask!=0, axis=0))[0]
    try:
        top_left = (min(columns), min(rows))
        bottom_right = (max(columns), max(rows))
    except:
        top_left = (-1,-1)
        bottom_right = (-1, -1)
    return top_left, bottom_right

def find_center(bound_box):
    ''' 
    Find center of the detected image 
    '''
    if not((bound_box[0][0] == -1 and bound_box[0][1] == -1) or (bound_box[1][0] == -1 and bound_box[1][1] == -1) ):
        x = int((bound_box[1][0]+bound_box[0][0])/2)
        y = int((bound_box[1][1]+bound_box[0][1])/2)
        return (x,y)
    else:
        return (-1,-1)


def bar_plot(pos, x=None, ylim=(0,1), title=None, c='#30a2da',**kwargs):
    ax = plt.gca()
    if x is None:
        x = np.arange(len(pos))
    ax.bar(x, pos, color=c, **kwargs)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(np.asarray(x), x)
    if title is not None:
        plt.title(title)


def lh_space(track, z, z_prob):
    ''' 
    Compute likelihood that a measurement matches positions in the track. 
    '''
    try:
        scale = z_prob / (1.0 - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(track))
    likelihood[track==z] *= scale

    return likelihood



class Train(object):

    def __init__(self, track_len, kernel=[1.0], sensor_accuracy=0.9):
        self.track_len = track_len
        self.pos = 0
        self.kernel = kernel
        self.sensor_accuracy = sensor_accuracy

    def move(self, distance=1):
        '''
        Move in the specified direction with some small chance of error
        '''
        self.pos += distance
        # insert random movement error according to kernel
        r = random.random()
        s = 0
        offset = -(len(self.kernel) - 1) / 2
        for k in self.kernel:
            s += k
            if r <= s:
                break
            offset += 1
        self.pos = int((self.pos + offset) % self.track_len)
        return self.pos

    def sense(self):
        pos = self.pos
         # insert random sensor error
        if random.random() > self.sensor_accuracy:
            if random.random() > 0.5:
                pos += 1
            else:
                pos -= 1
        return pos


def train_filter(l,iterations, kernel, sensor_accuracy, move_distance, do_print=True):
    temp = [i for i in range(len(l))]
    track = np.array(temp)
    prior = np.array([.9] + [0.01]*1919999)
    posterior = prior[:]
    normalize(prior)
    
    object = Train(len(track), kernel, sensor_accuracy)
    for i in range(iterations):
        # move the robot and
        object.move(distance=move_distance)

        # peform prediction
        prior = predict(posterior, move_distance, kernel)       

        #  and update the filter
        m = object.sense()
        likelihood = lh_space(track, m, sensor_accuracy)
        posterior = update(likelihood, prior)
        index = np.argmax(posterior)

        if do_print:
            print(f'time {i}: pos {object.pos}, sensed {m}, at position {track[object.pos]}')
            conf = posterior[index] * 100
            print(f'estimated position is {index} with confidence {conf:.4f}%:')            

    bar_plot(posterior)
    if do_print:
        print()
        print('final position is', object.pos)
        index = np.argmax(posterior)
        print('''Estimated position is {} with '''
               '''confidence {:.4f}%:'''.format(
                 index, posterior[index]*100))


def main():
    # Import the simulation video
    cap = cv2.VideoCapture('Red_Ball_with_obstracle.mp4')

    if(cap.isOpened() == False):
        print("Error opening video")

    ret, frame = cap.read()

    # Create the background
    background = create_ground_truth(frame)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            diff = diff_mask(frame,background)
            frame_out = np.array(frame)
            bound_box = find_region(diff)
            center = find_center(bound_box)
            with_rect = cv2.rectangle(frame, bound_box[0], bound_box[1], (255, 0, 0), 2)
            
            #Apply Bayes Filter
            t = time.time()
            b = np.reshape(frame_out, (np.product(frame_out.shape),))
            #print(b.shape)
            np.set_printoptions(precision=2, suppress=True, linewidth=60)
            train_filter(b,1, kernel=[.1, .8, .1], sensor_accuracy=.9,move_distance=1, do_print=True)
            
            print("Time required to process frame: %.3f seconds"%(time.time()-t))


            cv2.imshow("video", cv2.resize(with_rect, (200, 200)))
            cv2.imshow("detector", cv2.resize(diff,(200,200)))
            
            if cv2.waitKey(20) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__  == '__main__':
    main()