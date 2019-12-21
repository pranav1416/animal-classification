
import tensorflow as tf
import cv2
import os
import numpy as np
import keras.backend as K
from keras.models import load_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)

model = load_model('model.hdf5')

TRAIN_DIR = 'images/train/'
TEST_DIR = 'images/test/'

animalss = os.listdir(TRAIN_DIR)
print(animalss)

img_height,img_width = (224,224)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

tf.train.write_graph(frozen_graph, '', "my_model14.pb", as_text=False)

# cap = cv2.VideoCapture('test.mp4')

# count = 0
# #cap.open('drivetest1.mp4')
# while(cap.isOpened()) :
#   count = count+1
#   ret, frame = cap.read()
#   #frame = np.asarray(frame)
#   if(count%25==0):
#     #print(count)
#     frame = cv2.resize(frame, (img_height,img_width), interpolation=cv2.INTER_CUBIC)
#     image = np.expand_dims(frame, axis=0)
#     sih1223 = model.predict(image)[0]
#     species = np.argmax(sih1223,axis = 0)
#     prob = max(sih1223)
#     if(prob>0.6):
#       print('Frame No:',count,' Class of animal : ', animalss[species],'\tConfidence score: ',prob)
  

# cap.release()
# cv2.destroyAllWindows()


frame = cv2.imread(TEST_DIR+'Img-3.jpg', cv2.IMREAD_COLOR)
frame = cv2.resize(frame, (img_height,img_width), interpolation=cv2.INTER_CUBIC)
frame.shape
image = np.expand_dims(frame, axis=0)
sih1223 = model.predict(image)[0]
species = np.argmax(sih1223,axis = 0)
prob = np.argmax(sih1223)
print("Class of animal : ", animalss[species])

# classify the input image and initialize the label and
# probability of the prediction
