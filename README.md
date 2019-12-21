# animal-classification
AI model to classify different animals using neural networks and image processing. 

Output of Training: 
'hippopotamus', 'rhinoceros', 'raccoon', 'killer+whale', 'dalmatian', 'moose', 'antelope', 'persian+cat', 'siamese+cat', 'chihuahua', 'ox', 'rat', 'otter', 'spider+monkey', 'collie', 'seal', 'mouse', 'beaver', 'wolf', 'chimpanzee', 'grizzly+bear', 'buffalo', 'bobcat', 'bat', 'weasel', 'walrus', 'mole', 'horse', 'german+shepherd', 'squirrel']
322 photos of hippopotamus
332 photos of rhinoceros
243 photos of raccoon
127 photos of killer+whale
241 photos of dalmatian
344 photos of moose
486 photos of antelope
353 photos of persian+cat
237 photos of siamese+cat
255 photos of chihuahua
347 photos of ox
148 photos of rat
367 photos of otter
136 photos of spider+monkey
454 photos of collie
470 photos of seal
89 photos of mouse
86 photos of beaver
264 photos of wolf
323 photos of chimpanzee
405 photos of grizzly+bear
420 photos of buffalo
288 photos of bobcat
181 photos of bat
117 photos of weasel
89 photos of walrus
47 photos of mole
780 photos of horse
484 photos of german+shepherd
565 photos of squirrel
(9000,)
(9000, 30)
Processed 0 of 9000
Processed 51 of 9000
Processed 102 of 9000
Processed 153 of 9000
Processed 204 of 9000
Processed 255 of 9000
Processed 306 of 9000
Processed 357 of 9000
Processed 408 of 9000
Processed 459 of 9000
Processed 510 of 9000
Processed 561 of 9000
Processed 612 of 9000
Processed 663 of 9000
Processed 714 of 9000
Processed 765 of 9000
Processed 816 of 9000
Processed 867 of 9000
Processed 918 of 9000
Processed 969 of 9000
Processed 1020 of 9000
Processed 1071 of 9000
Processed 1122 of 9000
Processed 1173 of 9000
Processed 1224 of 9000
Processed 1275 of 9000
Processed 1326 of 9000
Processed 1377 of 9000
Processed 1428 of 9000
Processed 1479 of 9000
Processed 1530 of 9000
Processed 1581 of 9000
Processed 1632 of 9000
Processed 1683 of 9000
Processed 1734 of 9000
Processed 1785 of 9000
Processed 1836 of 9000
Processed 1887 of 9000
Processed 1938 of 9000
Processed 1989 of 9000
Processed 2040 of 9000
Processed 2091 of 9000
Processed 2142 of 9000
Processed 2193 of 9000
Processed 2244 of 9000
Processed 2295 of 9000
Processed 2346 of 9000
Processed 2397 of 9000
Processed 2448 of 9000
Processed 2499 of 9000
Processed 2550 of 9000
Processed 2601 of 9000
Processed 2652 of 9000
Processed 2703 of 9000
Processed 2754 of 9000
Processed 2805 of 9000
Processed 2856 of 9000
Processed 2907 of 9000
Processed 2958 of 9000
Processed 3009 of 9000
Processed 3060 of 9000
Processed 3111 of 9000
Processed 3162 of 9000
Processed 3213 of 9000
Processed 3264 of 9000
Processed 3315 of 9000
Processed 3366 of 9000
Processed 3417 of 9000
Processed 3468 of 9000
Processed 3519 of 9000
Processed 3570 of 9000
Processed 3621 of 9000
Processed 3672 of 9000
Processed 3723 of 9000
Processed 3774 of 9000
Processed 3825 of 9000
Processed 3876 of 9000
Processed 3927 of 9000
Processed 3978 of 9000
Processed 4029 of 9000
Processed 4080 of 9000
Processed 4131 of 9000
Processed 4182 of 9000
Processed 4233 of 9000
Processed 4284 of 9000
Processed 4335 of 9000
Processed 4386 of 9000
Processed 4437 of 9000
Processed 4488 of 9000
Processed 4539 of 9000
Processed 4590 of 9000
Processed 4641 of 9000
Processed 4692 of 9000
Processed 4743 of 9000
Processed 4794 of 9000
Processed 4845 of 9000
Processed 4896 of 9000
Processed 4947 of 9000
Processed 4998 of 9000
Processed 5049 of 9000
Processed 5100 of 9000
Processed 5151 of 9000
Processed 5202 of 9000
Processed 5253 of 9000
Processed 5304 of 9000
Processed 5355 of 9000
Processed 5406 of 9000
Processed 5457 of 9000
Processed 5508 of 9000
Processed 5559 of 9000
Processed 5610 of 9000
Processed 5661 of 9000
Processed 5712 of 9000
Processed 5763 of 9000
Processed 5814 of 9000
Processed 5865 of 9000
Processed 5916 of 9000
Processed 5967 of 9000
Processed 6018 of 9000
Processed 6069 of 9000
Processed 6120 of 9000
Processed 6171 of 9000
Processed 6222 of 9000
Processed 6273 of 9000
Processed 6324 of 9000
Processed 6375 of 9000
Processed 6426 of 9000
Processed 6477 of 9000
Processed 6528 of 9000
Processed 6579 of 9000
Processed 6630 of 9000
Processed 6681 of 9000
Processed 6732 of 9000
Processed 6783 of 9000
Processed 6834 of 9000
Processed 6885 of 9000
Processed 6936 of 9000
Processed 6987 of 9000
Processed 7038 of 9000
Processed 7089 of 9000
Processed 7140 of 9000
Processed 7191 of 9000
Processed 7242 of 9000
Processed 7293 of 9000
Processed 7344 of 9000
Processed 7395 of 9000
Processed 7446 of 9000
Processed 7497 of 9000
Processed 7548 of 9000
Processed 7599 of 9000
Processed 7650 of 9000
Processed 7701 of 9000
Processed 7752 of 9000
Processed 7803 of 9000
Processed 7854 of 9000
Processed 7905 of 9000
Processed 7956 of 9000
Processed 8007 of 9000
Processed 8058 of 9000
Processed 8109 of 9000
Processed 8160 of 9000
Processed 8211 of 9000
Processed 8262 of 9000
Processed 8313 of 9000
Processed 8364 of 9000
Processed 8415 of 9000
Processed 8466 of 9000
Processed 8517 of 9000
Processed 8568 of 9000
Processed 8619 of 9000
Processed 8670 of 9000
Processed 8721 of 9000
Processed 8772 of 9000
Processed 8823 of 9000
Processed 8874 of 9000
Processed 8925 of 9000
Processed 8976 of 9000
(9000, 224, 224, 3)
(8550, 224, 224, 3)
(8550, 30)
(450, 224, 224, 3)
(450, 30)
2019-12-21 01:31:09.267621: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
conv1_pad (ZeroPadding2D)    (None, 225, 225, 3)       0         
_________________________________________________________________
conv1 (Conv2D)               (None, 112, 112, 32)      864       
_________________________________________________________________
conv1_bn (BatchNormalization (None, 112, 112, 32)      128       
_________________________________________________________________
conv1_relu (ReLU)            (None, 112, 112, 32)      0         
_________________________________________________________________
conv_dw_1 (DepthwiseConv2D)  (None, 112, 112, 32)      288       
_________________________________________________________________
conv_dw_1_bn (BatchNormaliza (None, 112, 112, 32)      128       
_________________________________________________________________
conv_dw_1_relu (ReLU)        (None, 112, 112, 32)      0         
_________________________________________________________________
conv_pw_1 (Conv2D)           (None, 112, 112, 64)      2048      
_________________________________________________________________
conv_pw_1_bn (BatchNormaliza (None, 112, 112, 64)      256       
_________________________________________________________________
conv_pw_1_relu (ReLU)        (None, 112, 112, 64)      0         
_________________________________________________________________
conv_pad_2 (ZeroPadding2D)   (None, 113, 113, 64)      0         
_________________________________________________________________
conv_dw_2 (DepthwiseConv2D)  (None, 56, 56, 64)        576       
_________________________________________________________________
conv_dw_2_bn (BatchNormaliza (None, 56, 56, 64)        256       
_________________________________________________________________
conv_dw_2_relu (ReLU)        (None, 56, 56, 64)        0         
_________________________________________________________________
conv_pw_2 (Conv2D)           (None, 56, 56, 128)       8192      
_________________________________________________________________
conv_pw_2_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_pw_2_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_dw_3 (DepthwiseConv2D)  (None, 56, 56, 128)       1152      
_________________________________________________________________
conv_dw_3_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_dw_3_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_pw_3 (Conv2D)           (None, 56, 56, 128)       16384     
_________________________________________________________________
conv_pw_3_bn (BatchNormaliza (None, 56, 56, 128)       512       
_________________________________________________________________
conv_pw_3_relu (ReLU)        (None, 56, 56, 128)       0         
_________________________________________________________________
conv_pad_4 (ZeroPadding2D)   (None, 57, 57, 128)       0         
_________________________________________________________________
conv_dw_4 (DepthwiseConv2D)  (None, 28, 28, 128)       1152      
_________________________________________________________________
conv_dw_4_bn (BatchNormaliza (None, 28, 28, 128)       512       
_________________________________________________________________
conv_dw_4_relu (ReLU)        (None, 28, 28, 128)       0         
_________________________________________________________________
conv_pw_4 (Conv2D)           (None, 28, 28, 256)       32768     
_________________________________________________________________
conv_pw_4_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_pw_4_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_dw_5 (DepthwiseConv2D)  (None, 28, 28, 256)       2304      
_________________________________________________________________
conv_dw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_dw_5_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_pw_5 (Conv2D)           (None, 28, 28, 256)       65536     
_________________________________________________________________
conv_pw_5_bn (BatchNormaliza (None, 28, 28, 256)       1024      
_________________________________________________________________
conv_pw_5_relu (ReLU)        (None, 28, 28, 256)       0         
_________________________________________________________________
conv_pad_6 (ZeroPadding2D)   (None, 29, 29, 256)       0         
_________________________________________________________________
conv_dw_6 (DepthwiseConv2D)  (None, 14, 14, 256)       2304      
_________________________________________________________________
conv_dw_6_bn (BatchNormaliza (None, 14, 14, 256)       1024      
_________________________________________________________________
conv_dw_6_relu (ReLU)        (None, 14, 14, 256)       0         
_________________________________________________________________
conv_pw_6 (Conv2D)           (None, 14, 14, 512)       131072    
_________________________________________________________________
conv_pw_6_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_6_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_7 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_7_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_7 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_7_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_7_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_8 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_8_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_8 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_8_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_8_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_9 (DepthwiseConv2D)  (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_9_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_9 (Conv2D)           (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_9_bn (BatchNormaliza (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_9_relu (ReLU)        (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_10 (DepthwiseConv2D) (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_10_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_10 (Conv2D)          (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_10_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_10_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_dw_11 (DepthwiseConv2D) (None, 14, 14, 512)       4608      
_________________________________________________________________
conv_dw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_dw_11_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pw_11 (Conv2D)          (None, 14, 14, 512)       262144    
_________________________________________________________________
conv_pw_11_bn (BatchNormaliz (None, 14, 14, 512)       2048      
_________________________________________________________________
conv_pw_11_relu (ReLU)       (None, 14, 14, 512)       0         
_________________________________________________________________
conv_pad_12 (ZeroPadding2D)  (None, 15, 15, 512)       0         
_________________________________________________________________
conv_dw_12 (DepthwiseConv2D) (None, 7, 7, 512)         4608      
_________________________________________________________________
conv_dw_12_bn (BatchNormaliz (None, 7, 7, 512)         2048      
_________________________________________________________________
conv_dw_12_relu (ReLU)       (None, 7, 7, 512)         0         
_________________________________________________________________
conv_pw_12 (Conv2D)          (None, 7, 7, 1024)        524288    
_________________________________________________________________
conv_pw_12_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_12_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
conv_dw_13 (DepthwiseConv2D) (None, 7, 7, 1024)        9216      
_________________________________________________________________
conv_dw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_dw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 30)                30750     
=================================================================
Total params: 3,259,614
Trainable params: 3,237,726
Non-trainable params: 21,888
_________________________________________________________________
Train on 8550 samples, validate on 450 samples
Epoch 1/5
8550/8550 [==============================] - 1451s 170ms/step - loss: 3.0011 - acc: 0.2220 - val_loss: 2.1632 - val_acc: 0.4111
8550/8550 [==============================] - 1451s 170ms/step - loss: 3.0011 - acc: 0.2220 - val_loss: 2.1632 - val_acc: 0.4111
8550/8550 [==============================] - 1451s 170ms/step - loss: 3.0011 - acc: 0.2220 - val_loss: 2.1632 - val_acc: 0.4111
8550/8550 [==============================] - 1451s 170ms/step - loss: 3.0011 - acc: 0.2220 - val_loss: 2.1632 - val_acc: 0.4111
8550/8550 [==============================] - 1451s 170ms/step - loss: 3.0011 - acc: 0.2220 - val_loss: 2.1632 - val_acc: 0.4111
