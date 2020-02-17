#/usr/bin/python
# coding=utf-8

"""
@version: 
@author: Chenxing Li
@license: Apache Licence 
@contact: lichenxing007@gmail.com
@site: 
@software: PyCharm Community Edition
@file: RunDnn.py
@time: 12/21/16 2:15 PM
"""

import tensorflow as tf
import zpkg.io.fea as zfea
import zpkg.io.feacfg as zfeacfg
import zpkg.nnet.layers as layers
import zpkg.io.file as zfile
import zpkg.io.mod as zmod
import zpkg.io.kmod as zkmod
from tensor2tensor.common_attention import multihead_attention, attention_bias_ignore_padding
import tensorflow.contrib.slim as slim

import argparse
import numpy as np
import os
import shutil
import math

#args
parser = argparse.ArgumentParser()
trn_fea_cfg = zfeacfg.FeaCfg()
cv_fea_cfg = zfeacfg.FeaCfg()

#batch
FEA_TRAIN_BATCH_SIZE = 1
CV_BATCH_SIZE=1
READ_THREAD_NUM = 1

#optimizer
LEARNING_RATE_INIT = 0.001
MOMENTUM = 0.9

#GPU CPU
#GPU_DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2','/gpu:3','/gpu:4','/gpu:5','/gpu:6','/gpu:7']
GPU_DEVICE_NUM = 1
MAX_GRAD_NROM=5.0

def getfilelst(scp_file_path):
    # get tf list
    tf_list = []
    with open(scp_file_path) as list_file:
        for line in list_file:
            tf_list.append(line.strip())
    return  tf_list

def saveModel(sess, saver, model_path):
    print('[*] Writing checkpoints...')
    saver.save(sess, model_path)
    return True

def loadModel(sess, saver, model_path):
    print('[*] Reading checkpoints...')
    saver.restore(sess, model_path)
    return True

def lrelu(x, leak = 0.2):
    return tf.maximum(x, leak * x)

def batch_norm(x, train, data_format ='NHWC', name = 'batch_norm', act = lrelu, epsilon = 1e-5, momentum = 0.9):
    return slim.batch_norm(x, decay=momentum, updates_collections = None, epsilon = epsilon, \
        scale = True, fused = True, is_training = train, activation_fn = act, data_format = data_format, scope = name)

def conv2d(input_, output_dim, k_h, k_w, s_h, s_w, 
           stddev=0.02, name="conv2d", is_pooling=False, bn=True, is_train=True, reuse=True):
    with tf.variable_scope(name,reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        
        if bn:
            h = batch_norm(h, train=is_train)
        # Maxpooling over the outputs
        if is_pooling :
            h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],)
            #        padding='SAME',name="pool")
            #h = tf.reduce_max(h,axis=1)
    return h

def conv2d_withoutact(input_, output_dim, k_h, k_w, s_h, s_w, 
           stddev=0.02, name="conv2d", is_pooling=False, bn=True, is_train=True, reuse=True):
    with tf.variable_scope(name,reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')
        
        h = tf.nn.bias_add(conv, b)

        if bn:
            h = batch_norm(h, train=is_train)

        # Maxpooling over the outputs
        if is_pooling :
            h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],)
            #        padding='SAME',name="pool")
            #h = tf.reduce_max(h,axis=1)
    return h

def gatedConv2d(input_, input_dim, output_dim, k_h, k_w, s_h, s_w, 
           stddev=0.02, name="gatedConv2d", is_pooling=False, bn=True, is_train=True, reuse=True):
    with tf.variable_scope(name,reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_dim, output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        v = tf.get_variable('v', [k_h, k_w, input_dim, output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b',[output_dim], initializer=tf.constant_initializer(0.0))
        c = tf.get_variable('c',[output_dim], initializer=tf.constant_initializer(0.0))
        convw = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')
        convv = tf.nn.conv2d(input_, v, strides=[1, s_h, s_w, 1], padding='SAME')

        hw = tf.nn.bias_add(convw, b)
        hv = tf.nn.bias_add(convv, c)

        if bn:
            #hw = tf.layers.batch_normalization(hw, training=is_train)
            hv = batch_norm(hv, train=is_train)

        # Apply nonlinearity
        h = tf.multiply(tf.nn.sigmoid(hw, name="sigmoid"),hv)

        # Maxpooling over the outputs
        if is_pooling :
            h = tf.nn.max_pool(h, ksize=[1, 1, 2, 1],strides=[1, 1, 2, 1],
                  padding='SAME',name="pool")
            #h = tf.reduce_max(h,axis=1)
    return h

def pool(input_, k_h, k_w, s_h, s_w, 
           stddev=0.02, name="pool",reuse=True):
    with tf.variable_scope(name,reuse=reuse):
            h = tf.nn.max_pool(input_, ksize=[1, k_h, k_w, 1],strides=[1, s_h, s_w, 1],
                  padding='SAME',name="pool")
            #h = tf.reduce_max(h,axis=1)
    return h

def dnn_layer(input, scope='dnn_layer',training=True,reuse=True):
    """input layer"""

    with tf.variable_scope(scope,reuse=reuse):
        outputs = tf.reshape(input, [-1, trn_fea_cfg.fea_dim])
        outputs = tf.layers.dense(outputs, units=param.hidden_size,
                                      activation=tf.nn.tanh,
                                      reuse=tf.get_variable_scope().reuse)
        if training:
            outputs = tf.reshape(
                outputs, [FEA_TRAIN_BATCH_SIZE, -1, param.hidden_size])
        else:
            outputs = tf.reshape(
                outputs, [1, -1, param.hidden_size])

    return outputs

def output_layer(lstm_output, output_dim, scope='output_layer',reuse=True):
    """Output layer"""
    input_dim = lstm_output.get_shape().as_list()[2]
    output_dim = output_dim
    with tf.variable_scope(scope,reuse=reuse):
      # Output layer
      linearity = tf.get_variable(
          "output_linearity", [input_dim, output_dim], dtype=tf.float32,
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      bias = tf.get_variable(
          "output_bias", [output_dim], dtype=tf.float32,
          initializer=tf.constant_initializer(0.0))

      predicts = tf.matmul(tf.reshape(lstm_output, [-1, input_dim]), linearity) + bias

    return predicts

def deconv2d(input_, output_dim, k_h, k_w, s_h, s_w, 
           stddev=0.02, name="deconv2d",reuse=True):
    with tf.variable_scope(name,reuse=reuse):
        deconv = tf.layers.conv2d_transpose(input_, output_dim, [k_h, k_w], strides=[s_h, s_w],padding='SAME')

    return deconv
    
def residual(inputs, frame_weights, dropout_rate, is_training, name='self_attention_block',reuse=True):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float.
        is_training: A bool.

    Returns:
        A Tensor.
    """
    with tf.variable_scope(name,reuse=reuse):
        if is_training:
            mul_atten=multihead_attention(query_antecedent=inputs,
                            bias=frame_weights,
                            total_key_depth=param.atten_hidden_units,
                            total_value_depth=param.atten_hidden_units,
                            output_depth=param.atten_hidden_units,
                            num_heads=param.atten_num_heads,
                            dropout_rate=param.atten_dropout_rate,
                            name='mul_atten1',
                            summaries=False,
                            reuse=reuse)
        else:
            mul_atten=multihead_attention(query_antecedent=inputs,
                            bias=frame_weights,
                            total_key_depth=param.atten_hidden_units,
                            total_value_depth=param.atten_hidden_units,
                            output_depth=param.atten_hidden_units,
                            num_heads=param.atten_num_heads,
                            dropout_rate=0.0,
                            name='mul_atten1',
                            summaries=False,
                            reuse=reuse)
        
        att_output = inputs + mul_atten
        att_output = layer_norm(att_output,name='ln1',reuse=reuse)

    return att_output

def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
      filters = x.get_shape()[-1]
  with tf.variable_scope(name, default_name="layer_norm", values=[x], reuse=reuse):
      scale = tf.get_variable(
          "layer_norm_scale", [filters], initializer=tf.ones_initializer())
      bias = tf.get_variable(
          "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
      mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    
      variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
      result = norm_x * scale + bias
    
  return result


def build_CLDNN_model_graph_train(inputs,frame_weights,is_training=True,reuse=False):

    #encoder
    cnn_input = tf.expand_dims(inputs,-1)
    cnn_output1 = gatedConv2d(cnn_input, 1, 8, 3, 3, 1, 1, name = "cnn_layer1", is_pooling=True, bn=True, is_train=is_training, reuse=reuse)
    cnn_output2 = gatedConv2d(cnn_output1, 8, 8, 3, 3, 1, 1, name = "cnn_layer2", is_pooling=True, bn=True, is_train=is_training, reuse=reuse)
    
    #self attention
    attention_out1 = residual(cnn_output2, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block1',reuse=reuse)
    attention_input2 = gatedConv2d(attention_out1, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer3", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    
    attention_out2 = residual(attention_input2, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block2',reuse=reuse)
    attention_input3 = gatedConv2d(attention_out2, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer4", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    
    attention_out3 = residual(attention_input3, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block3',reuse=reuse)
    attention_input4 = gatedConv2d(attention_out3, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer5", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)

    attention_out4 = residual(attention_input4, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block4',reuse=reuse)
    
    attention_out5=attention_out4+cnn_output2
    deconv_output1=deconv2d(attention_out5, 8, 3, 3, 1, 2, name="deconvHidden1",reuse=reuse)
    deconv_output1 = deconv_output1+cnn_output1
    deconv_output2=deconv2d(deconv_output1, 1, 3, 3, 1, 2, name="deconvHidden2",reuse=reuse)

    cnn_output1_mask = gatedConv2d(cnn_input, 1, 8, 3, 3, 1, 1, name = "mask_cnn_layer1", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output1_mask = pool(cnn_output1_mask, 3, 3, 2, 2,name="pool1",reuse=reuse)
    cnn_output2_mask = gatedConv2d(cnn_output1_mask, 8, 8, 3, 3, 1, 1, name = "mask_cnn_layer2", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output2_mask = pool(cnn_output2_mask, 3, 3, 2, 2,name="pool2",reuse=reuse)
    cnn_output3_mask = gatedConv2d(cnn_output2_mask, 8, 8, 3, 3, 1, 1, name = "mask_cnn_layer3", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    deconv_mask_output1=deconv2d(cnn_output3_mask, 8, 3, 3, 2, 2, name="deconvHidden_mask1",reuse=reuse)
    deconv_maks_output2=deconv2d(deconv_mask_output1, 8, 3, 3, 2, 2, name="deconvHidden_mask2",reuse=reuse)
    
    cnn_output4_mask = conv2d_withoutact(deconv_maks_output2, 1, 3, 3, 1, 1, name = "mask_cnn_layer4", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output4_mask = tf.sigmoid(cnn_output4_mask)
    deconv_output2 = tf.multiply(deconv_output2, cnn_output4_mask)

    return deconv_output2

def build_CLDNN_model_graph_test(inputs,frame_weights,is_training=False,reuse=True):
    """Test model"""
    
    cnn_input = tf.expand_dims(inputs,-1)
    cnn_output1 = gatedConv2d(cnn_input, 1, 8, 3, 3, 1, 1, name = "cnn_layer1", is_pooling=True, bn=True, is_train=is_training, reuse=reuse)
    cnn_output2 = gatedConv2d(cnn_output1, 8, 8, 3, 3, 1, 1, name = "cnn_layer2", is_pooling=True, bn=True, is_train=is_training, reuse=reuse)
    
    #self attention
    attention_out1 = residual(cnn_output2, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block1',reuse=reuse)
    attention_input2 = gatedConv2d(attention_out1, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer3", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    
    attention_out2 = residual(attention_input2, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block2',reuse=reuse)
    attention_input3 = gatedConv2d(attention_out2, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer4", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    
    attention_out3 = residual(attention_input3, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block3',reuse=reuse)
    attention_input4 = gatedConv2d(attention_out3, param.atten_hidden_units, 8, 3, 3, 1, 1, name = "conv_layer5", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    
    attention_out4 = residual(attention_input4, frame_weights, param.residual_dropout_rate, is_training, name='self_atten_block4',reuse=reuse)
    
    attention_out5=attention_out4+cnn_output2
    deconv_output1=deconv2d(attention_out5, 8, 3, 3, 1, 2, name="deconvHidden1",reuse=reuse)
    deconv_output1 = deconv_output1+cnn_output1
    deconv_output2=deconv2d(deconv_output1, 1, 3, 3, 1, 2, name="deconvHidden2",reuse=reuse)

    cnn_output1_mask = gatedConv2d(cnn_input, 1, 8, 3, 3, 1, 1, name = "mask_cnn_layer1", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output1_mask = pool(cnn_output1_mask, 3, 3, 2, 2,name="pool1",reuse=reuse)
    cnn_output2_mask = gatedConv2d(cnn_output1_mask, 8, 8, 3, 3, 1, 1, name = "mask_cnn_layer2", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output2_mask = pool(cnn_output2_mask, 3, 3, 2, 2,name="pool2",reuse=reuse)
    cnn_output3_mask = gatedConv2d(cnn_output2_mask, 8, 8, 3, 3, 1, 1, name = "mask_cnn_layer3", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    deconv_mask_output1=deconv2d(cnn_output3_mask, 8, 3, 3, 2, 2, name="deconvHidden_mask1",reuse=reuse)
    deconv_maks_output2=deconv2d(deconv_mask_output1, 8, 3, 3, 2, 2, name="deconvHidden_mask2",reuse=reuse)
    
    cnn_output4_mask = conv2d_withoutact(deconv_maks_output2, 1, 3, 3, 1, 1, name = "mask_cnn_layer4", is_pooling=False, bn=True, is_train=is_training, reuse=reuse)
    cnn_output4_mask = tf.sigmoid(cnn_output4_mask)
    deconv_output2 = tf.multiply(deconv_output2, cnn_output4_mask)

    return deconv_output2


def build_train_graph(tf_lst):

    # start building tf graph
    print("Building train graph ...")

    lr = tf.placeholder(tf.float32)

    # read batch feature
    mix_feature,mix_batch, mix_batch_angle, s1_batch, s1_batch_angle, weights_batch = input_pipeline(filenames=tf_lst, read_threads=READ_THREAD_NUM, batch_size = FEA_TRAIN_BATCH_SIZE*GPU_DEVICE_NUM)

    # frm_num_sum = tf.cast(tf.shape(example_batch)[0] * tf.shape(example_batch)[1], tf.float32)
    pad = tf.slice(tf.zeros_like(weights_batch),[0,0,0],[FEA_TRAIN_BATCH_SIZE*GPU_DEVICE_NUM,-1,3])
    mix_fe = tf.concat([mix_feature, pad],axis=2)
    weights_batch = tf.concat([weights_batch, pad],axis=2)
    # Split the batch of images and labels for towers.
    mix_fe = tf.split(value=mix_fe, num_or_size_splits=GPU_DEVICE_NUM, axis=0)
    mix_splits = tf.split(value=mix_batch, num_or_size_splits=GPU_DEVICE_NUM, axis=0)
    mix_splits_angle = tf.split(value=mix_batch_angle, num_or_size_splits=GPU_DEVICE_NUM, axis=0)
    s1_splits = tf.split(value=s1_batch, num_or_size_splits=GPU_DEVICE_NUM, axis=0)
    s1_splits_angle = tf.split(value=s1_batch_angle, num_or_size_splits=GPU_DEVICE_NUM, axis=0)
    
    weights_batch = pool(tf.expand_dims(weights_batch,-1), 1, 2, 1, 2)
    weights_batch = pool(weights_batch, 1, 2, 1, 2)
    weights_batch = tf.equal(weights_batch, 0)
    weights_batch = attention_bias_ignore_padding(tf.squeeze(weights_batch,[-1]))
    weights_splits = tf.split(value=weights_batch, num_or_size_splits=GPU_DEVICE_NUM, axis=0)

    # optimizer
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.MomentumOptimizer(lr, MOMENTUM)
    # optimizer = tf.train.AdagradOptimizer(lr)
    # optimizer = tf.train.AdadeltaOptimizer(lr)
    optimizer = tf.train.AdamOptimizer()
    # build Dnn
    tower_grads = []
    
    loss_sum = tf.convert_to_tensor(0.0)

    for i in xrange(GPU_DEVICE_NUM):
        with tf.device('/gpu:%d' % i):
            if i == 0:
                logits1 = build_CLDNN_model_graph_train(mix_fe[i],weights_splits[i], True, reuse=False)
            else:
                logits1 = build_CLDNN_model_graph_train(mix_fe[i],weights_splits[i], True, reuse=True)
            
            # compute loss

            #logits1 = tf.reshape(logits1, [FEA_TRAIN_BATCH_SIZE, -1, trn_fea_cfg.fea_dim])
            logits1 = tf.slice(tf.reshape(logits1, [FEA_TRAIN_BATCH_SIZE, -1, trn_fea_cfg.fea_dim+3]),[0,0,0],[FEA_TRAIN_BATCH_SIZE,-1,trn_fea_cfg.fea_dim])
                     
            cleaned_1 = tf.multiply(logits1, mix_splits[i])
            
            # Compute loss(Mse)
            cost_pha = tf.reduce_mean( tf.reduce_sum(tf.pow(cleaned_1-s1_splits[i]*tf.cos(mix_splits_angle[i]-s1_splits_angle[i]),2),1),1) 
            cost_am = tf.reduce_mean( tf.reduce_sum(tf.pow(cleaned_1-s1_splits[i],2),1),1)  
 
            loss_mse = tf.reduce_sum(cost_pha)
            cost_pha_sum=tf.reduce_sum(cost_pha)
            cost_am_sum=tf.reduce_sum(cost_am)
            # Compute Gradients
            t_vars = tf.trainable_variables()

            # alternative
            grads = []
            for t in tf.gradients(loss_mse, t_vars):
              print(t.name)
              if 'blstm_layers' in t.name:
                grads.append(tf.clip_by_value(t, -MAX_GRAD_NROM, MAX_GRAD_NROM))
                #grads.append(tf.clip_by_norm(t, self.max_grad_norm, axes=0))
              else:
                grads.append(t)

            grad=zip(grads, t_vars)
            #grad = optimizer.compute_gradients(cross_entropy)
            #print(cross_entropy)
            tower_grads.append(grad)
            # Compute Loss
            #loss = tf.reduce_sum(cross_entropy, name='xentropy_mean')
            loss_sum += loss_mse

    loss_sum = loss_sum / FEA_TRAIN_BATCH_SIZE / GPU_DEVICE_NUM

    tf.summary.scalar('loss', loss_sum)

    with tf.device('/gpu:0'):
        # merge gradients, sum and average
        print("merge gradients ...")
        #print(tower_grads)
        grads_and_vars = _average_gradients(tower_grads)

        print("minimize loss ...")
        optm = optimizer.apply_gradients(grads_and_vars)

    return loss_sum, cost_pha_sum, cost_am_sum, optm, lr
    
def build_test_graph(tf_lst):

    # start building tf graph
    print("Building test graph ...")

    filename_queue = tf.train.string_input_producer(string_tensor=tf_lst, num_epochs=None, shuffle=None)

    feature, mix, mix_angle, s1, s1_angle, weights  = read_my_file_format(filename_queue=filename_queue)
   
    print mix
    utt_length = tf.shape(mix)[0]
    
    feature = tf.reshape(feature,[1, -1, trn_fea_cfg.fea_dim])
    mix = tf.reshape(mix,[1, -1, trn_fea_cfg.fea_dim])
    mix_angle = tf.reshape(mix_angle,[1, -1, trn_fea_cfg.fea_dim])
    s1 =  tf.reshape(s1,[1, -1, trn_fea_cfg.fea_dim])
    s1_angle = tf.reshape(s1_angle,[1, -1, trn_fea_cfg.fea_dim])
    
    weights = tf.reshape(weights,[1, -1,trn_fea_cfg.fea_dim])
    pad = tf.slice(tf.zeros_like(weights),[0,0,0],[1,-1,3])

    weights = tf.concat([weights, pad],axis=2)
    weights = pool(tf.expand_dims(weights,-1), 1, 2, 1, 2)
    weights = pool(weights, 1, 2, 1, 2)
    weights = tf.equal(weights, 0)
    weights = attention_bias_ignore_padding(tf.squeeze(weights,[-1]))
    
    mix_fe = tf.concat([feature, pad],axis=2)

    frm_num_sum = tf.shape(mix)[0]
    loss_sum = tf.convert_to_tensor(0.0)
    cv_sdr_sum = tf.convert_to_tensor(0.0)
    
    with tf.device('/gpu:0'):
        logits1 = build_CLDNN_model_graph_test(mix_fe,weights,False, reuse=True)
        logits1 = tf.slice(tf.reshape(logits1, [1, -1, trn_fea_cfg.fea_dim+3]),[0,0,0],[1,-1,trn_fea_cfg.fea_dim])

        # compute loss
        cleaned_1 = tf.multiply(logits1, mix)

        # Compute loss(Mse)
        cost_pha = tf.reduce_mean( tf.reduce_sum(tf.pow(cleaned_1-s1*tf.cos(mix_angle-s1_angle),2),1) ,1) 
        cost_am = tf.reduce_mean( tf.reduce_sum(tf.pow(cleaned_1-s1,2),1) ,1) 

        loss_mse = tf.reduce_sum(cost_pha)

        loss_sum += loss_mse

    return loss_sum, cost_pha, cost_am, frm_num_sum


def run_train(sess, train_graph, learning_rate):

    trn_loss_sum, loss_pha, loss_am, trn_optm,trn_lr = train_graph

    trn_step_count1 = trn_fea_cfg.utt_num/FEA_TRAIN_BATCH_SIZE / GPU_DEVICE_NUM 
    trn_step_count2 = trn_step_count1 / 10
    if trn_step_count2 > 1000:
        trn_step_count2 = 1000

    trn_loss = 0
    trn_loss_step = 0

    trn_step = 0
    trn_mse1 = 0
    trn_mse2 = 0

    print 'Do Training'
    try:
        for trn_step in xrange(trn_step_count1):
            _, lo, lossmse_pha, lossmse_am = sess.run([trn_optm, trn_loss_sum, loss_pha, loss_am], feed_dict={trn_lr: learning_rate})
            trn_loss_step += lo
            trn_loss += lo

            trn_mse1 += lossmse_pha
            trn_mse2 += lossmse_am

            if ((trn_step + 1) % trn_step_count2 == 0):
                print '%d / %d: loss %0.5f; pha loss %0.5f, am loss %0.5f' % (trn_step + 1, trn_step_count1, trn_loss_step / trn_step_count2,trn_mse1 / trn_step_count2, trn_mse2 / trn_step_count2)
                trn_loss_step = 0
                trn_mse1 = 0
                trn_mse2 = 0

    except tf.errors.OutOfRangeError:
        print('Done Training -- epoch limit reached')

    trn_loss = trn_loss/trn_step_count1
    print 'loss %0.5f;' % (trn_loss)
    
    return trn_loss

def run_cv(sess, test_graph):

    cv_loss_sum, loss_mse_1, loss_mse_2, cv_frm_num_sum= test_graph
    
    cv_frm_num = 0
    cv_loss = 0
    cv_loss_step = 0
    cv_loss_mse_1 = 0 
    cv_loss_mse_2 = 0

    print 'Do Testing'
    try:
        for cv_step in xrange(cv_fea_cfg.utt_num):
            lo, l1, l2, fn= sess.run([cv_loss_sum, loss_mse_1, loss_mse_2, cv_frm_num_sum])
            cv_loss += lo
            cv_frm_num += fn
            cv_loss_step += lo
       
            if (cv_step + 1) % 1000 == 0:
                print '  %d / %d: loss %0.5f,' % (cv_step + 1, cv_fea_cfg.utt_num, cv_loss_step/1000)
                cv_loss_step = 0

    except tf.errors.OutOfRangeError:
        print('Done Testing -- epoch limit reached')

    cv_loss = cv_loss / cv_fea_cfg.utt_num
    print 'loss %0.5f;' % (cv_loss)
    
    return  cv_loss

def run_all():

    #zfile.z_rmdir(param.moddir)
    zfile.z_mkdir(param.moddir)

    #zfile.z_rmdir(param.logdir)
    zfile.z_mkdir(param.logdir)

    trn_fea_cfg.readfrom(param.trndir + '/fea.cfg')
    trn_fea_cfg.printcfg()
    trn_lst = getfilelst(param.trndir + '/tf.lst')

    cv_fea_cfg.readfrom(param.cvdir + '/fea.cfg')
    cv_lst = getfilelst(param.cvdir + '/tf.lst')

    #build graph
    train_graph = build_train_graph(trn_lst)
    test_graph = build_test_graph(cv_lst)

    #learning rate decrease
    lr_decrease = False
    learning_rate = LEARNING_RATE_INIT

    #model saver
    saver = tf.train.Saver(max_to_keep=50)
    cv_loss_last = 1000000000.0
    tfmodel_path_last = ""

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if len(param.initmod) > 0:
            loadModel(sess, saver, param.initmod)

            cv_loss = run_cv(sess=sess, test_graph=test_graph)
            tfmodel_path_last = param.initmod
            cv_loss_last = cv_loss
            print  'InitModel %s with cv %0.4f' % (tfmodel_path_last, cv_loss)

        for step in xrange(param.iter):

            print 'Iter %d Started(lr=%f):' % (step, learning_rate)

            trn_loss = run_train(sess=sess, train_graph=train_graph, learning_rate=learning_rate)
            cv_loss = run_cv(sess=sess, test_graph=test_graph)

            model_path = "%s/iter%d_lr%f_trn_%0.4f_cv%0.4fmod" % (param.moddir, step, learning_rate, trn_loss, cv_loss)           
            print 'Iter %d Ended(lr=%f, trn=%0.4f, cv=%0.4f ):' % (step, learning_rate, trn_loss, cv_loss)

            
            if cv_loss < cv_loss_last:
                cv_loss_last = cv_loss
                tfmodel_path_last = model_path
                saveModel(sess, saver, model_path)
                    
        coord.request_stop()
        coord.join(threads)

    os.system("ln -s %s %s/final.mod"%(tfmodel_path_last,param.moddir))

    return

if __name__ == '__main__':
    parser.add_argument('--iter', type=int, default=500, help='Hidden Layer Dim')
    parser.add_argument('--initmod', type=str, default='', help='model init')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM Hidden Layer Num')
    parser.add_argument('--project_size', type=int, default=256, help='LSTM Projection Dim')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden Layer Dim')
    parser.add_argument('--atten_hidden_units', type=int, default=8, help='Hidden Layer Dim')
    parser.add_argument('--atten_num_heads', type=int, default=4, help='Hidden Layer Dim')
    parser.add_argument('--atten_dropout_rate', type=int, default=0.5, help='Hidden Layer Dim')
    parser.add_argument('--residual_dropout_rate', type=int, default=0.5, help='Hidden Layer Dim')
    parser.add_argument('--trndir', type=str, default='se_tr_0124', help='Train Tf Folder')
    parser.add_argument('--cvdir', type=str, default='se_tt_0124', help='Test Tf Folder')
    parser.add_argument('--moddir', type=str, default='', help='model dir')
    parser.add_argument('--logdir', type=str, default='', help='log dir')

    param, _ = parser.parse_known_args()
    print param

    run_all()

