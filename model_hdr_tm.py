# [2017-09] HDR Imaging: Shangzhe Wu
#   + License: MIT

import os
import time
from glob import glob
import tensorflow as tf
from random import shuffle
from tensorflow.python import pywrap_tensorflow
from ops import *
from utils import *
from load_data import load_data

import pdb

def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
    return varlist

# total number of examples = 19094*8 = 152752
#TTL_NUM_EX = 153320
TTL_NUM_EX = 5511*8

########## Additional Helper Functions ##########
GAMMA = 2.2 # LDR&HDR domain transform parameter
def LDR2HDR(img, expo): # input/output -1~1
    return (((img+1)/2.)**GAMMA / expo) *2.-1

def HDR2LDR(img, expo): # input/output -1~1
    return (tf.clip_by_value(((img+1)/2.*expo),0,1)**(1/GAMMA)) *2.-1

MU = 5000. # tunemapping parameter
def tonemap(images): # input/output -1~1
    return tf.log(1 + MU * (images+1)/2.) / tf.log(1 + MU) *2. -1

def tonemap_np(images): # input/output -1~1
    return np.log(1 + MU * (images+1)/2.) / np.log(1 + MU) *2. -1

# load testing data
##################################################
# Caution: returned LDRs are sorted by L-M-H, RGB
##################################################
def get_input(scene_dir, image_size):
    c = 3
    in_LDR_paths = sorted(glob(os.path.join(scene_dir, '*aligned.tif')))
    ns = len(in_LDR_paths)

    in_exps_path = os.path.join(scene_dir, 'input_exp.txt')
    print(in_exps_path)
    in_exps = np.array(open(in_exps_path).read().split('\n')[:ns]).astype(np.float32)
    in_exps -= in_exps.min()
    in_LDRs = np.zeros(image_size + [c*ns], dtype=np.float32)
    in_HDRs = np.zeros(image_size + [c*ns], dtype=np.float32)
    
    for i, image_path in enumerate(in_LDR_paths):
        print(i)
        img = get_image(image_path, image_size=image_size, is_crop=True)
        in_LDRs[:,:,c*i:c*(i+1)] = img
        in_HDRs[:,:,c*i:c*(i+1)] = LDR2HDR(img, 2.**in_exps[i])
        
    ref_HDR = get_image(os.path.join(scene_dir, 'ref_hdr_aligned.hdr'), image_size=image_size, is_crop=True)
    print(os.path.join(scene_dir, 'tone_mapped_aligned.tif'))
    ref_TM = get_image(os.path.join(scene_dir, 'tone_mapped_aligned_.tif'), image_size = image_size, is_crop=True)
    return in_LDRs, in_HDRs, ref_HDR, ref_TM
    #return in_LDRs, in_HDRs

# save sample results
def save_results(imgs, out_path):
    imgC = 3
    batchSz, imgH, imgW, c_ = imgs[0].shape
    assert (c_ % imgC == 0)
    ns = c_ // imgC

    nRows = np.ceil(batchSz/4)
    nCols = min(4, batchSz) #4

    res_imgs = np.zeros((batchSz*len(imgs)*ns, imgH, imgW, imgC))
    # rearranging the images, this is a bit complicated
    for n, img in enumerate(imgs):
        for i in range(batchSz):
            for j in range(ns):
                idx = ((i//nCols)*len(imgs)+n)*ns*nCols + (i%nCols)*ns + j
                res_imgs[idx,:,:,:] = img[i,:,:,j*imgC:(j+1)*imgC]
    save_images(res_imgs, [nRows*len(imgs), nCols*ns], out_path)
########## Additional Helper Functions ##########

######U-Net upsampling and downsampling layers ########
def merge_and_conv2d(in1, in2, depth, kernel, name, strides=(1, 1), padding="SAME"):
    merge1 = tf.concat([in1, in2], axis=3, name=name+'concat')
    conv1 = tf.layers.conv2d(merge1, filters=depth, kernel_size=kernel, strides=(1,1), padding="same", activation=tf.nn.relu, name = name+'mc1')
    conv1 = tf.layers.dropout(conv1, 0.2)
    conv1 = tf.layers.conv2d(conv1, filters=depth, kernel_size=kernel, strides=(1,1), padding="same", activation=tf.nn.relu, name = name + 'mc2')
    return conv1

def downsample_2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    conv1 = tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=(1,1), padding="same", activation=tf.nn.relu, name=name+'c1')
    conv1 = tf.layers.dropout(conv1, 0.2)
    conv1 = tf.layers.conv2d(conv1, filters=depth, kernel_size=kernel, strides=(1,1), padding="same", activation=tf.nn.relu, name=name+'c2')
    #pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=strides, padding=padding, name=name+'p1')
    return conv1
################################################################################################

    
class model(object):
    def __init__(self, sess, config, train=True):
        self.sess = sess
        self.batch_size = config.batch_size
        if train:
            self.image_size = config.fine_size
        else:
            self.test_h = config.test_h
            self.test_w = config.test_w
            
        self.c_dim = config.c_dim
        self.num_shots = config.num_shots
        
        self.sample_freq = int(10*64/config.batch_size)
        #self.sample_freq = 2
        if config.save_freq == 0:
            self.save_freq = int(10*64/config.batch_size)
        else:
            self.save_freq = config.save_freq

        self.gf_dim = 64        
        self.num_res_blocks = 9

        self.build_model(config=config, train=train)
        self.model_name = "model"
        self.help = 24.99

        
    def build_model(self, config, train):
        
        if train:
            tfrecord_list = glob(os.path.join(config.dataset, '**', '*.tfrecords'), recursive=True)
            assert (tfrecord_list)
            shuffle(tfrecord_list)
            print('\n\n====================\ntfrecords list:')
            [print(f) for f in tfrecord_list]
            print('====================\n\n')
            
            with tf.device('/cpu:0'):
                filename_queue = tf.train.string_input_producer(tfrecord_list)
                self.in_LDRs, self.in_HDRs, self.ref_LDRs, self.ref_HDR, self.ref_TM, _, _ = load_data(filename_queue, config)

            self.G_HDR ,self.G_tonemapped = self.generator(self.in_LDRs,self.in_HDRs, train=train)
            self.G_sum = tf.summary.image("G", self.G_tonemapped) 
            
            # l2 loss
            #self.g_loss = tf.reduce_mean((self.G_tonemapped - tonemap(self.ref_HDR))**2) # after tonemapping
            #self.g_loss = tf.reduce_mean((self.G_tonemapped - self.ref_TM)**2)
            self.g_loss = tf.reduce_mean(tf.squared_difference(self.G_tonemapped, self.ref_TM))
            self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
            
            t_vars = tf.trainable_variables()
            self.g_vars = [var for var in t_vars if 'g_' in var.name]

            with tf.device('/cpu:0'):
                sample_tfrecord_list = glob(os.path.join(
                    './dataset/tf_records', '**', '*.tfrecords'), recursive=True)
                shuffle(sample_tfrecord_list)
                filename_queue_sample = tf.train.string_input_producer(sample_tfrecord_list)
                self.in_LDRs_sample, self.in_HDRs_sample, self.ref_LDRs_sample, self.ref_HDR_sample, self.ref_TM_sample, _, _ = \
                    load_data(filename_queue_sample, config)
            
            self.sampler_HDR, self.sampler_tonemapped = self.generator(self.in_LDRs_sample, self.in_HDRs_sample, train=False, reuse = True)
            #self.sampler_tonemapped = tonemap(self.sampler_HDR)
            

        # testing
        else:
            self.in_LDRs_sample = tf.placeholder(
                tf.float32, [self.batch_size, config.test_h, config.test_w, self.c_dim*self.num_shots], name='input_LDR_sample')
            self.in_HDRs_sample = tf.placeholder(
                tf.float32, [self.batch_size, config.test_h, config.test_w, self.c_dim*self.num_shots], name='input_HDR_sample')
            
            self.sampler_HDR, self.sampler_tonemapped = self.generator(self.in_LDRs_sample, self.in_HDRs_sample, train=False, free_size=True)
           

        varlist=print_tensors_in_checkpoint_file(file_name= 'pretrained_ckp/pretrained',all_tensors=True,tensor_name=None)
        #print(varlist)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver1 = tf.train.Saver(variables[:len(varlist)])
        self.saver = tf.train.Saver(max_to_keep=50)
        
        
    def train(self, config):
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        
        counter = 1
        start_time = time.time()
        self.load(config.checkpoint_dir, True)

        #if self.load(config.checkpoint_dir):
         #   print("An existing model was found in the checkpoint directory.")
        #else:
         #   print("An existing model was not found in the checkpoint directory. Initializing a new one."
        print(TTL_NUM_EX)
        print(self.batch_size)
        for epoch in range(config.epoch):
            batch_idxs = TTL_NUM_EX // self.batch_size

            for idx in range(0, batch_idxs):
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss])
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d], time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, time.time() - start_time, errG))
                
                if np.mod(counter, self.sample_freq) == 1:
                #if np.mod(counter, 100) == 1:
                    in_LDRs_samples, ref_HDR_samples, ref_TM_samples, hdr_samples, res_samples = \
                        self.sess.run([self.in_LDRs_sample, self.ref_HDR_sample, self.ref_TM_sample, self.sampler_HDR,self.sampler_tonemapped])
                    
            
                    #cv2.imwrite("img.png", (hdr_samples[0,:,:,:]+1)*12700.5)
                    save_results([res_samples, ref_TM_samples],
                                 os.path.join(config.sample_dir, 'train_{:02d}_{:04d}_TM.png'.format(epoch, idx)))
                    save_results([tonemap_np(hdr_samples), tonemap_np(ref_HDR_samples)],
                                 os.path.join(config.sample_dir, 'train_{:02d}_{:04d}_HDR.png'.format(epoch, idx)))
                    save_results([in_LDRs_samples],
                                 os.path.join(config.sample_dir, 'train_{:02d}_{:04d}_LDRs.png'.format(epoch, idx)))

                if np.mod(counter, self.save_freq) == 2:
                    self.save(config.checkpoint_dir, counter)

        coord.request_stop()
        coord.join(threads)
    
    
    def test(self, config):
        
        tf.initialize_all_variables().run()

        isLoaded = self.load(config.checkpoint_dir, False)
        assert(isLoaded)
        #saver = tf.train.import_meta_graph(os.path.join(config.checkpoint_dir, 'model-4002.meta'))
        #saver.restore(self.sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        
        scene_dirs = sorted(os.listdir(config.dataset))
        nScenes = len(scene_dirs)
        num_batch = int(np.ceil(nScenes/self.batch_size))
        
        psnr_f = open(os.path.join(config.results_dir, 'psnr.txt'),'w')
        psnr = []
        ssim_f = open(os.path.join(config.results_dir, 'ssim.txt'),'w')
        ssim = []
        for idx in range(0, num_batch):
            print('batch no. %d:' %(idx+1))
            psnr_f.write('batch no. %d:\n' %(idx+1))
            
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nScenes)
            batchSz = u-l
            batch_scene_dirs = scene_dirs[l:u]
            batch_in_LDRs = []
            batch_in_HDRs = []
            batch_ref_HDR = []
            batch_ref_TM = []
            for i, scene_dir in enumerate(batch_scene_dirs):
                _LDRs, _HDRs, _HDR, _TM = get_input(os.path.join(config.dataset, scene_dir), 
                                               [config.test_h, config.test_w])
                #_LDRs, _HDRs = get_input(os.path.join(config.dataset, scene_dir), 
                #                               [config.test_h, config.test_w])
                batch_in_LDRs = batch_in_LDRs + [_LDRs]
                batch_in_HDRs = batch_in_HDRs + [_HDRs]
                batch_ref_HDR = batch_ref_HDR + [_HDR]
                batch_ref_TM = batch_ref_TM + [_TM]
            batch_in_LDRs = np.array(batch_in_LDRs, dtype=np.float32)
            batch_in_HDRs = np.array(batch_in_HDRs, dtype=np.float32)
            batch_ref_HDR = np.array(batch_ref_HDR, dtype=np.float32)
            batch_ref_TM = np.array(batch_ref_TM, dtype=np.float32)
            
            # deal with last batch
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_in_LDRs = np.pad(batch_in_LDRs, padSz, 'wrap').astype(np.float32)
                batch_in_HDRs = np.pad(batch_in_HDRs, padSz, 'wrap').astype(np.float32)
				
            
            st = time.time()
            res_samples, tm_samples = self.sess.run([self.sampler_HDR, self.sampler_tonemapped],
                feed_dict={ self.in_LDRs_sample: batch_in_LDRs, self.in_HDRs_sample: batch_in_HDRs })
            print("time: %.4f" %(time.time()-st))
            
            curr_psnr = [compute_psnr(tonemap_np(res_samples[i]), tonemap_np(batch_ref_HDR[i])) for i in range(batchSz)]
            print("PSNR: %.4f\n" %np.mean(curr_psnr))
            psnr_f.write("PSNR: %.4f\n\n" %np.mean(curr_psnr))
            psnr += curr_psnr
            
            #curr_ssim = [compute_ssim(tm_samples[i], batch_ref_TM[i]) for i in range(batchSz)]
            #print("SSIM: %.4f\n" %np.mean(curr_ssim))
            #ssim_f.write("SSIM: %.4f\n\n" %np.mean(curr_ssim))
            #ssim += curr_ssim
            
            save_results([tm_samples[:batchSz]],
                         os.path.join(config.results_dir, 'test_{:03d}_{:03d}_TM.png'.format(l,u)), True)
            save_results([batch_ref_TM[:batchSz]],
                         os.path.join(config.results_dir, 'test_{:03d}_{:03d}_ref_TM.png'.format(l,u)))
            #save_results([tonemap_np(res_samples[:batchSz])],
             #            os.path.join(config.results_dir, 'test_{:03d}_{:03d}_tonemapped.png'.format(l,u)))
            save_results([batch_in_LDRs],
                         os.path.join(config.results_dir, 'test_{:03d}_{:03d}_LDRs.png'.format(l,u)))
        
        avg_psnr = np.mean(psnr)
        print("Average PSNR: %.4f" %avg_psnr)
        psnr_f.write("Average PSNR: %.4f" %avg_psnr)
        psnr_f.close()
        
        avg_ssim = np.mean(ssim)
        print("Average SSIM: %.4f" %avg_ssim)
        psnr_f.write("Average SSIM: %.4f" %avg_ssim)
        psnr_f.close()
    
    
    def generator(self, in_LDR, in_HDR, train=True, reuse = False, free_size=False):
        with tf.variable_scope("generator", reuse = reuse):
            if free_size:
                s_h, s_w = self.test_h, self.test_w
            else:
                s_h, s_w = self.image_size, self.image_size
            s2_h, s4_h, s8_h, s16_h, s2_w, s4_w, s8_w, s16_w = \
                int(s_h/2), int(s_h/4), int(s_h/8), int(s_h/16), int(s_w/2), int(s_w/4), int(s_w/8), int(s_w/16)
            
            def residule_block(x, dim, ks=3, s=1, train=True, name='res'):
                p = int((ks - 1) / 2)
                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name+'_c1'),
                               train=train, name=name+'_bn1')
                y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name+'_c2'),
                               train=train, name=name+'_bn2')
                return y + x
            
            image1 = tf.concat([tf.slice(in_LDR, [0,0,0,0], [-1,-1,-1,self.c_dim]), 
                                tf.slice(in_HDR, [0,0,0,0], [-1,-1,-1,self.c_dim])], 3)
            image2 = tf.concat([tf.slice(in_LDR, [0,0,0,self.c_dim], [-1,-1,-1,self.c_dim]),
                                tf.slice(in_HDR, [0,0,0,self.c_dim], [-1,-1,-1,self.c_dim])], 3)
            image3 = tf.concat([tf.slice(in_LDR, [0,0,0,self.c_dim*2], [-1,-1,-1,self.c_dim]),
                                tf.slice(in_HDR, [0,0,0,self.c_dim*2], [-1,-1,-1,self.c_dim])], 3)
            
            with tf.variable_scope("encoder1"):
                # image is (256 x 256 x input_c_dim)
                e1_1 = conv2d(image1, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e1_2 = batch_norm(conv2d(lrelu(e1_1), self.gf_dim*2, name='g_e2_conv'), train=train, name='g_e2_bn')
            
            with tf.variable_scope("encoder2"):
                # image is (256 x 256 x input_c_dim)
                e2_1 = conv2d(image2, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e2_2 = batch_norm(conv2d(lrelu(e2_1), self.gf_dim*2, name='g_e2_conv'), train=train, name='g_e2_bn')
            
            with tf.variable_scope("encoder3"):
                # image is (256 x 256 x input_c_dim)
                e3_1 = conv2d(image3, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
                e3_2 = batch_norm(conv2d(lrelu(e3_1), self.gf_dim*2, name='g_e2_conv'), train=train, name='g_e2_bn')
            
            with tf.variable_scope('merger'):
                e_2 = tf.concat([e1_2, e2_2, e3_2], 3)
                # e2 is (64 x 64 x self.gf_dim*2*3)
                e_3 = batch_norm(conv2d(lrelu(e_2), self.gf_dim*4, name='g_e3_conv'), train=train, name='g_e3_bn')
                # e3 is (32 x 32 x self.gf_dim*4)
                
                res_layer = e_3
                for i in range(self.num_res_blocks):
                    #res_layer = batch_norm(conv2d(lrelu(res_layer), self.gf_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, 
                    #                              name='g_e5_conv_%d' %(i+1)), train=train, name='g_e5_bn_%d' %(i+1))
                    res_layer = residule_block(tf.nn.relu(res_layer), self.gf_dim*4, ks=3, train=train, name='g_r%d' %(i+1))

            with tf.variable_scope("decoder"):
                d0 = tf.concat([res_layer, e_3], 3)
                # d0 is (32 x 32 x self.gf_dim*4*2)
                
                d1 = batch_norm(conv2d_transpose(tf.nn.relu(d0), 
                    [self.batch_size, s4_h, s4_w, self.gf_dim*2], name='g_d1'), train=train, name='g_d1_bn')
                d1 = tf.concat([d1, e1_2, e2_2, e3_2], 3)
                # d1 is (64 x 64 x self.gf_dim*2*4)

                d2 = batch_norm(conv2d_transpose(tf.nn.relu(d1), 
                    [self.batch_size, s2_h, s2_w, self.gf_dim], name='g_d2'), train=train, name='g_d2_bn')
                d2 = tf.concat([d2, e1_1, e2_1, e3_1], 3)
                # d2 is (128 x 128 x self.gf_dim*1*4)

                d3 = batch_norm(conv2d_transpose(tf.nn.relu(d2), 
                    [self.batch_size, s_h, s_w, self.gf_dim], name='g_d3'), train=train, name='g_d3_bn')
                # d3 is (256 x 256 x self.gf_dim)
                
                out = tf.nn.tanh(conv2d(tf.nn.relu(d3), self.c_dim, d_h=1, d_w=1, name='g_d_out_conv'))

                #return tf.nn.tanh(out)
                with tf.variable_scope("unet"):
                    out1 = downsample_2d(out, 32, 3, 'g_layer1', (2, 2))
                    pool1 = tf.layers.average_pooling2d(out1, pool_size=2, strides=(2,2), name='g_pool1')
                    out2 = downsample_2d(pool1, 64, 3, 'g_layer2', (2, 2))
                    pool2 = tf.layers.average_pooling2d(out2, pool_size=2, strides=(2,2), name='g_pool2')
                    out3 = downsample_2d(pool2, 128, 3, 'g_layer3', (2, 2))
                    out3_upsample = tf.layers.conv2d_transpose(out3, 64, 2, 2, padding='valid', name='g_upsample1')
                    out4 = merge_and_conv2d(out3_upsample, out2, 64, 3, 'g_layer4')
                    out4_upsample = tf.layers.conv2d_transpose(out4, 32, 2, 2, padding='valid', name='g_upsample2')
                    out5 = merge_and_conv2d(out4_upsample, out1, 32, 3, 'g_layer5')
                    out6 = tf.layers.conv2d(out5, filters=3, kernel_size=(1,1), strides=(1,1), padding="same", activation=tf.nn.tanh, name='g_final')
                    return [out, out6]      	
            	    	
    
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

        
    def load(self, checkpoint_dir, train):
        print(" [*] Reading checkpoints...")

        if train:
            self.saver1.restore(self.sess, 'pretrained_ckp/pretrained')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
