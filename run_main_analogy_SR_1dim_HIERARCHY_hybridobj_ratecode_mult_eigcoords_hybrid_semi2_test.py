import tensorflow as tf
import numpy as np
import os
import classic_AE_inv_test_norm_gen_SR_hybridobj_onelevel_onedim_HIERARCHY_mult_eigcoords_fixed_hybrid_sem2_sepfamilies
import plot_utils_analogy
import glob
import pdb
import argparse
import scipy as scipy
import shutil
from tempfile import TemporaryFile
from scipy.io import loadmat
from PIL import Image
import scipy.stats as stats
import sklearn

outfile = TemporaryFile()
IMAGE_SIZE_MNIST = 5
"""parsing and configuration"""
def parse_args():
    desc = "an auto-encoder for relation learning'"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--results_path', type=str, default='results_fixed_1x56_HIERARCHY_full_harder_eigcoords_b_rolepop_sepfamilies2_100test_3eig_1_latent_100decoder_1000samp_0trials_50pctkeepprob')
        #help='File path of output images')
    parser.add_argument('--transform', type=bool, default=False, help='use a transformation as the recon target?')
    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')
    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of first latent vector', required = True)
    parser.add_argument('--dim_r', type=int, default='10', help='Dimension of second latent vector', required = True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    #parser.add_argument('--learn_rate', type=float, default=3e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--learn_rate', type=float, default=3e-4, help='Learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--PRR', type=bool, default=False,
                        help='Boolean for plot-reproduce-result')
    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')
    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')
    parser.add_argument('--PRR_resize_factor', type=float, default=10.0,
                        help='Resize factor for each displayed image')
    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')
    parser.add_argument('--PMLR_n_img_x', type=int, default=5,
                        help='Number of images along x-axis')
    parser.add_argument('--PMLR_n_img_y', type=int, default=5,
                        help='Number of images along y-axis')
    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')
    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')
    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*.jpg*')
    #pdb.set_trace()
    for f in files:
        os.remove(f)
    remaining_dir = glob.glob(args.results_path+'/*')
    for r in remaining_dir:
        shutil.rmtree(r)
    
    os.mkdir(args.results_path +'/images')
    os.mkdir(args.results_path +'/images/train')

    #"/PMLR_epoch_r_%02d_%02d" % (epoch, cur_r) + ".jpg"
    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args

def main(args):
    """ parameters """
    n_samples = 500
    n_TestSamples =500
    n_dimensions=1
    n_values = 56
    RESULTS_DIR = args.results_path
    # network architecture
    ADD_NOISE = False
    n_hidden = args.n_hidden
    dim_img = n_dimensions*n_values  # number of pixels for a MNIST image
    dim_z = args.dim_z
    dim_r = args.dim_r
    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    # Plot
    PRR = args.PRR                             # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space
    results = np.zeros([20000,9])
    byeig = np.zeros([20000,n_values])
    #eig_contents = loadmat('/users/stevenmf/Desktop/prespec_eig.mat')
    #eig_v = eig_contents['v']
    #use backprop to recostruct the SR representation of the output.
    #pdb.set_trace()
    SR_load = loadmat('/users/stevenmf/Desktop/transitions/1x56_familytree_symmetric_unequalgen_twofam.mat')
    SR = SR_load['M']
    labels = SR_load['labels']
    #eigvecs = SR_load['b']
    #eigcoords = np.dot(SR_load['U'],SR_load['V'])
    #eigcoords = SR_load['U']
    #pdb.set_trace()
    #eigs=np.zeros([20,2])
    #eigcoords = SR_load['Uint']
    #eigs[:,0]=eigcoords
    eigcoords = stats.zscore(np.dot(SR_load['U'],SR_load['V']),0)
    #place code
    #eigcoords = np.eye(20)
    #eigs=eigcoords
    #eigs = eigcoords[:,0:4]
    eigcoords[:,0]=eigcoords[:,1]
    eigs = eigcoords[:,0:3]
    #pdb.set_trace()
    np.random.shuffle(eigs)
    #eigs[:,0] = eigcoords[:,1]
    #eigs[:,1]=eigs[:,1]
    #eigs[:,2]=eigcoords[:,1]
    #eigs = eigvecs[:,0:2]
    #eigs=eigvecs
    #eigs = SR[:,0:6]
    #eigvecs=SR_load['eigs']
    #pdb.set_trace()
    ##build graph
    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    dim_img=3
    byeig = np.zeros([args.num_epochs,dim_img])
    x_trans = tf.placeholder(tf.float32, shape=[None, dim_img], name='trans_img')
    #x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    c_mag = tf.placeholder(tf.float32, shape=[None, dim_img], name='source_img2')
    d_mag = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img2')
    #d = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    diff = tf.placeholder(tf.float32, shape=[None, dim_img], name='diff_img')
    # dropout
    direction = tf.placeholder(tf.float32, shape=[None,1], name='direction_label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    eig_vec = tf.placeholder(tf.float32, name='eig_vec')
    SR_diff = tf.placeholder(tf.float32, name='SR_diff')
    #pdb.set_trace()
    # input for PMLR. modified here to deal w/ non-2d cases.
    z_in = tf.placeholder(tf.float32, shape=[PMLR_n_img_x, dim_z], name='latent_variable1')
    r_in = tf.placeholder(tf.float32, shape=[PMLR_n_img_x, dim_r], name='latent_variable2')
    # network architecture
    #y, z, loss, neg_marginal_likelihood, KL_divergence = vae_analogy.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)
    y_xA, y_xB, r, prediction_mag, loss, analogy_MSE, MSE = classic_AE_inv_test_norm_gen_SR_hybridobj_onelevel_onedim_HIERARCHY_mult_eigcoords_fixed_hybrid_sem2_sepfamilies.onelevel_autoencoder(x,x_trans, c_mag,d_mag,direction,dim_img, dim_z, dim_r, n_hidden, keep_prob)
    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    #train_op = tf.train.MomentumOptimizer(learn_rate, 0.5).minimize(loss)
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99
    #[train_dataA, train_dataB, train_dataC, train_dataD, source_identity, target_identity, source_magnitude, target_magnitude, source_direction, target_direction, foils, avoid, SR_diff] = classic_AE_inv_test_norm_withindim_gen.make_analogy_data(n_dimensions, n_values, n_samples, n_TestSamples, SR)
    #pdb.set_trace()
    [train_dataA, train_dataB, train_dataC, train_dataD, test_dataA, test_dataB, test_dataC, test_dataD, foils, train_place_code] = classic_AE_inv_test_norm_gen_SR_hybridobj_onelevel_onedim_HIERARCHY_mult_eigcoords_fixed_hybrid_sem2_sepfamilies.make_analogy_data(n_dimensions, n_values, n_samples, n_TestSamples, eigs, labels)

    alltrain = np.concatenate([train_dataA, train_dataB, train_dataC, train_dataD])
    alltest = np.concatenate([test_dataA, test_dataB, test_dataC, test_dataD])
    ##normalize?
    # train_dataA = (train_dataA-min(alltrain))/(max(alltrain)-min(alltrain))*100
    # train_dataB = (train_dataB-min(alltrain))/(max(alltrain)-min(alltrain))*100
    # train_dataC = (train_dataC-min(alltrain))/(max(alltrain)-min(alltrain))*100
    # train_dataD = (train_dataD-min(alltrain))/(max(alltrain)-min(alltrain))*100
    # test_dataA = (test_dataA-min(alltest))/(max(alltest)-min(alltest))*100
    # test_dataB = (test_dataB-min(alltest))/(max(alltest)-min(alltest))*100
    # test_dataC = (test_dataC-min(alltest))/(max(alltest)-min(alltest))*100
    # test_dataD = (test_dataD-min(alltest))/(max(alltest)-min(alltest))*100
    # foils = (foils-min(alltest))/(max(alltest)-min(alltest))*100
    # train_dataA = (train_dataA*100)+36
    # train_dataB = (train_dataB*100)+36
    # train_dataC = (train_dataC*100)+36
    # train_dataD = (train_dataD*100)+36
    # test_dataA =  (test_dataA*100)+36
    # test_dataB = (test_dataB*100)+36
    # test_dataC = (test_dataC*100)+36
    # test_dataD = (test_dataD*100)+36
    # foils = (foils*100)+36

    train_dataA = (train_dataA*100)
    train_dataB = (train_dataB*100)
    train_dataC = (train_dataC*100)
    train_dataD = (train_dataD*100)
    test_dataA =  (test_dataA*100)
    test_dataB = (test_dataB*100)
    test_dataC = (test_dataC*100)
    test_dataD = (test_dataD*100)
    foils = (foils*100)
    # #pdb.set_trace()
    # train

    if PRR:
        PRR = plot_utils_analogy.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, n_dimensions, dim_img, PRR_resize_factor)
        x_PRR_input_trainA = train_dataA[0:PRR.n_tot_imgs, :]
        x_PRR_img_trainA = x_PRR_input_trainA.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_input_trainB = train_dataB[0:PRR.n_tot_imgs, :]
        x_PRR_img_trainB = x_PRR_input_trainB.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_input = test_dataA[0:PRR.n_tot_imgs, :]
        x_PRR_testA_img = x_PRR_input.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_target = test_dataB[0:PRR.n_tot_imgs, :]
        x_PRR_testB_img = x_PRR_target.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_source_mag_test = test_dataC[0:PRR.n_tot_imgs, :]
        x_PRR_testC_img = x_PRR_source_mag_test.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_target_mag_test = test_dataD[0:PRR.n_tot_imgs, :]
        x_PRR_testD_img = x_PRR_target_mag_test.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        #pdb.set_trace()
        x_PRR_foil1 = foils[0:PRR.n_tot_imgs, 0, :]
        x_PRR_foil1_img = x_PRR_foil1.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)

        x_PRR_foil2 = foils[0:PRR.n_tot_imgs, 1, :]
        x_PRR_foil2_img = x_PRR_foil2.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)
        
        x_PRR_foil3 = foils[0:PRR.n_tot_imgs, 2, :]
        x_PRR_foil3_img = x_PRR_foil3.reshape(PRR.n_tot_imgs,n_dimensions, dim_img)
        
        trans_img_pre = []

        PRR.save_images(x_PRR_img_trainA, name='trainA.jpg')
        PRR.save_images(x_PRR_img_trainB, name='trainB.jpg')
        PRR.save_images(x_PRR_testA_img, name='testA.jpg')
        PRR.save_images(x_PRR_testB_img, name='testB.jpg')
        PRR.save_images(x_PRR_testC_img, name='testC.jpg')
        PRR.save_images(x_PRR_testD_img, name='testD.jpg')
        PRR.save_images(x_PRR_foil1, name='foil1.jpg')
        PRR.save_images(x_PRR_foil2, name='foil2.jpg')
        PRR.save_images(x_PRR_foil3, name='foil3.jpg')
    idx=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 1})
        batchctr=0
        for epoch in range(n_epochs):
            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                ######TRAIN
                #train relations.
                batch_xs_input = train_dataA[offset:(offset + batch_size), :]
                batch_xs_trans = train_dataB[offset:(offset + batch_size), :]
                    #for analogical training.
                batch_train_c_mag = train_dataC[offset:(offset + batch_size), :]
                batch_train_d_mag = train_dataD[offset:(offset + batch_size), :]
                #####TEST
                #test relations
                batch_xs_input_diff= train_place_code[offset:(offset + batch_size), :]
                #pdb.set_trace()

                batch_xs_input_test = test_dataA[offset:(offset + batch_size), :]
                batch_xs_trans_test = test_dataB[offset:(offset + batch_size), :]
                    #for analogical reasoning
                batch_test_c_mag = test_dataC[offset:(offset + batch_size), :]
                batch_test_d_mag = test_dataD[offset:(offset + batch_size), :]

                foil_batch = foils[offset:(offset + batch_size), :]
                #pdb.set_trace()

                _, tot_loss, analogyMSE, firstMSE = sess.run(
                    (train_op, loss, analogy_MSE, MSE),
                    feed_dict={c_mag: batch_train_c_mag, d_mag: batch_train_d_mag, x: batch_xs_input, x_trans: batch_xs_trans,direction: batch_xs_input_diff,keep_prob: 0.5})
                batchctr+=1
            #pdb.set_trace()
                if batchctr%10==0:
                    idx+=1
                    y_xA_test, y_xB_test, prediction_mag_test = sess.run(
                        (y_xA, y_xB, prediction_mag),
                        feed_dict={c_mag: batch_test_c_mag,d_mag: batch_train_d_mag, x: batch_xs_input_test, x_trans: batch_xs_trans_test,direction: batch_xs_input_diff,keep_prob : 1})

                    mag_perf = np.zeros(batch_size)
                    predfoil_perf = np.zeros(batch_size)
                    copy_perf = np.zeros(batch_size)
                    copy_case = np.zeros(batch_size)
                    true_diff = np.zeros(batch_size)
                    true_diff_MSE = np.zeros([batch_size, dim_img])
                    foils_diff_MSE = np.zeros(batch_size)
                    true_diff_cosine = np.zeros(batch_size)
                    foils_diff_cosine = np.zeros(batch_size)
                    r_true_diff = np.zeros(batch_size)
                    a_sign = np.zeros(batch_size)
                    b_sign = np.zeros(batch_size)
                    d_sign = np.zeros(batch_size)
                    parallelogramtest = np.zeros(batch_size)
                        
                        #pdb.set_trace()
                    for d in range(batch_size):
                        #pdb.set_trace()
                        #true_diff[d] = np.mean((prediction_mag_test[d,:]-batch_test_d_mag[d,:])**2)
                        k=np.random.choice(batch_size,3)
                        while any(k==d):
                            k=np.random.choice(batch_size,3)
                            #predfoil1 = np.mean((prediction_mag_test[k,:]-batch_test_d_mag[d,:])**2)
                        predfoil1 =scipy.spatial.distance.cosine(prediction_mag_test[k[0],:], batch_test_d_mag[d,:])
                        predfoil2 =scipy.spatial.distance.cosine(prediction_mag_test[k[1],:], batch_test_d_mag[d,:])
                        predfoil3 =scipy.spatial.distance.cosine(prediction_mag_test[k[2],:], batch_test_d_mag[d,:])

                        true_diff[d]=scipy.spatial.distance.cosine(prediction_mag_test[d,:], batch_test_d_mag[d,:])
                        #true_diff[d]=np.mean((prediction_mag_test[d,:]-batch_test_d_mag[d,:])**2)
                        #pdb.set_trace()
                        #foil1 = np.mean((foil_batch[d,0,:]-batch_test_d_mag[d,:])**2)
                        foil1=scipy.spatial.distance.cosine(foil_batch[d,0,:], batch_test_d_mag[d,:])
                        #foil2 = np.mean((foil_batch[d,1,:]-batch_test_d_mag[d,:])**2)
                        foil2=scipy.spatial.distance.cosine(foil_batch[d,1,:], batch_test_d_mag[d,:])
                        #foil3 = np.mean((foil_batch[d,2,:]-batch_test_d_mag[d,:])**2)
                        foil3=scipy.spatial.distance.cosine(foil_batch[d,2,:], batch_test_d_mag[d,:])
                        copy_case[d]=scipy.spatial.distance.cosine(prediction_mag_test[d,:], batch_test_c_mag[d,:])
                        #copy_case[d] = np.mean((batch_test_c_mag[d,:]-batch_test_d_mag[d,:])**2)
                        #pdb.set_trace()
                        #a_sign[d]=np.mean(np.sign(y_xA_test[d,:]-batch_xs_trans_test[d,:])==np.sign(batch_xs_input_test[d,:]-batch_xs_trans_test[d,:]))
                        #b_sign[d]=np.mean(np.sign(y_xB_test[d,:]-batch_xs_input_test[d,:])==np.sign(batch_xs_trans_test[d,:]-batch_xs_input_test[d,:]))
                        #d_sign[d]=np.mean(np.sign(prediction_mag_test[d,:]-batch_test_c_mag[d,:])==np.sign(batch_test_d_mag[d,:]-batch_test_c_mag[d,:]))
                        #pdb.set_trace()
                        r_true_diff[d] = np.mean(np.mean((y_xA_test[d,:]-batch_xs_input_test[d,:])**2) + np.mean((y_xB_test[d,:]-batch_xs_trans_test[d,:])**2))
                        #foils_diff_MSE[d]=np.mean((foil_batch[d,1,:]-batch_test_d_mag[d,:])**2)
                        true_diff_MSE[d,:]=(prediction_mag_test[d,:]-batch_test_d_mag[d,:])**2
                        
                        foils_diff_cosine[d]=foil1
                        true_diff_cosine[d]=true_diff[d]
                        if (true_diff[d]<foil1) and (true_diff[d]<foil2) and (true_diff[d]<foil3):
                            mag_perf[d]=1
                        
                        if (true_diff[d]<predfoil1):
                            predfoil_perf[d]=1

                        # if (foil1<foil2) and (foil1<foil3):
                        #     mag_perf[d]=1

                        if true_diff[d]<copy_case[d]:
                            copy_perf[d]=1

                        #parallelogram
                        d_parallel = (batch_xs_trans_test[d,:]-batch_xs_input_test[d,:])+batch_test_c_mag[d,:]
                        parallelogramtest[d]=scipy.spatial.distance.cosine(batch_test_d_mag[d,:],d_parallel)

                    #pdb.set_trace()
                    results[idx,0]=epoch
                    results[idx,1]=tot_loss
                    results[idx,2]=np.mean(mag_perf)
                    results[idx,3]=np.mean(true_diff)
                    results[idx,4]=np.mean(copy_perf)
                    results[idx,5]=np.mean(predfoil_perf)
                    results[idx,6]=np.mean(d_sign)
                    results[idx,7]=np.mean(r_true_diff)
                    #results[epoch,8]=np.mean(true_diff_MSE-foils_diff_MSE)
                    byeig[idx,:]=np.mean(true_diff_MSE, axis=0)
                    #print("epoch %d: loss %03.2f: analogy_loss %03.2f: first_loss %03.2f: relation_tot_diff %03.2f: analogy_tot_diff %03.2f: foil_diff %03.2f: copy check %03.2f: predfoil_perf %03.2f analogy_perf %03.2f" % (epoch,tot_loss, analogyMSE, firstMSE, np.mean(r_true_diff),np.mean(true_diff_cosine),np.mean(foils_diff_cosine),np.mean(copy_perf),np.mean(predfoil_perf),np.mean(mag_perf)))
                    #pdb.set_trace()
                    
                    if epoch==0:
                        thesevars = sess.run(tf.get_collection(tf.GraphKeys.VARIABLES))
                        #thisvar = sess.run(tf.get_variable('level1_decoder'))
                        #encodervar = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'relation_encoder_direct'))
                        encodervar = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'relation_encoder_direct'))
                        decodervar = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'level1_decoder'))
                        decoder=decodervar[0]
                        # latent1=latentvar[0]
                        # latent2=latentvar[2]
                        encoder=encodervar[0]
                    #print(prediction_mag_test[d,:]-batch_test_d_mag[d,:])
                    #print(prediction_mag_test[1,:]-batch_test_d_mag[1,:])
                    #print(prediction_mag_test[0,:]-batch_test_d_mag[0,:])
                        # print("encoder %03.2f:" % (encoder))
                        # print("latent %03.2f:" % (latent))
                        # print("decoder %03.2f:" % (decoder))
                        #print("encoder")f
                        #print(encoder)
                        # print("latents")
                        # print(latent1)
                        # print(latent2)
                        #print("decoder")
                        #print(decoder)
                    #print("epoch %d: loss %03.2f: relation_tot_diff %03.2f: analogy_tot_diff %03.2f: foil_diff %03.2f: copy check %03.2f: analogy_perf %03.2f" % (epoch,tot_loss, np.mean(r_true_diff),np.mean(true_diff_cosine),np.mean(foils_diff_cosine),np.mean(copy_perf),np.mean(mag_perf)))
                    np.savetxt('results_unequal_1x56_hierarchy_full_eigcoords_shuffle_sepfamilies2_100test_3eig_1latent_100decoder_0trial_50pctkeepprob.out', results, delimiter=',')
                    #np.savetxt('results_standard_1x80_rate_eigcoords_6eig_6latent_100decoder_1000samp_3trials_50pctkeepprob_weights_l1_100.out', encoder, delimiter=',')
                    #np.savetxt('results_separate_hybrid_1_overcomplete_1000samp_SR_1x80_rate_eigcoords_lowfreq_6_99pct_75keepprob_late_eigs6.out', byeig, delimiter=',')
                    #print(prediction_mag_test[d,:], batch_test_d_mag[d,:])
                    #batchctr+=1
                    # print(batchctr)
                if epoch==0:
                    for j in range(20):
                        #pdb.set_trace()
                        #x_PRR_input_testA = batch_xs_input_test[j, :]          
                        filename = args.results_path + '/images/PRR_%02d_testA_epoch_%02d.jpg' %(j,epoch)
                        testA_img = batch_xs_input_test.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(testA_img[j,:,:]).save(filename)

                        filename = args.results_path + '/images/PRR_%02d_testB_epoch_%02d.jpg' %(j,epoch)
                        testB_img = batch_xs_trans_test.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(testB_img[j,:,:]).save(filename)

                        filename = args.results_path + '/images/PRR_%02d_testC_epoch_%02d.jpg' %(j,epoch)
                        testC_img = batch_test_c_mag.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(testC_img[j,:,:]).save(filename)

                        filename = args.results_path + '/images/PRR_%02d_testD_epoch_%02d.jpg' %(j,epoch)
                        testD_img = batch_test_d_mag.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(testD_img[j,:,:]).save(filename)
                if batchctr%100==0:
                    print("batchctr %d: loss %03.2f: analogy_loss %03.2f: first_loss %03.2f: relation_tot_diff %03.2f: analogy_tot_diff %03.2f: foil_diff %03.2f: copy check %03.2f: predfoil_perf %03.2f analogy_perf %03.2f" % (batchctr,tot_loss, analogyMSE, firstMSE, np.mean(r_true_diff),np.mean(true_diff_cosine),np.mean(foils_diff_cosine),np.mean(copy_perf),np.mean(predfoil_perf),np.mean(mag_perf)))
                    for j in range(20):
                        y_PRR = sess.run(y_xA,feed_dict={c_mag: batch_test_c_mag, d_mag: batch_test_d_mag, x: batch_xs_input_test, x_trans: batch_xs_trans_test,keep_prob : 1})
                        filename = args.results_path + '/images/PRR_%02d_ahat_epoch_%02d.jpg' %(j,epoch)
                        ahat_img = y_PRR.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(ahat_img[j,:,:]).save(filename)

                        y_PRR = sess.run(y_xB,feed_dict={c_mag: batch_test_c_mag, d_mag: batch_test_d_mag, x: batch_xs_input_test, x_trans: batch_xs_trans_test,keep_prob : 1})
                        filename = args.results_path + '/images/PRR_%02d_bhat_epoch_%02d.jpg' %(j,epoch)
                        bhat_img = y_PRR.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(bhat_img[j,:,:]).save(filename)

                        y_PRR = sess.run(prediction_mag,feed_dict={c_mag: batch_test_c_mag, d_mag: batch_test_d_mag, x: batch_xs_input_test, x_trans: batch_xs_trans_test,keep_prob : 1})
                        filename = args.results_path + '/images/PRR_%02d_dhat_epoch_%02d.jpg' %(j,epoch)
                        dhat_img = y_PRR.reshape(batch_size,n_dimensions, dim_img)
                        scipy.misc.toimage(dhat_img[j,:,:]).save(filename)
                    # y_PRR_pred_mag_img = y_PRR.reshape(batch_size, n_dimensions, n_values)
                    # PRR.save_images(y_PRR_pred_mag_img, name="/PRR_trial_%02d_p_mag" %(j) + ".jpg")
                    # filename = args.results_path + '/images/PRR_%02d_dhat_epoch_%02d.jpg' %(j,epoch)
                    # dhat_img = prediction_mag.reshape(batch_size,n_dimensions, n_values)
                    # scipy.misc.toimage(dhat_img[j,:,:]).save(filename)

            #             scipy.misc.toimage(x_PRR_img[j,:,:]).save(filename)
            #             filename = args.results_path + '/images/PRR_%02d_x_trans_epoch_%02d.jpg' %(j,epoch)
            #             scipy.misc.toimage(x_PRR_target_img[j,:,:]).save(filename)
                        
            #             filename = args.results_path + '/images/train/PRR_%02d_A_%02d.jpg' %(j,epoch)
            #             scipy.misc.toimage(x_PRR_img_trainA[j,:,:]).save(filename)
            #             filename = args.results_path + '/images/train/PRR_%02d_B_%02d.jpg' %(j,epoch)
            #             scipy.misc.toimage(x_PRR_img_trainB[j,:,:]).save(filename)
            #             filename = args.results_path + '/images/train/PRR_%02d_mag_source_epoch_%02d.jpg' %(j,epoch)
            #             scipy.misc.toimage(x_PRR_source_mag_img[j,:,:]).save(filename)
            #             filename = args.results_path + '/images/train/PRR_%02d_mag_target_epoch_%02d.jpg' %(j,epoch)
            #             scipy.misc.toimage(x_PRR_target_mag_img[j,:,:]).save(filename)
            

            #     pdb.set_trace()
            #print("epoch %d: a_sign %03.2f: b_sign %03.2f: analogy_sign %03.2f" % (epoch,np.mean(a_sign),np.mean(b_sign),np.mean(d_sign)))
	        # if minimum loss is updated or final epoch, plot results
            # if min_tot_loss > tot_loss or epoch+1 == n_epochs:
	           #  min_tot_loss = tot_loss
	           #  # recon image
	           #  #pdb.set_trace()
	           #  if PRR:
	           #  	y_PRR = sess.run(y_xA, feed_dict={x: x_PRR_input,x_trans: x_PRR_target, keep_prob : 1})
	           #  	y_PRR_input_img = y_PRR.reshape(PRR.n_tot_imgs, n_dimensions, n_values)
	           #  	PRR.save_images(y_PRR_input_img, name="/PRR_epoch_%02d_input" %(epoch) + ".jpg")

	           #  	y_PRR = sess.run(y_xB, feed_dict={x: x_PRR_input,x_trans: x_PRR_target, keep_prob : 1})
	           #  	y_PRR_target_img = y_PRR.reshape(PRR.n_tot_imgs, n_dimensions, n_values)
	           #  	PRR.save_images(y_PRR_target_img, name="/PRR_epoch_%02d_trans" %(epoch) + ".jpg")

	           #  	y_PRR = sess.run(prediction_mag, feed_dict={c_mag: x_PRR_source_mag,x: x_PRR_input,x_trans: x_PRR_target, keep_prob : 1})
	           #  	y_PRR_pred_mag_img = y_PRR.reshape(PRR.n_tot_imgs, n_dimensions, n_values)
	           #  	PRR.save_images(y_PRR_pred_mag_img, name="/PRR_epoch_%02d_p_mag" %(epoch) + ".jpg")

	           #  if epoch==1:
	           #  	x_PRR_source_mag_img = x_PRR_source_mag.reshape(PRR.n_tot_imgs, n_dimensions, n_values)
	           #  	for j in range(20):
		          #      	filename = args.results_path + '/images/PRR_%02d_mag_source_test_epoch_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_source_mag_test_img[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/PRR_%02d_mag_target_test_epoch_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_target_mag_test_img[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/PRR_%02d_x_epoch_%02d.jpg' %(j,epoch)
		                
		          #       scipy.misc.toimage(x_PRR_img[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/PRR_%02d_x_trans_epoch_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_target_img[j,:,:]).save(filename)
		                
		          #       filename = args.results_path + '/images/train/PRR_%02d_A_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_img_trainA[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/train/PRR_%02d_B_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_img_trainB[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/train/PRR_%02d_mag_source_epoch_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_source_mag_img[j,:,:]).save(filename)
		          #       filename = args.results_path + '/images/train/PRR_%02d_mag_target_epoch_%02d.jpg' %(j,epoch)
		          #       scipy.misc.toimage(x_PRR_target_mag_img[j,:,:]).save(filename)
            # if epoch%1000==0:
	           #  y_PRRb = sess.run(prediction_mag, feed_dict={c_mag: x_PRR_source_mag, x: x_PRR_input,x_trans: x_PRR_target, keep_prob : 1})
	           #  y_PRR_pred_mag_img = y_PRRb.reshape(PRR.n_tot_imgs, n_dimensions, n_values)
	           #      #pdb.set_trace()
	           #  for j in range(PRR.n_tot_imgs):
	           #      filename = args.results_path + '/images/PRR_%02d_mag_p_epoch_%02d.jpg' %(j,epoch)
	           #      scipy.misc.toimage(y_PRR_pred_mag_img[j,:,:]).save(filename)

if __name__ == '__main__':
	# parse arguments
	args = parse_args()
	if args is None:
	    exit()
	# main
	main(args)