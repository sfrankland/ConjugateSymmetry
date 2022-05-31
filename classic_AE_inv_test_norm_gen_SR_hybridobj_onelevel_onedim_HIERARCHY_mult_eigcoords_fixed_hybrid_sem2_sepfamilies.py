import tensorflow as tf
import numpy as np
from tempfile import TemporaryFile
import pdb
import random as random
import copy
from scipy.io import loadmat
import scipy.stats as stats
import sklearn

def make_analogy_data(n_dimensions, n_values, n_samples, n_TestSamples, eigs, labels):
    print('generating non-overlapping training and test data....')
    print('%02d training samples, %02d test samples' % (n_samples, n_TestSamples))
    print('%d dimensions X %d values' % (n_dimensions, n_values))
    #labels[0,13]=3
    minlevel=min(labels[0,:])
    maxlevel=max(labels[0,:])
    min_diff = 1
    max_diff=maxlevel-minlevel
    #here these should be 1 and 3.
    num_components= 3
    train_dataA = np.zeros([n_samples,1])
    train_dataB = np.zeros([n_samples,1])
    test_dataC = np.zeros([n_samples,1])
    test_dataD = np.zeros([n_samples,1])

    train_dataA_eig = np.zeros([n_samples, num_components])
    train_dataB_eig = np.zeros([n_samples, num_components])
    test_dataC_eig = np.zeros([n_samples, num_components])
    test_dataD_eig = np.zeros([n_samples, num_components])
    #pdb.set_trace()
    #n_values=int(n_values/2)
    outfile = TemporaryFile()
    # magnitude_source = np.zeros([n_samples, num_components])
    # magnitude_target = np.zeros([n_samples, num_components])
    foils = np.zeros([n_samples, 3, num_components])
    testdims = np.random.choice(n_dimensions, n_samples)
    magnitude_test_dims = np.random.choice(n_samples)
    #train
    magsource_train = np.zeros([n_samples,1])
    magtarget_train = np.zeros([n_samples,1])
    magtarget_test = np.zeros([n_samples,1])
    magsource_test = np.zeros([n_samples,1])

    magsource_train_eig = np.zeros([n_samples, num_components])
    magtarget_train_eig = np.zeros([n_samples, num_components])
    magtarget_test_eig = np.zeros([n_samples, num_components])
    magsource_test_eig = np.zeros([n_samples, num_components])

    dim2change=0
    testsamples=50
    #random A/B training samples. b is equivalent to A along all dimensions, except one.
    #randsample w/out replacement one x,y pair for each dimension. rule out 1/8.
    x2 = np.zeros([testsamples, 2])
    for j in range(testsamples):
        nogood=True
        while nogood:
            #pdb.set_trace()
            x2[j,:] = np.random.choice(n_values,size=2, replace=False)
            #pdb.set_trace()
            sdiff = int(labels[0,int(x2[j,0])])-int(labels[0,int(x2[j,1])])
            #signed_diff_labels = int(labels[0,int(currentC[0])]) - int(labels[0,int(currentD[0])])
            #constrained to be within same family.
            if abs(x2[j,0]-x2[j,1])>=n_values-1 or (labels[1,int(x2[j,0])]!=labels[1,int(x2[j,1])]):
            #if abs(x2[j,0]-x2[j,1])>=n_values-1 or (labels[1,int(x2[j,0])]!=labels[1,int(x2[j,1])]) or (labels[0,int(x2[j,0])]==labels[0,int(x2[j,1])]):
            #     nogood=True
            # else:
            #     nogood=False
            #if abs(x2[j,0]-x2[j,1])>=n_values-1 or labels[1,int(x2[j,0])]!=labels[1,int(x2[j,1])] or sdiff==0:
                 nogood=True
            else:
                 nogood=False
    #for extrapolation            
    #x2 = x2+40
    #pdb.set_trace()
    for i in range(n_samples):
        magnitude_source_data_train = np.zeros([n_dimensions,1])
        magnitude_target_data_train = np.zeros([n_dimensions,1])
        signed_diff=0
        train_dataA[i]=np.random.choice(n_values)
        while signed_diff==0:
            #ensure same family.
            candidates1 = np.where(labels[1,:]==labels[1,int(train_dataA[i])])
            #may want to keep out same level.
            candidates2 = np.where(labels[0,:]!=labels[0,int(train_dataA[i])])
            candidates = np.intersect1d(candidates1,candidates2)
            #candidates = np.concatenate((candidates1,candidates2), axis=1)
            #pdb.set_trace()
            v=np.arange(candidates.shape[0])
            pick = np.random.choice(v)
            train_dataB[i] = candidates[pick]
            #train_dataB[i] = candidates[0][pick]
        #train_dataB[i]=np.random.choice()
            signed_diff = train_dataA[i]-train_dataB[i]

        #ensure exclusion of test cases from training set.
        nocheck=1
        while nocheck==1:
            tocheck = np.where(x2[:,0]==train_dataA[i])
            checklength = len(tocheck[0])
            if checklength>0:
                res = np.zeros(checklength)
                for j in range(checklength):
                    res[j]=(train_dataB[i]==x2[tocheck[0][j],1])
                isthere = np.where(res)
                #pdb.set_trace()
                while train_dataB[i]==x2[isthere, 1]:
                	candidates = np.where(labels[1,:]==labels[1,int(train_dataA[i])])
                	pick = np.random.choice(candidates[0].shape[0])
                	train_dataB[i] = candidates[0][pick]
                nocheck=0
            else:
                nocheck=0

        # while signed_diff==0:
        #     train_dataB[i]=np.random.choice(n_values)
        #     signed_diff = train_dataA[i]-train_dataB[i]

        #signed_diff_mag = np.where(train_dataA[i])[0]-np.where(train_dataB[i])[0]
        #idx = np.where(train_dataA[i]==x2[:,0])
        # for l in range(n_values-1):
        #     possibility = np.random.choice(n_values)
        #     if (possibility!=train_dataA[i]):
        #         magnitude_source_data_train[dim2change] = possibility
        #         magnitude_target_data_train[dim2change] = possibility-signed_diff
        #         if magnitude_target_data_train[dim2change]<0:
        #             magnitude_target_data_train[dim2change]=n_values+magnitude_target_data_train[dim2change]
        #         elif magnitude_target_data_train[dim2change]>n_values-1:
        #             magnitude_target_data_train[dim2change]=magnitude_target_data_train[dim2change]-n_values
        #         break

        for l in range(n_values-1):
            possibility = np.random.choice(n_values)
            if (possibility - signed_diff)<n_values and (possibility - signed_diff)>-1 and (possibility!=train_dataA[i]):
                magnitude_source_data_train[dim2change] = possibility
                #level only. simple hierarchy.
                #pdb.set_trace()
                candidates = np.where(labels[1,:]!=labels[1,possibility])
                #candidates = np.where(labels[0,:]!=labels[0,possibility])
                pick = np.random.choice(candidates[0].shape[0])
                magnitude_target_data_train[dim2change] = candidates[0][pick]
                break
        #pdb.set_trace()
        magsource_train[i] = np.asarray(magnitude_source_data_train).reshape(-1)
        magtarget_train[i] = np.asarray(magnitude_target_data_train).reshape(-1)
        #pdb.set_trace()

        train_dataA_eig[i] = eigs[int(train_dataA[i]),:]
        train_dataB_eig[i] = eigs[int(train_dataB[i]),:]
        magsource_train_eig[i] = eigs[int(magsource_train[i]),:]
        magtarget_train_eig[i] = eigs[int(magtarget_train[i]),:]
    #pdb.set_trace()
    ########TEST
    #n_values=int(n_values*2)
    nTest=0
    foilpick = np.zeros([3,n_TestSamples])
    while nTest<n_TestSamples:
        curtrain_dataC = np.zeros([n_dimensions, 1])
        curtrain_dataD = np.zeros([n_dimensions, 1])
        magnitude_source_data_test = np.zeros([n_dimensions,1])
        magnitude_target_data_test = np.zeros([n_dimensions,1])
        
        dim2change = 0
        curtest=np.random.choice(testsamples)
        curtrain_dataC[dim2change]=int(x2[curtest,0])
        curtrain_dataD[dim2change]=int(x2[curtest,1])
        current_family = labels[1,int(x2[curtest,0])]
        analogy_family = current_family*-1+3
        #analogy_family=current_family
        currentC = np.asarray(curtrain_dataC).reshape(-1)
        currentD = np.asarray(curtrain_dataD).reshape(-1)        
        AC = min(np.sum((train_dataA - currentC) **2, 0))>0
        BC = min(np.sum((train_dataB - currentC) **2, 0))>0
        AD = min(np.sum((train_dataA - currentD) **2, 0))>0
        BD = min(np.sum((train_dataB - currentD) **2, 0))>0
        alltrue = AC*BC*AD*BD
        #pdb.set_trace()
        #make sure it's not one extreme to the other, in order to facilitate magnitude test.
        signed_diff = currentC[0] - currentD[0]
        #pdb.set_trace()
        signed_diff_labels = int(labels[0,int(currentC[0])]) - int(labels[0,int(currentD[0])])
        # if abs(signed_diff_labels)>2:
        #     pdb.set_trace()
        diff = abs(x2[dim2change,1]-x2[dim2change,0])
        direction = signed_diff>0
        #signed_diff_mag = np.where(currentC[0])-np.where(currentD[0])
        #print(signed_diff_labels)
        if alltrue:
            test_dataC[nTest,:] = np.asarray(curtrain_dataC).reshape(-1)
            test_dataD[nTest,:] = np.asarray(curtrain_dataD).reshape(-1)
            #CD_R = np.concatenate(labels[test_dataC,:],;labels[test_dataD,:])
            #pdb.set_trace()
            #constrained to not extend beyond tree structure of 3 levels.
            #for l in range(n_values-1):
            #pdb.set_trace()
            goodcandidate=0
            itr=1
            while goodcandidate==0:
                itr+=1
                print(goodcandidate)
                print(itr)
                print(signed_diff_labels)
                if signed_diff_labels==-3:
                    #A to B goes from top to bottom. So C needs to be at top (min) level too.
                    possibilitiesa = np.where(labels[0,:]==minlevel)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilitiesa,fam)
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1

                    #pdb.set_trace()
                if signed_diff_labels==3:
                    #A to B goes from bottom to top. So C needs to be at bottom (top) level too.
                    possibilitiesa = np.where(labels[0,:]==maxlevel)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilitiesa,fam)
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1
                    #pdb.set_trace()

                if signed_diff_labels==0:
                    possibilitiesa = np.arange(n_values)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilitiesa,fam)
                    #pdb.set_trace()
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1

                if signed_diff_labels==2:
                    #A to B goes from top to bottom. So C needs to be at top (min) level too.
                    possibilitiesa = np.where(labels[0,:]==maxlevel)
                    possibilitiesb = np.where(labels[0,:]==maxlevel-1)
                    #pdb.set_trace()
                    possibilities=np.concatenate((possibilitiesa,possibilitiesb), axis=1)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilities,fam)
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1
                    #pdb.set_trace()
                if signed_diff_labels==-2:
                    #A to B goes from bottom to top. So C needs to be at bottom (top) level too.
                    #and max level -1
                    possibilitiesa = np.where(labels[0,:]==minlevel)
                    possibilitiesb = np.where(labels[0,:]==minlevel+1)
                    #pdb.set_trace()
                    possibilities=np.concatenate((possibilitiesa,possibilitiesb), axis=1)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilities,fam)
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1

                if signed_diff_labels==1:
                #     # a difference of one. exclude bottom level.
                    possibilitiesa = np.where(labels[0,:]!=minlevel)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilitiesa,fam)
                    #pdb.set_trace()
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1

                if signed_diff_labels==-1:
                #     # a difference of -1one. exclude top level.
                    possibilitiesa = np.where(labels[0,:]!=maxlevel)
                    fam = np.where(labels[1,:]==analogy_family)
                    possibilities = np.intersect1d(possibilitiesa,fam)
                    #pdb.set_trace()
                    possibilities2= possibilities!=test_dataC[nTest,:]
                    possibility = possibilities[np.random.choice(possibilities2.shape[0])]
                    #pdb.set_trace()
                    if possibility!=test_dataC[nTest, :]:
                        goodcandidate=1

            #make sure it's in the same family
            if int(labels[0,possibility]!=1):
                candidates = np.where(labels[1,:]==labels[1,possibility])
            else:
                candidates = np.where(labels[1,:])
                
            pick = candidates[0][np.random.choice(candidates[0].shape[0])]

                #pdb.set_trace()
                #if the difference between C/D does not match the difference between A/B, keep trying.
            print('signed_diff_labels %d'  % (signed_diff_labels))
            print('signed_diff %d'  % (signed_diff))
            print('test_dataC %d'  % (test_dataC[nTest, :]))
            print('test_dataD %d'  % (test_dataD[nTest,:]))
            print('possibility %d'  % (possibility))
            print('inital pick %d' % (pick))
            tctr=1
            TF = int(labels[1,possibility])!=int(labels[1,pick]) and int(labels[0,possibility]!=1)
            while pick == possibility or int(labels[0,possibility])-int(labels[0,pick])!=int(signed_diff_labels):
                pick = candidates[0][np.random.choice(candidates[0].shape[0])]
                TF = int(labels[1,possibility])!=int(labels[1,pick]) and int(labels[0,possibility]!=1)
                print('possible pick %d' % (pick))
                print('signed_diff_labels %d'  % (signed_diff_labels))
                print('signed_diff %d'  % (signed_diff))
                print('test_dataC %d'  % (test_dataC[nTest, :]))
                print('test_dataD %d'  % (test_dataD[nTest,:]))
                print('possibility %d'  % (possibility))
                print('counter %d' % tctr)
                tctr+=1
                if tctr>50000:
                     pdb.set_trace()

            #pdb.set_trace()
            print('final pick %d' % (pick))
            #goodcandidate=1
            magnitude_source_data_test[dim2change] = possibility
            magnitude_target_data_test[dim2change] = pick
            #break
            print(nTest)
            #print(signed_diff_labels)

        magsource_test[nTest] = np.asarray(magnitude_source_data_test).reshape(-1)
        magtarget_test[nTest] = np.asarray(magnitude_target_data_test).reshape(-1)
            #magtarget_oppsign = np.asarray(magnitude_source_data_test-signed_diff*-1).reshape(-1)
            #magtarget_oppsign =magnitude_source_data_test-(signed_diff*-1)
            #pdb.set_trace()
        for m in range(3):
            #pdb.set_trace()
            foilcandidates = np.where(labels[0,:]!=labels[0,int(magtarget_test[nTest])])
            #same generation,but other family.
            i1 = np.where(labels[0,:]==labels[0,int(magtarget_test[nTest])])
            i2 = np.where(labels[1,:]!=labels[1,int(magtarget_test[nTest])])
            i3 = np.intersect1d(i1,i2)
            #same generation, same family, but nephew/uncle.
            i1b = np.where(labels[0,:]==labels[0,int(magtarget_test[nTest])])
            i2b = np.where(labels[1,:]==labels[1,int(magtarget_test[nTest])])
            #pdb.set_trace()
            i3b = np.intersect1d(i1b,i2b)
            i4b = np.where(labels[3,:]!=labels[3,int(magtarget_test[nTest])])
            i5b = np.intersect1d(i3b,i4b)
            #foilcandidates2 = np.concatenate((foilcandidates[0],i3),axis=0)
            foilcandidates2 = np.concatenate((foilcandidates[0],i3,i5b),axis=0)
            foilpick[m,nTest] = foilcandidates[0][np.random.choice(foilcandidates[0].shape[0])]
            #currentfoil = np.random.choice(foilcandidates)
            #break
            #while abs(currentfoil-magnitude_target_data_test)<2 or abs(currentfoil-magtarget_oppsign)<2:
            # while abs(currentfoil-magnitude_target_data_test)<4:
            #     currentfoil = np.random.choice(n_values)
           #
            #pdb.set_trace()
            foils[nTest, m,:]=eigs[int(foilpick[m,nTest]),:]
        #pdb.set_trace()
        
        test_dataC_eig[nTest] = eigs[int(test_dataC[nTest]), :]
        test_dataD_eig[nTest] = eigs[int(test_dataD[nTest]),:]
        magsource_test_eig[nTest] = eigs[int(magsource_test[nTest]),:]
        magtarget_test_eig[nTest] = eigs[int(magtarget_test[nTest]),:]

        nTest+=1
        train_place_code=magsource_train-magtarget_train

    #pdb.set_trace()
    return train_dataA_eig, train_dataB_eig, magsource_train_eig, magtarget_train_eig, test_dataC_eig, test_dataD_eig, magsource_test_eig, magtarget_test_eig, foils, train_place_code

def direction_decoder_MLP_regression(x, n_hidden, keep_prob):
    with tf.variable_scope("direction_decoder_MLP_regression"):
        #2 layers
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        w0_db = tf.get_variable('w0_db', [x.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_db = tf.get_variable('b0_db', [n_hidden], initializer=b_init_d)
        l1_d = tf.matmul(x, w0_db) + b0_db 
        #layer 2
        w1_db = tf.get_variable('w1_db', [l1_d.get_shape()[1], 1], initializer=w_init_d)
        b1_db = tf.get_variable('b1_db', [1], initializer=b_init_d)
        l2_d = tf.matmul(l1_d, w1_db) + b1_db
    return l2_d


def level1_encoder(x, n_hidden, n_output, keep_prob, reuse=None):
    with tf.variable_scope("level1_encoder", reuse=reuse):
        #2 layers
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        w0_db = tf.get_variable('w0_db', [x.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_db = tf.get_variable('b0_db', [n_hidden], initializer=b_init_d)
        l1_d = tf.matmul(x, w0_db) + b0_db 
        #layer 2
        w1_db = tf.get_variable('w1_db', [l1_d.get_shape()[1], n_hidden], initializer=w_init_d)
        b1_db = tf.get_variable('b1_db', [n_hidden], initializer=b_init_d)
        l2_d = tf.matmul(l1_d, w1_db) + b1_db
    return l2_d

# taking latent code (z) as input, mapping to R.
def relation_encoder1(x, n_hidden, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder1"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [n_hidden], initializer=b_init_d)
        h0_d = tf.matmul(x, w0_d) + b0_d
        # 2nd hidden layer
        w1_d = tf.get_variable('w1_d', [h0_d.get_shape()[1], n_hidden], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [n_hidden], initializer=b_init_d)
        h1_d = tf.nn.sigmoid(tf.matmul(h0_d, w1_d) + b1_d)
        # output layer-mean
        wo_d = tf.get_variable('wo_d', [h1_d.get_shape()[1], dim_r], initializer=w_init_d)
        bo_d = tf.get_variable('bo_d', [dim_r], initializer=b_init_d)
        y_d = tf.matmul(h1_d, wo_d) + bo_d
    return y_d

def relation_encoder_direct(x, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_direct"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        # w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        # b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        # y_d = tf.matmul(x, w0_d) + b0_d
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        #w_init_d = tf.truncated_normal_initializer(0,0.5)
        w_init_d = tf.constant_initializer(0.)
        # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        Rb = tf.matmul(x, w0_d) + b0_d
        #Ra=Rb*-1
        Ra=Rb*-1
    return Ra,Rb

def relation_encoder_direct_quadratic(x, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_direct"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        # w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        # b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        # y_d = tf.matmul(x, w0_d) + b0_d
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        #w_init_d = tf.truncated_normal_initializer(0,0.5)
        w_init_d = tf.constant_initializer(0.)
        # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        Rb = (tf.matmul(x, w0_d) + b0_d)**2
        #Ra=Rb*-1
        Ra=Rb*-1
    return Ra,Rb

def relation_encoder_notdirect(x, n_hidden,dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_notdirect"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [n_hidden], initializer=b_init_d)
        y_d = tf.nn.relu(tf.matmul(x, w0_d) + b0_d)
        # 1st hidden layer
        w1_d = tf.get_variable('w1_d', [y_d.get_shape()[1], dim_r], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        Rb = tf.matmul(y_d, w1_d) + b1_d
        #Ra=Rb*-1
        Ra=Rb*-1
    return Ra,Rb

def relation_encoder_notdirect_quadratic(x, n_hidden,dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_notdirect_quadratic"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [n_hidden], initializer=b_init_d)
        y_d = tf.nn.relu(tf.matmul(x, w0_d) + b0_d)
        # 1st hidden layer
        w1_d = tf.get_variable('w1_d', [y_d.get_shape()[1], dim_r], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        Rb = abs(tf.matmul(y_d, w1_d) + b1_d)
        #Ra=Rb*-1
        Ra=Rb*-1
    return Ra,Rb


def relation_encoder_direct_notfixed_back(x, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_direct_notfixed_back"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        # w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        # b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        # y_d = tf.matmul(x, w0_d) + b0_d
        #this was changed 9/18.
        #w_init_d = tf.contrib.layers.variance_scaling_initializer()
        #w_init_d = tf.truncated_normal_initializer(0,0.5)
        w_init_d = tf.constant_initializer(0.)
        # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        Rb0 = tf.matmul(x, w0_d) + b0_d

        # 1st hidden layer
        w1_d = tf.get_variable('w1_d', [Rb0.get_shape()[1], dim_r], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        Ra = tf.matmul(Rb0, w1_d) + b1_d

        # 1st hidden layer
        w2_d = tf.get_variable('w2_d', [Ra.get_shape()[1], dim_r], initializer=w_init_d)
        b2_d = tf.get_variable('b2_d', [dim_r], initializer=b_init_d)
        Rb = tf.matmul(Ra, w2_d) + b2_d

    return Ra,Rb

# #latent chain in R
def relation_encoder2(r, dim_r, keep_prob, reuse=None):
    with tf.variable_scope("relation_encoder2", reuse=reuse):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # r2a
        w0_d = tf.get_variable('w0_d', [r.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        Rb = tf.matmul(r, w0_d) + b0_d
        #a2b
        # w1_d = tf.get_variable('w1_d', [w0_d.get_shape()[1], dim_r], initializer=w_init_d)
        # b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        # Ra = tf.sigmoid(tf.matmul(r, w1_d) + b1_d)
        w1_d = tf.get_variable('w1_d', [r.get_shape()[1], dim_r], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        Ra = tf.matmul(r, w1_d) + b1_d
        # w2_d = tf.matrix_inverse(w1_d)
        # b2_d = tf.get_variable('b2_d', [dim_r], initializer=b_init_d)
        # Rb0 = tf.matmul(Ra, w2_d) + b2_d

    return Ra, Rb

def relation_encoder_direct_inverse(x, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder_direct_inverse"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # # 1st hidden layer
        # w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        # b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        # y_d = tf.matmul(x, w0_d) + b0_d
        # w_init_d = tf.contrib.layers.variance_scaling_initializer()
        # #w_init_d = tf.truncated_normal_initializer(0,0.5)
        # w_init_d = tf.constant_initializer(0.)
        # 1st hidden layer
        w0_d = tf.get_variable('w0_d', [x.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        Rb0 = tf.matmul(x, w0_d) + b0_d

        w1_d = tf.get_variable('w1_d', [dim_r, dim_r], initializer=w_init_d)
        b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        Ra = tf.matmul(Rb0, w1_d) + b1_d
        
        w2_d = tf.matrix_inverse(w1_d)
        b2_d = tf.get_variable('b2_d', [dim_r], initializer=b_init_d)
        Rb = tf.matmul(Ra, w2_d) + b2_d

    return Ra,Rb

# #latent chain in R
def relation_encoder3(r, dim_r, keep_prob):
    with tf.variable_scope("relation_encoder3"):
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        # r2a
        w0_d = tf.get_variable('w0_d', [r.get_shape()[1], dim_r], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d', [dim_r], initializer=b_init_d)
        R = tf.matmul(r, w0_d) + b0_d
        #a2b
        # # w1_d = tf.get_variable('w1_d', [w0_d.get_shape()[1], dim_r], initializer=w_init_d)
        # # b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        # # Ra = tf.sigmoid(tf.matmul(r, w1_d) + b1_d)
        # w1_d = tf.get_variable('w1_d', [r.get_shape()[1], dim_r], initializer=w_init_d)
        # b1_d = tf.get_variable('b1_d', [dim_r], initializer=b_init_d)
        # Ra = tf.matmul(r, w1_d) + b1_d
        # w2_d = tf.matrix_inverse(w1_d)
        # b2_d = tf.get_variable('b2_d', [dim_r], initializer=b_init_d)
        # Rb0 = tf.matmul(Ra, w2_d) + b2_d

    return R

def level1_decoder(z, n_hidden, n_output, keep_prob, reuse=None):
    with tf.variable_scope("level1_decoder", reuse=reuse):
        #2 layers
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        w0_d1 = tf.get_variable('w0_d1', [z.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_d1 = tf.get_variable('b0_d1', [n_hidden], initializer=b_init_d)
        l1_d = tf.sin(tf.matmul(z, w0_d1) + b0_d1)
        
        w1_d1 = tf.get_variable('w1_d1', [l1_d.get_shape()[1], (n_hidden*n_hidden)/100], initializer=w_init_d)
        b1_d1 = tf.get_variable('b1_d1', [(n_hidden*n_hidden)/100], initializer=b_init_d)
        l2_d = tf.sin(tf.matmul(l1_d, w1_d1) + b1_d1)
        l2_d = tf.nn.dropout(l2_d, keep_prob)

        w4_d1 = tf.get_variable('w4_d1', [l2_d.get_shape()[1], n_output], initializer=w_init_d)
        b4_d1 = tf.get_variable('b4_d1', [n_output], initializer=b_init_d)
        l5_d = tf.matmul(l2_d, w4_d1) + b4_d1
    return l5_d


def level1_decoder_b(z, n_hidden, n_output, keep_prob, reuse=None):
    with tf.variable_scope("level1_decoder_b", reuse=reuse):
        #2 layers
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        w0_d1b = tf.get_variable('w0_d1b', [z.get_shape()[1], n_hidden], initializer=w_init_d)
        b0_d1b = tf.get_variable('b0_d1b', [n_hidden], initializer=b_init_d)
        l1_db= tf.matmul(z, w0_d1b) + b0_d1b
        
        w1_d1b = tf.get_variable('w1_d1b', [l1_db.get_shape()[1], (n_hidden*n_hidden)/100], initializer=w_init_d)
        b1_d1b = tf.get_variable('b1_d1b', [(n_hidden*n_hidden)/100], initializer=b_init_d)
        l2_db = tf.nn.relu(tf.matmul(l1_db, w1_d1b) + b1_d1b)
        l2_db = tf.nn.dropout(l2_db, keep_prob)

        w4_d1b = tf.get_variable('w4_d1', [l2_db.get_shape()[1], n_output], initializer=w_init_d)
        b4_d1b = tf.get_variable('b4_d1', [n_output], initializer=b_init_d)
        l5_db = tf.matmul(l2_db, w4_d1b) + b4_d1b
    return l5_db


def level1_decoder2(z, n_hidden, n_output, keep_prob, reuse=None):
    with tf.variable_scope("level1_decoder2", reuse=reuse):
        #2 layers
        w_init_d1 = tf.contrib.layers.variance_scaling_initializer()
        b_init_d1 = tf.constant_initializer(0.)
        #pdb.set_trace()
        w0_d1_b = tf.get_variable('w0_d1_b', [z.get_shape()[1], 20], initializer=w_init_d1)
        b0_d1_b = tf.get_variable('b0_d1_b', [20], initializer=b_init_d1)
        l1_d_b = tf.nn.relu(tf.matmul(z, w0_d1_b) + b0_d1_b)
        l1_d_b = tf.nn.dropout(l1_d_b, keep_prob)
        
        w1_d1_b = tf.get_variable('w1_d1_b', [l1_d_b.get_shape()[1], (n_hidden)], initializer=w_init_d1)
        b1_d1_b = tf.get_variable('b1_d1_b', [(n_hidden)], initializer=b_init_d1)
        l2_d_b = tf.nn.relu(tf.matmul(l1_d_b, w1_d1_b) + b1_d1_b)
        l2_d_b = tf.nn.dropout(l2_d_b, keep_prob)

        w2_d1_b = tf.get_variable('w2_d1_b', [l2_d_b.get_shape()[1], n_output], initializer=w_init_d1)
        b2_d1_b = tf.get_variable('b2_d1_b', [n_output], initializer=b_init_d1)
        l3_d_b = tf.matmul(l2_d_b, w2_d1_b) + b2_d1_b
    return l3_d_b

def level1_decoder_direct(x, z, n_output, keep_prob, reuse=None):
    with tf.variable_scope("level1_decoder_direct", reuse=reuse):
        #w_init_d = tf.truncated_normal_initializer(0,0.5)
        w_init_d = tf.contrib.layers.variance_scaling_initializer()
        b_init_d = tf.constant_initializer(0.)
        w0_d = tf.get_variable('w0_d1', [z.get_shape()[1], n_output], initializer=w_init_d)
        b0_d = tf.get_variable('b0_d1', [n_output], initializer=b_init_d)

        w_init_d1 = tf.contrib.layers.variance_scaling_initializer()
        b_init_d1 = tf.constant_initializer(0.)
        w0_d1 = tf.get_variable('copy_w', [x.get_shape()[1], n_output], initializer=w_init_d1)
        b0_d1 = tf.get_variable('copy_b', [n_output], initializer=b_init_d)
        
        l1_d = (tf.matmul(z, w0_d) + b0_d) + tf.matmul(x, w0_d1)
        #freeze copy operation
        tf.Variable(w0_d1, trainable=False)
    return l1_d

###autoencoder function
def onelevel_autoencoder(x, x_trans, c_mag,d_mag,dir_test, dim_img, dim_z, dim_r, n_hidden, keep_prob):
    #####################level1
    ######################level 2
    #relational encoding of low-d objects.
    #Rb,Ra = relation_encoder_notdirect(tf.concat([x,x_trans], 1),5,dim_z, keep_prob)
    Rb,Ra = relation_encoder_direct(tf.concat([x,x_trans], 1),dim_z, keep_prob)
    #Ra,Rb = relation_encoder_direct_notfixed_back(tf.concat([x,x_trans], 1),dim_z, keep_prob)
    #Rb = relation_encoder1(tf.concat([x,x_trans],1), dim_z,n_hidden, keep_prob)
    #Rb = relation_encoder3(tf.concat([x,x_trans],1), dim_z,keep_prob)
    #Ra=Rb
    #activity-regularization
    # Rb_norm = tf.nn.l2_loss(Rb)*10
    # Rb_norm = tf.nn.l2_loss(Rb)*10
    #direction_pred = level1_decoder(tf.concat([Rb], 1),dim_z, 1,keep_prob)
    #Ra,Rb = relation_encoder_direct(R,dim_z, keep_prob) 
    #R to Z decoder
    #pdb.set_trace()
    with tf.variable_scope("level1_decoder2", reuse=None) as scope:
        #y_xB = level1_decoder(tf.concat([x,tf.concat([Rb,1],1)],1),n_hidden,dim_img, keep_prob)
        y_xB = level1_decoder2(tf.concat([x,Rb],1),n_hidden,dim_img, keep_prob)
    #y_xB = tf.clip_by_value(y_xB, 1e-8, 1 - 1e-8)
    #reconstruct A, given R,B
    with tf.variable_scope("level1_decoder2", reuse=True) as scope:
        #y_xA = level1_decoder_b(tf.concat([x_trans,tf.concat([Rb,-1],1)],1),n_hidden,dim_img, keep_prob)
        y_xA = level1_decoder2(tf.concat([x_trans,Ra],1),n_hidden,dim_img, keep_prob)
    #Rb=[]
    # with tf.variable_scope("level1_decoder2", reuse=None) as scope:
    #     y_xA = level1_decoder2(tf.concat([x,tf.subtract(x,x_trans)],1),n_hidden,dim_img, keep_prob)
    # # #y_xB = tf.clip_by_value(y_xB, 1e-8, 1 - 1e-8)
    # # #reconstruct A, given R,B
    # with tf.variable_scope("level1_decoder2", reuse=True) as scope:
    #     y_xB = level1_decoder2(tf.concat([x_trans,tf.subtract(x_trans,x)],1),n_hidden,dim_img, keep_prob)

    #dir_test = x[:,2]<x_trans[:,2]*1
    SE1_temp = (y_xA-x)**2
    SE2_temp = (y_xB-x_trans)**2 
    MSE = tf.reduce_mean(tf.stack([tf.reduce_mean(SE1_temp), tf.reduce_mean(SE2_temp)]))
    #loss=w_MSE
##analogy test. generate predicted 'd' for each source.
    # with tf.variable_scope("level1_decoder2", reuse=True) as scope:
    #      prediction_mag = level1_decoder2(tf.concat([c_mag,Rb],1), n_hidden, dim_img, keep_prob)
    with tf.variable_scope("level1_decoder2", reuse=True) as scope:
        #prediction_mag = level1_decoder(tf.concat([c_mag,tf.concat([Rb,1],1)],0), n_hidden, dim_img, keep_prob)
        prediction_mag = level1_decoder2(tf.concat([c_mag,Rb],1), n_hidden, dim_img, keep_prob)
    with tf.variable_scope("level1_decoder2", reuse=True) as scope:
        prediction_mag_c = level1_decoder2(tf.concat([d_mag,Ra],1), n_hidden, dim_img, keep_prob)
        #prediction_mag_c = level1_decoder(tf.concat([d_mag,tf.concat([Rb,-1],1)],0), n_hidden, dim_img, keep_prob)
    # with tf.variable_scope("level1_decoder2", reuse=None) as scope:
    #     direction_pred = level1_decoder2(tf.concat([prediction_mag,prediction_mag_c], 1),dim_z, 1,keep_prob)
    # with tf.variable_scope("direction_decoder_MLP_regression") as scope:
    #     direction_pred = direction_decoder_MLP_regression(tf.concat([y_xA,y_xB],1), n_hidden, keep_prob)
    # with tf.variable_scope("level1_decoder2", reuse=True) as scope:
    #     prediction_mag = level1_decoder(tf.concat([c_mag,tf.subtract(x_trans,x)],1), n_hidden, dim_img, keep_prob)
    #w_a=1
    #testdiffs = w_a*((c_mag-d_mag)**2)+1
    #randchoice = np.random.choice(64,64)
    #id_rows = tf.constant([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]])
    id_rows = tf.constant([[0],[1],[2],[3],[4],[5],[6],[7],[8]])
    #id_rows = tf.constant([[0]])
    prediction_mag_sub = tf.gather_nd(prediction_mag,id_rows)
    d_mag_sub = tf.gather_nd(d_mag,id_rows)
    prediction_mag_c_sub = tf.gather_nd(prediction_mag_c,id_rows)
    c_mag_sub = tf.gather_nd(c_mag,id_rows)
    
    #direction_pred_sub = tf.gather_nd(direction_pred,id_rows)
    #dir_test_sub = tf.gather_nd(dir_test,id_rows)
    #pdb.set_trace()
    SE3_temp = (prediction_mag_sub-d_mag_sub)**2
    SE4_temp = (prediction_mag_c_sub-c_mag_sub)**2
    #SE5_temp = (direction_pred_sub-dir_test_sub)**2
    
    #SE5_temp = (direction_pred-dir_test)**2
    #diff_MSE = tf.reduce_mean(SE5_temp)
    # l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.99, scope=None)
    # #weights = tf.trainable_variables() # all vars
    # #pdb.set_trace()
    # weights =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'relation_encoder3/w0_d:0')
    # l1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    #pdb.set_trace()
    #dir_loss = tf.losses.softmax_cross_entropy(direction_pred_sub, dir_test_sub)
    #dir_loss = tf.losses.softmax_cross_entropy(direction_pred, dir_test)
    #SE3_temp = (prediction_mag-d_mag)**2
    #SE4_temp = (prediction_mag_c-c_mag)**2
    #analogy_MSE = tf.reduce_mean(SE3_temp)
    # randchoice = np.random.choice(64,64)
    # SE3_temp_subset = tf.reduce_mean(SE3_temp[randchoice[:][1:60]])
    # SE4_temp_subset = tf.reduce_mean(SE4_temp[randchoice[:][1:60]])
    #print(randchoice)
    #analogy_MSE = tf.reduce_mean(tf.stack([tf.reduce_mean(SE3_temp[np.arange(64)[0]]), tf.reduce_mean(SE4_temp[np.arange(64)[0]])]))
    analogy_MSE = tf.reduce_mean(tf.stack([SE3_temp,SE4_temp]))
    #loss = tf.reduce_mean(tf.stack([MSE, MSE,MSE,MSE, MSE,MSE,MSE, MSE,MSE,analogy_MSE]))
    #loss=tf.reduce_mean(tf.stack([MSE,MSE,MSE,analogy_MSE]))
    #loss=MSE+analogy_MSE
    #loss=tf.concat([MSE_loss,analogy_MSE[0]],0)
    #MSE=analogy_MSE
    loss=MSE
    #loss=MSE+analogy_MSE
    return y_xA, y_xB, Rb, prediction_mag,loss, analogy_MSE, MSE