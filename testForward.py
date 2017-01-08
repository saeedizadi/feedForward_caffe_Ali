#!/usr/bin/env python
import numpy as np
import cPickle as pickle
from sklearn.neighbors import NearestNeighbors
import caffe

def unpickle(file):
    fo = open(file,'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict
    
def prepare_data(file):
    
    dict = unpickle(file)
    
    labels = dict['labels']
    labels = np.array(labels)
    
    data = dict['data']
    data = data.reshape([-1,3,32,32])
    data = data.transpose([0,2,3,1])
    #data = data.astype(np.float32)
    return data,labels

    
def doKNN(data):
    nbrs = NearestNeighbors(n_neighbors=2,algorithm='brute').fit(data)
    distances, indices = nbrs.kneighbors(data)
    indices = indices[:,1]
    distances = distances[:,1]
    return distances, indices
    
    
def main():
    
    np.set_printoptions(precision=2)
    caffe_root = '/home/saeed/Projects/caffe/'
    
    ## Load mean file for CIFAR-10
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( caffe_root + 'examples/cifar10/mean.binaryproto' , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob))
    meanArray = arr[0]
    
    ## Initialize the network
    model_file = caffe_root + 'models/bvlc_alexnet/bvlc_reference_caffenet.caffemodel'
    deploy_file = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
    net = caffe.Net(deploy_file,model_file,caffe.TEST)
    net.blobs['data'].reshape(1,3,32,32)              
    caffe.set_mode_cpu()

    
    ## Initialize the Transformer instance
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data',meanArray.mean(1).mean(1))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_channel_swap('data',(2,1,0))
    #transformer.set_raw_scale('data',255.0)

    
    #label_mapping = np.loadtxt("synset_words.txt", str, delimiter='\t')
    #image = caffe.io.load_image('BMW-2-series.jpg')

    
    pool5s = np.empty((0,4096),np.float32)
    labels = []
    data,labels = prepare_data('/home/saeed/Downloads/cifar-10-batches-py/test_batch')

    for index in range(data.shape[0]):
        print index
        img = data[index,:,:,:]                        
        net.blobs['data'].data[...] = transformer.preprocess('data',img)
        output = net.forward()
        temp = net.blobs['fc7'].data.flatten()
        temp = temp.reshape(1,temp.shape[0])                
        pool5s = np.append(pool5s,np.array(temp),axis=0)        
        
    print pool5s.shape
    print "---------------------------------------"
    dists, indices = doKNN(pool5s)
    preds = labels[np.array(indices)]
    #labels = labels[0:20]
    
    error = np.mean( preds != labels )
    print error
    print "---------------------------------------"

##    img = caffe.io.load_image('/home/saeed/Projects/caffe/examples/images/fish-bike.jpg')
##    img = Image.fromarray(data[100,:,:,:],'RGB')
##    img.show()

#        best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#        print label_mapping[best_n]

     ## Load the CIFAR-10 dataset
#    lmdb_env = lmdb.open(caffe_root + 'examples/cifar10/cifar10_test_lmdb')
#    lmdb_txn = lmdb_env.begin()
#    lmdb_cursor = lmdb_txn.cursor()
#    datum = caffe.proto.caffe_pb2.Datum()
    #    for key,value in lmdb_cursor:
#        datum.ParseFromString(value)
#        label = datum.label
#        data = caffe.io.datum_to_array(datum)        
#        data = data.astype(np.float32)
#        
#        test = net.blobs['data'].data[0]
#        #test= (test* 255).round().astype(np.uint8)
if __name__ == "__main__":
    main()
