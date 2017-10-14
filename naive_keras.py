import numpy as np
import pandas as pd
import os, cv2
from time import time
from sklearn.svm import LinearSVC

try:
    from IPython.display import display
except:
    display = print

class Layer:
    def __init__(self, input_shape=None, output_shape=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Define output_shape as a function of input_shape, unless this is not provided.
        # Even if output_shape is provided, check that it matches.
        if input_shape is not None:
            if output_shape is None:
                self.set_output_shape()
            else:
                self.set_output_shape()
                assert self.output_shape == tuple(output_shape)

    def set_output_shape(self, input_shape=None):
        raise NotImplementedError("Please Implement this method (class of type: {})".format(type(self)))

    def check_input_shape(self, x):
        bad = [i for i in range(1, len(x.shape)) if x.shape[i] != self.input_shape[i]]
        if len(bad) > 0:
            raise NameError('The {}-th dim of x is {}, should be {} (plus {} more dims)'.format(bad[0], x.shape[bad[0]], self.input_shape[bad[0]], len(bad)-1))

class Dense(Layer):
    def __init__(self, units, activation='relu', input_shape=None, output_shape=None):
        self.units = units
        self.activation = activation
        
        Layer.__init__(self, input_shape, output_shape)

    def set_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape

        output_shape = (input_shape[0], self.units)
        assert self.output_shape is None or self.output_shape == output_shape
        self.output_shape = output_shape
        # Initialize as normal vectors
        self.weights = (np.random.randn(np.prod(input_shape[1:]), self.units) / np.sqrt(np.random.randn(np.prod(input_shape[1:]))),
                        np.zeros(self.units))

    def predict(self, x):
        self.check_input_shape(x)

        # Flatten all dimension of x (except sample)
        x = x.reshape(x.shape[0], -1)
        
        output = x.dot(self.weights[0]) + self.weights[1].reshape(1,-1)

        if self.activation == 'relu':
            return np.maximum(output, 0)
        elif self.activation is None:
            return output
        elif self.activation == 'softmax':
            e_x = np.exp(output - np.max(output))
            return e_x / np.sum(e_x)
        
    

class Conv2D(Layer):
    def __init__(self, filters, kernel_size=3, activation='relu', input_shape=None, output_shape=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.save_memory = False

        Layer.__init__(self, input_shape, output_shape)
                                        

    def set_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        output_shape = (input_shape[0], input_shape[1] -self.kernel_size+1, input_shape[2]-self.kernel_size+1, self.filters)
        assert self.output_shape is None or self.output_shape == output_shape
        self.output_shape = output_shape
        # Initialize as normal vectors
        self.weights = (np.random.randn(self.kernel_size**2 * input_shape[3], self.filters) / (self.kernel_size*self.input_shape[3]**0.5),
                        np.zeros(self.filters))


    def predict(self, x):
        self.check_input_shape(x)
        if not self.save_memory:
            # Simpler format, just use "extract_submatrices".
            output = (self.extract_submatrices(x).dot(self.weights[0]) + \
                      self.weights[1].reshape(1,1,-1)).reshape(x.shape[0], *self.output_shape[1:])
        else:
            # With for loops, but does not create >9x memory overhead.
            output = np.zeros((x.shape[0], *self.output_shape[1:]))
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    output += x[:, i:i+self.output_shape[1], j:j+self.output_shape[2]].dot(self.weights[0].reshape(self.kernel_size, self.kernel_size, -1, self.filters)[i,j,:]) +\
                              self.weights[1].reshape(1,1,1,-1)

        if self.activation == 'relu':
            return np.maximum(output, 0)
        elif self.activation is None:
            return output
        else:
            raise NameError('Unknown activation function {}'.format(self.activation))

    def extract_submatrices(self, x):
        return np.array([x[:, i:i+self.output_shape[1], j:j+self.output_shape[2]] for i in range(self.kernel_size) for j in range(self.kernel_size)]).transpose([1,2,3,0,4]).reshape(x.shape[0], self.output_shape[1]*self.output_shape[2], self.kernel_size ** 2 * self.input_shape[3])
        

class ZeroPadding2D(Layer):
    def set_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        output_shape = (input_shape[0], input_shape[1]+2, input_shape[2]+2, input_shape[3])
        assert self.output_shape is None or self.output_shape == output_shape
        self.output_shape = output_shape

    def predict(self, x):
        self.check_input_shape(x)
        output = np.zeros((x.shape[0], *self.output_shape[1:]))
        output[:,1:-1, 1:-1, :] = x
        return output


class MaxPooling2D(Layer):
    def set_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        assert input_shape[1] % 2 == 0 and input_shape[2] % 2 == 0, 'Should have even middle dimensions, instead {} and {}'.format(*input_shape[1:3])

        output_shape = (self.input_shape[0], self.input_shape[1] // 2, self.input_shape[2] // 2, self.input_shape[3])
        assert self.output_shape is None or self.output_shape == output_shape
        self.output_shape = output_shape

    def predict(self, x):
        self.check_input_shape(x)
        # At this point, the dimensions should be even already by the constructor...
        return np.maximum(np.maximum(x[:, ::2, ::2], x[:,1::2,::2]), np.maximum(x[:,::2,1::2], x[:,1::2,1::2]))


class Sequential:
    def __init__(self, *args):
        assert all([isinstance(m, Layer) for m in args]), 'All args of Model should be Layers! Instead {}'.format([type(m) for m in args])
        assert args[0].input_shape is not None, 'The first model has to have a specified input_shape.'

        self.layers = args
        
        for i in range(len(self.layers)):
            if self.layers[i].input_shape is None:
                # here i must be > 0 by the assertion above
                self.layers[i].input_shape = self.layers[i-1].output_shape
            elif i > 0:
                assert self.layers[i].input_shape == self.layers[i-1].output_shape
            self.layers[i].set_output_shape()

    def predict(self, x, up_to_layer = None, until_layer = None):
        ''' Predicts all through the network. Unless one of the two silly additional
            parameters are set, in which case stops at the first between up_to_layer or
            the one before until_layer. '''
        for l in self.layers:
            if l == until_layer:
                break
            x = l.predict(x)
            if l == up_to_layer:
                break
        return x

    def summary(self):
        output = pd.DataFrame(index = [type(l) for l in self.layers], columns=['input_shape', 'output_shape'])
        output['input_shape'] = [l.input_shape[1:] for l in self.layers]
        output['output_shape'] = [l.output_shape[1:] for l in self.layers]
        display(output)
        

    # From here on, experimental stuff on initializing by SVM
    def initialize_by_SVM(self, folder, batch_size=64):
        self.folder = folder
        for l in self.layers:
            # Fit quickly on small batches for convolution layers or dense layers - except the last one.
            if isinstance(l, Conv2D) or (isinstance(l, Dense) and [l1 for l1 in self.layers if isinstance(l1, Dense)][::-1].index(l) == 0)
                last_svms = self.initialize_layer_by_SVM(l, batch_size=batch_size)
            elif isinstance(l, Dense):
                # Last dense layer, do a proper fit, with random load, shuffling, etc.
                self.fit_Dense_layer(l, epochs=1)

        # Returns only the last svms for debugging purposes.
        return last_svms

    def initialize_layer_by_SVM(self, l, batch_size=64):
        input_shape = self.layers[0].input_shape

        svms = []

        l.weights = (np.zeros((l.kernel_size**2 * l.input_shape[3], l.filters)), np.zeros(l.filters))

        time_in_load = 0
        time_in_predict = 0
        time_in_fit = 0
        total_time = 0
        
        for f in range(l.filters):

            tstart = time()
            pics, labels = self.load_random_input(batch_size, input_shape[1:], normalize=True, balanced=True)
            time_in_load += time() - tstart
            
            if len(set(labels)) != 2:
                raise NameError('For the moment, only 2 classes case is implemented...')

            # For the moment, only do the version "without memory saving"

            # Predict up to l (excluding) and take all the submatrices.
            tstart = time()
            pics = self.predict(pics, until_layer=l)
            time_in_predict += time() - tstart
            
            reshaped = sm.reshape(-1, sm.shape[2])

            tstart = time()
            if isinstance(l, Conv2D):
                sm = l.extract_submatrices(pics)
                svms = fit_and_append(svms, sm.reshape(-1, sm.shape[2]), np.repeat(labels, sm.shape[1]))
            elif isinstance(l, Dense):
                svms = fit_and_append(svms, pics.reshape(pics.shape[0], -1), labels)
            time_in_fit += time()-tstart
            
            print('Round {} of {}: {:.1f}% of the weights was used'.format(f, l.filters, np.mean(sample_weights)*100))
                
            # As the CNN weights go, there is no difference between f(x) and -f(x)...
            # So let's force positive bias, to avoid dead neurons.

            # Actually, now with balanced=True it's possibly redundant. To check whether it should
            # be removed...
            #            raise NameError("To Check: whether forcing the intercept to be positive is still adequate.")
            #            sign = np.sign(svms[-1].intercept_)
            sign = 1
            norm = np.sqrt(np.sum(svms[-1].coef_**2))
            l.weights[0][:,f] = svms[-1].coef_.flatten() / norm * sign
            l.weights[1][f] = svms[-1].intercept_ / norm * sign

        print('\nElapsed {:.3f}s. Of these, {:.3f}s to load pics, {:.3f}s to predict, {:.3f}s to fit.\n'.format(
            time()-global_start, time_in_load, time_in_predict, time_in_fit))
        
        return svms

    def fit_and_append(svms, x, labels):
        '''
        svms is a vector of svm that we consider "already trained".
        x has shape N x K, labels N, as usual in svm. Assume labels is only 0 or 1,
        that we replace to -1 and 1 as in typical SVM.
        '''
        labels = labels*2-1
        sample_weights = np.ones(x.shape[0])

        # If it is not the first filter, give weight < 1 if a previous SVM already
        # classifies it more or less correctly.
        for s in svms:
            sample_weights = np.maximum(np.minimum(sample_weights, 1-np.maximum(s.decision_function(x) * labels, 0)), 0)
            # The following gives some numerical errors sometimes...
            # sample_weights = sample_weights * (1-np.maximum(svms[i].decision_function(reshaped) * svm_labels, 0))

        svms.append(LinearSVC(dual=False))
        svms[-1].fit(x, labels, sample_weight = sample_weights)

        return svms

    def fit_layer_on_all_pics(self, l, epochs=1):
        '''
        Fits a Dense layer by testing it on all the samples, like a usual neural network.
        On each epoch loads as many pics as there are in the folder, but shuffling, so
        actually each pic might be loaded more than once (or never) per epoch.
        '''
        raise NameError('Still to implement fit_Dense_layer')
            

    def load_random_input(self, batch_size, input_shape, normalize = True, balanced = False):
        ''' Loads a random sample of files and assigns them to classes using the subfolder
            names, like Keras does. '''
        folders = [f for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]

        assert len(folders) > 1, 'There are {} folder(s) in {}, how could we do any classification?'.format(len(folders), self.folder)
        assert input_shape[2] in [1, 3], 'Image should either be gray scale (last dim = 1) or BGR (last dim = 3), not {}'.format(input_shape[-1])

        # Select random quantities that total to batch_size
        if balanced:
            quantities = np.array([batch_size // len(folders)] * len(folders))
        else:
            quantities = 0
            while np.any(quantities==0):
                quantities = np.random.multinomial(batch_size, [1/len(folders)] * len(folders))

        # Select some random elements per subfolder, as many as "quantities"
        # Only replace if batch_size is too big
        files = [os.path.join(self.folder, subfol, f) for i,subfol in enumerate(folders) for f in np.random.choice(os.listdir(os.path.join(self.folder, subfol)),
                                                                                                                   quantities[i],
                                                                                                                   quantities[i] > len(os.listdir(os.path.join(self.folder, subfol))))]

        pics = np.zeros((batch_size, *input_shape), dtype=np.uint8)

        color = int(input_shape[2] > 1)

        order = np.random.permutation(list(range(len(files))))
        labels = np.repeat(list(range(len(folders))), quantities)[order]
        
        # Transform from BGR to RGB if in color
        for i, f in enumerate(np.array(files)[order]):
            pics[i] = cv2.resize(cv2.imread(os.path.join(self.folder, f), color)[:,:,::-1], input_shape[:2])

        # Optionally subtract the mean and divide by the std along the first two dimensions
        # (that are the batch dimension and spatial dimension, don't mix different filters/channels though).
        # It should probably be replaced by a "global" normalization instead (with average
        # and std throughout all the pixels of all the pictures), to avoid small batches
        # effects - but that's probably the same in the end.
        if normalize:
            mean = pics.reshape(-1,pics.shape[-1]).mean(axis=0)
            std = pics.reshape(-1,pics.shape[-1]).std(axis=0)
            pics = (pics - mean) / std
            
            
        return pics, labels
