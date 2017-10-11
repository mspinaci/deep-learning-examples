import numpy as np
import pandas as pd
import os, cv2
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
            if isinstance(l, Conv2D):
                self.initialize_layer_by_SVM(l, batch_size=batch_size)

    def initialize_layer_by_SVM(self, l, batch_size=256):
        input_shape = self.layers[0].input_shape

        svms = []

        raise NameError('Problema da risolvere: quando / cosa bisogna centrare e rinormalizzare? Ad ogni step? Perche'' ReLU e` abbastanza selvaggio, tutto e` positivo...')

        l.weights = (np.zeros((l.kernel_size**2 * l.input_shape[3], l.filters)), np.zeros(l.filters))
        
        for f in range(l.filters):
            
            pics, labels = self.load_random_input(batch_size, input_shape[1:], normalize=True)
            if len(set(labels)) != 2:
                raise NameError('For the moment, only 2 classes case is implemented...')

            # For the moment, only do the version "without memory saving"

            # Predict up to l (excluding) and take all the submatrices.
            sm = l.extract_submatrices(self.predict(pics, until_layer=l))
            reshaped = sm.reshape(-1, sm.shape[2])

            # If it is not the first filter, give weight < 1 if a previous SVM already
            # classifies it more or less correctly.
            svm_labels = np.repeat(labels, sm.shape[1])
            svm_labels[svm_labels == 0] = -1
            sample_weights = np.ones(reshaped.shape[0])
            for i in range(len(svms)):
                # The following two alternatives give roughly the same result... but hopefully the first one is more stable
                sample_weights = np.minimum(sample_weights, 1-np.maximum(svms[i].decision_function(reshaped) * svm_labels, 0))
                # sample_weights = sample_weights * (1-np.maximum(svms[i].decision_function(reshaped) * svm_labels, 0))

            print('Round {} of {}: {:.1f}% of the weights used'.format(f, l.filters, np.mean(sample_weights)*100))
                
            svms.append(LinearSVC())
            # Fit the (Linear) support vector machine on all the submatrices; so we need to repeat the label quite a bit.
            svms[-1].fit(reshaped, np.repeat(labels, sm.shape[1]), sample_weight = sample_weights)

            # As the CNN weights go, there is no difference between f(x) and -f(x)...
            # So let's force positive bias, to avoid dead neurons.
            sign = np.sign(svms[-1].intercept_)
            norm = np.sqrt(np.sum(svms[-1].coef_**2))
            l.weights[0][:,f] = svms[-1].coef_.flatten() / norm * sign
            l.weights[1][f] = svms[-1].intercept_ / norm * sign
            


    def load_random_input(self, batch_size, input_shape, normalize = True):
        ''' Loads a random sample of files and assigns them to classes using the subfolder
            names, like Keras does. '''
        folders = [f for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]

        assert len(folders) > 1, 'There are {} folder(s) in {}, how could we do any classification?'.format(len(folders), self.folder)
        assert input_shape[2] in [1, 3], 'Image should either be gray scale (last dim = 1) or BGR (last dim = 3), not {}'.format(input_shape[-1])

        # Select random quantities that total to batch_size
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
