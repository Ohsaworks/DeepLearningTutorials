import scipy.io as spio
import pickle
import os
import sys

def pickle_dump(data, path):
    with open(path,'w') as outfile:
        pickle.dump(data, outfile)
    print 'Pickle file size: ', os.path.getsize(path)/1000, ' kB'

path_train = sys.argv[1]
path_test = sys.argv[2]
path_pickle = sys.argv[3]

print 'Original train file size: ', os.path.getsize(path_train)/1000, ' kB'
print 'Original test file size: ', os.path.getsize(path_test)/1000, ' kB'

print "Loading Train Data"
data_train = spio.loadmat(path_train)
print "Loading Test Data"
data_test = spio.loadmat(path_test)

def flatten(X):
    num_samples = len(X[0][0][0])
    ret_X = [[0]*32*32*3 for i in xrange(0, num_samples)]
    idx = 0
    for row in X:
        for cell in row:
            for channel in cell:
                for i in xrange(0, num_samples):
                    ret_X[i][idx] = channel[i]
                idx += 1;
    return ret_X

def summerize_x(label, x):
    print label
    print len(x)
    print len(x[0])

def summerize_y(label, y):
    print label
    print len(y)


print "Flatting Train Data"
train_X_flatten = flatten(data_train["X"])
print "Flatting Test Data"
test_X_flatten = flatten(data_test["X"])

num_train_real = len(train_X_flatten) * 4 / 5
print "num_train: ", len(train_X_flatten)
print "num_train_real: ", num_train_real

print "Reformatting Data"
train_set_x = train_X_flatten[:num_train_real]
valid_set_x = train_X_flatten[num_train_real:]
test_set_x = test_X_flatten

summerize_x("tr", train_set_x)
summerize_x("vl", valid_set_x)
summerize_x("ts", test_set_x)

train_set_y = [label[0] for label in data_train["y"][:num_train_real]]
valid_set_y = [label[0] for label in data_train["y"][num_train_real:]]
test_set_y  = [label[0] for label in data_test["y"]]

summerize_y("tr", train_set_y)
summerize_y("vl", valid_set_y)
summerize_y("ts", test_set_y)

overall_output = ((train_set_x, train_set_y), (valid_set_x, valid_set_y), \
(test_set_x, test_set_y))

print "Saving Data"
pickle_dump(overall_output, path_pickle)
