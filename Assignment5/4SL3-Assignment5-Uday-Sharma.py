import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_FILE = 'data_banknote_authentitcation.txt'
RANDOM_SEED = 9248
TRIALS = 3
EPOCHS = 1000































'''
Processess the text file

processData() -> X_train, t_train, X_val, t_val, X_test, t_test

'''
def processData():
    sc = StandardScaler()
    data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
    split = 0.2
    val_split = 0.1

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(data)
    
    

    # split dataset: training, validation, test
    length = data.shape[0]
    train_data = data[:int(length * split)]
    val_data = data[int(length * split):int(length * split) + int(length * val_split)]
    test_data = data[int(length * split) + int(length * val_split):]

    # separate features and labels
    num_features = data.shape[1] - 1
    X_train, t_train = train_data[:, :num_features], train_data[:, num_features]
    X_val, t_val = val_data[:, :num_features], val_data[:, num_features]
    X_test, t_test = test_data[:, :num_features], test_data[:, num_features]

    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    
    t_train = np.expand_dims(t_train, axis=1)
    t_test = np.expand_dims(t_test, axis=1)
    t_val = np.expand_dims(t_val, axis=1)

    return X_train, t_train, X_val, t_val, X_test, t_test