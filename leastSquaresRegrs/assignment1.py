
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RN_SEED = 7484


def gen_x_vec(set_size):
    return np.linspace(0, 1, set_size)

def gen_ftrue(x_vec):
    return np.sin(4 * np.pi * x_vec + np.pi / 2)

def gen_y_vec(set_size, x_vec):
    ## Gaussian distribution with mean 0 and std dev of 0.0625
    np.random.seed(RN_SEED)
    noise = np.random.normal(0, np.sqrt(0.0625), size=x_vec.size)
    return gen_ftrue(x_vec) + noise


def gen_examples(train_set_size, valid_set_size):
    ## -- Training Set -- ##
    x_train_vec = gen_x_vec(train_set_size)
    t_train_vec = gen_y_vec(train_set_size, x_train_vec)

    x_train_vec = x_train_vec.reshape(-1, 1)
    t_train_vec = t_train_vec.reshape(-1, 1)

    ## -- Validation Set -- ##
    x_valid_vec = gen_x_vec(valid_set_size)
    t_valid_vec = gen_y_vec(valid_set_size, x_valid_vec)
    t_true_vec = gen_ftrue(x_valid_vec)

    x_valid_vec = x_valid_vec.reshape(-1, 1)
    t_valid_vec = t_valid_vec.reshape(-1, 1)
    t_true_vec = t_true_vec.reshape(-1, 1)

    return x_train_vec, t_train_vec, x_valid_vec, t_valid_vec, t_true_vec



########### --------------- Training and Error Functions --------------- ###########
def create_X(degree, x_vec):
    X = np.ones((len(x_vec), 0))

    # Add the feature values and its powers
    for i in range(degree + 1):
        X = np.concatenate((X, x_vec ** i), axis=1)
    
    return X

def get_w(matrix_X, t_vec):
    x_trans = np.transpose(matrix_X)
    return np.dot(np.linalg.inv(np.dot(x_trans, matrix_X)), np.dot(x_trans, t_vec))

def get_w_reg(matrix_X, t_vec, m_val, reg_lambda, set_size):
    x_trans = np.transpose(matrix_X)
    diag_elements = np.full(shape=m_val+1, fill_value=2*reg_lambda)
    matrix_B = np.diag(diag_elements)
    return np.dot(np.linalg.inv(np.dot(x_trans, matrix_X) + (set_size/2) * matrix_B), np.dot(x_trans, t_vec))

def get_error(true_vals, pred_vals, set_size):
    err_vec = np.square(true_vals - pred_vals)                                 # training error vector for plotting
    acm_err = (np.add.reduce(err_vec))/set_size
    return err_vec, acm_err

def get_err(matrix_X, w_vec, t_vec, set_size):
    term1 = np.subtract(np.dot(matrix_X, w_vec), t_vec)
    # print(f'term1 ({term1.shape})')
    return (1/set_size)*np.dot(np.transpose(term1), term1)


########### --------------- Assignment Functions --------------- ###########
def part1():
    '''
    
    '''
    train_errs = {}
    valid_errs = {}
    t_errs = {}
    
    reg_lambda = 1e-7

    train_set_size = 12
    valid_set_size = 120
        
    scaler = StandardScaler()
    
    for m_val in range(0, 12, 1):

        # print(f'\n{30*"-"} m = {m_val} {30*"-"}')

        x_train_vec, t_train_vec, x_valid_vec, t_valid_vec, t_true_vec = gen_examples(train_set_size, valid_set_size)
        _, err_t = get_error(t_true_vec, t_valid_vec, valid_set_size)       # fM vs ftrue error
        t_errs[m_val] = err_t

        if m_val == 11:
            x_train_vec = scaler.fit_transform(x_train_vec)                             # Standaridize the features
            x_valid_vec = scaler.transform(x_valid_vec)                             # Standaridize the features

        matrix_X_train = create_X(m_val, x_train_vec)
        matrix_X_valid = create_X(m_val, x_valid_vec)

        if m_val == 11:
            w_vec_train = get_w_reg(matrix_X_train, t_train_vec, m_val, reg_lambda, train_set_size)
            w_vec_valid = get_w_reg(matrix_X_valid, t_valid_vec, m_val, reg_lambda, valid_set_size)
        else:
            w_vec_train = get_w(matrix_X_train, t_train_vec)
            w_vec_valid = get_w(matrix_X_valid, t_valid_vec)
        
        ## training err
        train_err = get_err(matrix_X_train, w_vec_train, t_train_vec, train_set_size)
        train_errs[m_val] = train_err
        y_pred = np.dot(matrix_X_train, w_vec_train)

        ## validation err
        valid_err = get_err(matrix_X_valid, w_vec_valid, t_valid_vec, valid_set_size)
        valid_errs[m_val] = valid_err

    
        ##################### -- Plotting START -- #####################
        plt.figure()
        # [2] fM and ftrue
        plt.plot(x_train_vec, y_pred, color='red', label='fM(x)')                               # prediction line fM
        plt.plot(x_valid_vec, t_true_vec, color='green', label='ftrue(x)')                      # true values ftrue                                   
        
        # [3] training set points, validation set points
        plt.scatter(x_train_vec, t_train_vec, color='blue', label='training')                   # training examples
        plt.scatter(x_valid_vec, t_valid_vec, color='black', label='valid ex')                  # validation examples

        plt.title(f" Polynomial Predictor for M = {m_val}")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.legend(loc='lower right')
        plt.savefig(f'./plots/fig1_m{m_val}.png')

        ##################### -- Plotting END -- #####################
    
    plt.figure()
    # [4] In addition, plot the training and validation errors versus M
    train_lists = sorted(train_errs.items())  
    m_vals, terr_vals = zip(*train_lists)
    # print(f'terr_vals = {terr_vals}')
    plt.scatter(m_vals, terr_vals, color='blue', label='train err')

    valid_lists = sorted(valid_errs.items())  
    m_vals, verr_vals = zip(*valid_lists)
    # print(f'verr_vals = {verr_vals}')
    plt.scatter(m_vals, verr_vals, color='red', label='valid err')

    # [5] Also include in this plot the average squared error between the targets and the true function ftrue(x)
    # for the examples in the validation set (this will be a horizontal line)
    
    # print(f't_errs = {t_errs}')
    t_lists = sorted(t_errs.items())  
    m_vals, fterr_vals = zip(*t_lists)
    plt.plot(m_vals, fterr_vals, color='green', label='t vs ftrue err')

    
    plt.title(f" Training and Validation Errors vs M-Values for lambda of {reg_lambda}")
    plt.xlabel("M")
    plt.ylabel("Error")
    plt.ylim([-0.25, 2.8])
    plt.legend(loc='upper left')
    plt.savefig(f'./plots/err_vs_m_reg_L2_lambda{reg_lambda}.png')



    
def part2():
    '''
    
    '''
    train_errs = {}
    valid_errs = {}
    t_errs = {}

    train_set_size = 12
    valid_set_size = 120
        
    scaler = StandardScaler()
    
    m_val = 11
    lambda_vec = np.logspace(-100, 0, 1000)
    for lambda_ in lambda_vec:

        x_train_vec, t_train_vec, x_valid_vec, t_valid_vec, t_true_vec = gen_examples(train_set_size, valid_set_size)
        _, err_t = get_error(t_true_vec, t_valid_vec, valid_set_size)       # fM vs ftrue error
        t_errs[m_val] = err_t

        if m_val == 11:
            x_train_vec = scaler.fit_transform(x_train_vec)                             # Standaridize the features
            x_valid_vec = scaler.transform(x_valid_vec)                             # Standaridize the features

        matrix_X_train = create_X(m_val, x_train_vec)
        matrix_X_valid = create_X(m_val, x_valid_vec)

        w_vec_train = get_w_reg(matrix_X_train, t_train_vec, m_val, lambda_, train_set_size)
        w_vec_valid = get_w_reg(matrix_X_valid, t_valid_vec, m_val, lambda_, valid_set_size)
        
        ## training err
        train_err = get_err(matrix_X_train, w_vec_train, t_train_vec, train_set_size)
        train_errs[lambda_] = train_err
        y_pred = np.dot(matrix_X_train, w_vec_train)

        ## validation err
        valid_err = get_err(matrix_X_valid, w_vec_valid, t_valid_vec, valid_set_size)
        valid_errs[lambda_] = valid_err


    plt.figure()
    train_lists = sorted(train_errs.items())  
    lambda_vals, terr_vals = zip(*train_lists)
    terr_vals = [array.item() for array in terr_vals]
    plt.plot(lambda_vals, terr_vals, color='blue', label='train err')

    valid_lists = sorted(valid_errs.items())  
    lambda_vals, verr_vals = zip(*valid_lists)
    verr_vals = [array.item() for array in verr_vals]
    plt.plot(lambda_vals, verr_vals, color='red', label='valid err')

    plt.title(f"Training and Validation Errors vs Lambda")
    plt.xlabel("lamda")
    plt.ylabel("Error")
    plt.legend(loc='upper left')
    plt.savefig('./plots/lambda_vs_errs_m11.png')

    

if __name__ == '__main__':
    part1()
    part2()


