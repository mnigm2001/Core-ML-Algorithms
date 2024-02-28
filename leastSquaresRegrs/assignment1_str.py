
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


def train_model(poly, x_vec, t_vec, with_reg, reg_term=2, reg_lambda=None, std_scaler=None):
    if with_reg:
        x_vec = std_scaler.fit_transform(x_vec)                             # Standaridize the features
        x_poly = poly.fit_transform(x_vec)                              # Transform the feature matrix

        if reg_term == 1:
            model = Ridge(alpha=reg_lambda).fit(x_poly, t_vec)
        elif reg_term == 2:
            model = Lasso(alpha=reg_lambda).fit(x_poly, t_vec)

        return model

    x_poly = poly.fit_transform(x_vec) 
    model = LinearRegression().fit(x_poly, t_vec)                        # Create and train the model

    return model

def predict_values(poly, x_vec, model):
    x_pred_poly = poly.transform(x_vec)                                        # Transform the x-values for prediction
    y_pred = model.predict(x_pred_poly).reshape(-1, 1)                         # Predict y-values
    return y_pred

def get_error(true_vals, pred_vals, set_size):
    err_vec = np.square(true_vals - pred_vals)                                 # training error vector for plotting
    acm_err = (np.add.reduce(err_vec))/set_size
    # print(f'err_vec = {err_vec}')
    # print(f'acm_err = {acm_err}')

    return err_vec, acm_err


def create_X(degree, x_vec):
    X = np.ones((len(x_vec), 0))

    # Add the feature values and its powers
    for i in range(degree + 1):
        X = np.concatenate((X, x_vec ** i), axis=1)
    
    return X




def get_err(matrix_X, w_vec, t_vec, set_size):
    term1 = np.subtract(np.dot(matrix_X, w_vec), t_vec)
    # print(f'term1 ({term1.shape})')
    return (1/set_size)*np.dot(np.transpose(term1), term1)


########### --------------- Personal Implementation 
def get_w(matrix_X, t_vec):
    x_trans = np.transpose(matrix_X)
    return np.dot(np.linalg.inv(np.dot(x_trans, matrix_X)), np.dot(x_trans, t_vec))

def get_w_reg(matrix_X, t_vec, m_val, reg_lambda, set_size):
    x_trans = np.transpose(matrix_X)
    diag_elements = np.full(shape=m_val+1, fill_value=2*reg_lambda)
    matrix_B = np.diag(diag_elements)
    return np.dot(np.linalg.inv(np.dot(x_trans, matrix_X) + (set_size/2) * matrix_B), np.dot(x_trans, t_vec))
    

def main():
    '''
    
    '''
    train_errs = {}
    valid_errs = {}
    t_errs = {}
    
    reg_term = 2    # For choosing L1 or L2 regularization
    reg_lambda = 1e-7

    train_set_size = 12
    valid_set_size = 120
        
    scaler = StandardScaler()
    
    for m_val in range(0, 12, 1):

        print(f'\n{30*"-"} m = {m_val} {30*"-"}')

        x_train_vec, t_train_vec, x_valid_vec, t_valid_vec, t_true_vec = gen_examples(train_set_size, valid_set_size)
        _, err_t = get_error(t_true_vec, t_valid_vec, valid_set_size)       # fM vs ftrue error
        t_errs[m_val] = err_t


        ###
        poly = PolynomialFeatures(degree=m_val)
        # if m_val == 11:
        #     model = train_model(poly, x_train_vec, t_train_vec, with_reg=True, reg_term=reg_term, reg_lambda=reg_lambda, std_scaler=scaler)
        # else:
        #     model = train_model(poly, x_train_vec, t_train_vec, with_reg=False)
        
        #### ---- Equation Based Solution START ---- ####

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
        train_err_man = get_err(matrix_X_train, w_vec_train, t_train_vec, train_set_size)
        train_errs[m_val] = train_err_man
        y_pred = np.dot(matrix_X_train, w_vec_train)

        ## validation err
        valid_err_man = get_err(matrix_X_valid, w_vec_valid, t_valid_vec, valid_set_size)
        valid_errs[m_val] = valid_err_man

        # if m_val != 11:
        #     ## -- training
        #     matrix_X = create_X(m_val, x_train_vec)
        #     w_vec = get_w(matrix_X, t_train_vec)
        #     train_err_man = get_err(matrix_X, w_vec, t_train_vec, train_set_size)
        #     train_errs[m_val] = train_err_man
        #     y_pred = np.dot(matrix_X, w_vec)

        #     ## -- validation
        #     matrix_X = create_X(m_val, x_valid_vec)
        #     w_vec = get_w(matrix_X, t_valid_vec)
        #     valid_err_man = get_err(matrix_X, w_vec, t_valid_vec, valid_set_size)
        #     valid_errs[m_val] = valid_err_man
        # else:
        #     matrix_X = create_X(m_val, x_train_vec)
        #     w_vec = get_w_reg(matrix_X, t_train_vec, m_val, reg_lambda, train_set_size)
        #     train_err_man = get_err(matrix_X, w_vec, t_train_vec, train_set_size)
        #     train_errs[m_val] = train_err_man
        #     y_pred = np.dot(matrix_X, w_vec)

        #     matrix_X = create_X(m_val, x_valid_vec)
        #     w_vec = get_w_reg(matrix_X, t_valid_vec, m_val, reg_lambda, valid_set_size)
        #     valid_err_man = get_err(matrix_X, w_vec, t_valid_vec, valid_set_size)
        #     valid_errs[m_val] = valid_err_man




        #### ---- Equation Based Solution END ---- ####



        # y_pred = predict_values(poly, x_train_vec, model)
        # if m_val == 11:
        #     x_valid_vec = scaler.transform(x_valid_vec)                             # Standaridize the features
        # y_pred_valid = predict_values(poly, x_valid_vec, model)

        # err_train_vec, err_train = get_error(t_train_vec, y_pred, train_set_size)
        # err_valid_vec, err_valid = get_error(t_valid_vec, y_pred_valid, valid_set_size)
        # print(f'err_train = {err_train}')
        # print(f'err_valid = {err_valid}')
        # train_errs[m_val] = err_train
        # valid_errs[m_val] = err_valid


        '''

        ###

        ##################### -- Training START -- #####################
        poly = PolynomialFeatures(degree=m_val)
        scaler = StandardScaler()

        
        if m_val == 11:
            x_train_vec = scaler.fit_transform(x_train_vec)                             # Standaridize the features
                
        x_train_poly = poly.fit_transform(x_train_vec)                                  # Transform the feature matrix

        if m_val == 11:
            if reg_term == 1:
                model = Ridge(alpha=reg_lambda).fit(x_train_poly, t_train_vec)
            if reg_term == 2:
                model = Lasso(alpha=reg_lambda).fit(x_train_poly, t_train_vec)
        else:
            model = LinearRegression().fit(x_train_poly, t_train_vec)                       # Create and train the model
        
        x_pred_poly = poly.transform(x_train_vec)                                            # Transform the x-values for prediction
        print(f' x_pred_poly = {x_pred_poly}')
        y_pred = model.predict(x_pred_poly).reshape(-1, 1)                                             # Predict y-values
        print(f'y_pred = {y_pred}')

        err_train_vec = np.square(t_train_vec - y_pred)                                            # training error vector for plotting
        err_train = np.add.reduce(err_train_vec)/train_set_size
        train_errs[m_val] = err_train
        print(f'for m = {m_val} training error is {err_train}')
        ##################### -- Training END -- #####################


        ##################### -- Validation START -- #####################

        if m_val == 11:
            x_valid_vec = scaler.transform(x_valid_vec)                             # Standaridize the features
        
        x_pred_poly_valid = poly.transform(x_valid_vec)                                            # Transform the x-values for prediction
        y_pred_valid = model.predict(x_pred_poly_valid).reshape(-1, 1)                                             # Predict y-values

        err_valid_vec = np.square(t_valid_vec - y_pred_valid)                                     
        err_valid = np.add.reduce(err_valid_vec)/valid_set_size
        valid_errs[m_val] = err_valid
        print(f'for m = {m_val} validation error is {err_valid}')

        ##################### -- Validation END -- #####################

        
        '''

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

    
    plt.title(f" Training and Validation Errors vs M-Values")
    plt.xlabel("M")
    plt.ylabel("Error")
    plt.ylim([-0.25, 2.8])
    plt.legend(loc='upper left')
    plt.savefig(f'./plots/err_vs_m_reg_L{reg_term}_lambda{reg_lambda}.png')



    


    

if __name__ == '__main__':
    main()

'''
PLOT 1
1) DONE For each M you have to train the model using least squares and record the training and validation
errors.
2) DONE For each M, plot the prediction fM(x) function and the curve ftrue(x) versus x ∈ [0, 1]
3) DONE Additionally, include in the figure all the points in the training set and in the validation set with their true target values

PLOT 2
4) In addition, plot the training and validation errors versus M
5) Also include in this plot the average squared error between the targets and the true function ftrue(x)
for the examples in the validation set (this will be a horizontal line)

REPORT: What does this value represent?


'''








'''


# plt.figure()
        # [4] In addition, plot the training and validation errors versus M


        # plt.scatter(x_train_vec, y_pred, color='green', label=f'train error')
        # plt.scatter(x_train_vec, err_train_vec, color='green', label=f'train error')

        # plt.scatter(x_valid_vec, y_pred_valid, color='yellow', label='valid pred')
        # plt.scatter(x_valid_vec, err_valid_vec, color='black', label=f'valid error')

        # plt.title(f" Model functions for M = {m_val}")
        # # plt.xlabel("")
        # # plt.ylabel("")
        # plt.legend(loc='lower right')
        # plt.savefig('train.png')
'''

# def get_w(matrix_X, t_vec):
#     x_trans = np.transpose(matrix_X)
#     # print(f'x shape = {matrix_X.shape}')
#     # print(f't shape = {t_vec.shape}')
#     # print(f'xT shape = {x_trans.shape}')

#     dot1 = np.dot(x_trans, matrix_X)
#     # print(f'dot1 shape ({dot1.shape}) = {dot1}')

#     dot2 = np.dot(x_trans, t_vec)
#     # print(f'dot2 shape ({dot2.shape}) = {dot2}')

#     inv = np.linalg.inv(dot1)
#     # print(f'inv ({inv.shape}) = {inv}')

#     dot3 = np.dot(inv, dot2)
#     # print(f'dot3 shape ({dot3.shape}) = {dot3}')

#     return np.dot(inv, dot2)
#     # print('[1]', np.dot(np.dot(np.linalg.inv(np.dot(matrix_X, x_trans)), x_trans), t_vec))
#     # print('[1]', np.dot(np.linalg.inv(np.dot(matrix_X, x_trans)), np.dot(x_trans, t_vec)))
#     # return np.dot(np.linalg.inv(np.dot(matrix_X, x_trans)), np.dot(x_trans, t_vec))


# def get_w_reg(matrix_X, t_vec, m_val, reg_lambda, set_size):
#     x_trans = np.transpose(matrix_X)


#     # Create a 1D array of diagonal elements
#     diag_elements = np.full(shape=m_val+1, fill_value=2*reg_lambda)
#     matrix_B = np.diag(diag_elements)
#     print(f'matrix_B shape = {matrix_B.shape}')
    
#     x_xt = np.dot(x_trans, matrix_X)
#     print(f'x_xt shape = {x_xt.shape}')

#     mul1 = (set_size/2) * matrix_B
#     add1 = x_xt + mul1
#     print(f'add1 shape = {add1.shape}')
    
#     inv = np.linalg.inv(add1)
#     print(f'inv shape = {inv.shape}')

#     dot2 = np.dot(x_trans, t_vec)
#     print(f'dot2 shape = {dot2.shape}')

#     dot3 = np.dot(inv, dot2)
#     print(f'dot3 shape = {dot3.shape}')

#     return dot3