import numpy as np
import math

'''
ID_1: 342681004
ID_2: 026548446
'''

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        '''
        The total sum of probabilities has to be equal to 1 for the table to be valid. 
        Moreover, the probability of P(X=0) = 0.3 = P(X=0, Y=0) + P(X=0,Y=1) and obviously the same goes for any value of X or Y
        '''
        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.4,
            (1, 1): 0.3
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.4,
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        # Reminder: P(X=x, Y=y, C=c) = P(X=x, C=c) * P(Y=y, C=c) / P(C=c) 
        self.X_Y_C = {
            (0, 0, 0): 0.02,
            (0, 0, 1): 0.08,
            (0, 1, 0): 0.08,
            (0, 1, 1): 0.12,
            (1, 0, 0): 0.08,
            (1, 0, 1): 0.12,
            (1, 1, 0): 0.32,
            (1, 1, 1): 0.18,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for (a,b), p_xy in X_Y.items():
            p_x = X[a]
            p_y = Y[b]
            if not np.isclose(p_x*p_y, p_xy):
                return True
        return False
       

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C

        X_C = self.X_C # P(X=x, C=c)
        Y_C = self.Y_C # P(Y=y, C=c)
        X_Y_C = self.X_Y_C # P(X=x, Y=y, C=c) = P(X=x, C=c) * P(Y=y, C=c) / P(C=c)

        for c in C.keys():
            p_c = C[c]

            for x in X.keys():
                for y in Y.keys():

                    p_x_given_c = X_C[(x, c)] / p_c
                    p_y_given_c = Y_C[(y, c)] / p_c
                    p_xy_given_c = X_Y_C[(x, y, c)] / p_c

                    if not np.isclose(p_xy_given_c, p_x_given_c * p_y_given_c):
                        return False
        return True

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    # Correct but it doesn't have numerical stability
    # log_p = np.log((rate**k)*np.exp(-rate)/math.factorial(k))
    
    log_k_factorial = np.sum(np.log(np.arange(1,k+1))) if k>0 else 0.0
    log_p = k * np.log(rate) - rate - log_k_factorial
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    Simplified formula: -n * lambda - sum(ln(x_j!)) + ln(lambda) * sum(x_j)

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    samples = np.asarray(samples)
    rates = np.asarray(rates)

    n = len(samples)
    samples_summed = np.sum(samples)

    log_factorial = np.array([
        np.sum(np.log(np.arange(1,k+1))) if k>0 else 0.0 
        for k in samples
    ])

    sum_log_factorials = np.sum(log_factorial)

    likelihoods = - n * rates - sum_log_factorials + samples_summed * np.log(rates)
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    idx = np.argmax(likelihoods)
    rate = rates[idx]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.mean(samples)
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    const = 1 / (std * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    p = const * exponent
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.data =  dataset[dataset[:, -1] == class_value][:,:-1] # feature matrix for class value
        self.class_value = class_value
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = np.sum(self.dataset[:,-1] == self.class_value) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood porbability of the instance under the class according to the dataset distribution.
        """
        feature_likelihood = normal_pdf(x,self.mean,self.std)
        likelihood = np.prod(feature_likelihood)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior()* self.get_instance_likelihood(x) # since we ignore p(x)
        
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        post_0 = self.class0.get_instance_posterior(x)
        post_1 = self.class1.get_instance_posterior(x)
        pred = 0 if post_0 > post_1 else 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """

    x = test_set[:,:-1]
    y = test_set[:,-1]
    y_pred = np.array([map_classifier.predict(i) for i in x ])
    acc = np.mean(y == y_pred)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    d = mean.shape[0]
    const = (2 * np.pi)**(-d/2) * np.linalg.det(cov)**(-0.5)
    exponent = -0.5*(x-mean).T @ np.linalg.inv(cov) @ (x-mean)
    pdf = const*np.exp(exponent)
    
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        c = dataset[dataset[:,-1] == class_value][:,:-1]
        self.mean = c.mean(axis=0)
        self.cov  = np.cov(c.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = np.sum(self.dataset[:,-1] == self.class_value) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x,self.mean,self.cov)
        #likelihood = np.prod(feature_likelihood)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior()* self.get_instance_likelihood(x) # since we ignore p(x)
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        prior_0 = self.class0.get_prior()
        prior_1 = self.class1.get_prior()
        pred = 0 if prior_0 > prior_1 else 1  
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        likeli_0 = self.class0.get_instance_likelihood(x)
        likeli_1 = self.class1.get_instance_likelihood(x)
        pred = 0 if likeli_0 > likeli_1 else 1  
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.data = dataset[dataset[:, -1] == class_value][:, :-1]
        self.n_i = len(self.data)
        # Number of unique values per feature (from the entire dataset)
        self.V = np.array([len(np.unique(dataset[:, j])) for j in range(dataset.shape[1] - 1)])
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        prior = self.n_i / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = 1.0
        for j in range(len(x)):
            # Count how many training instances of this class have x[j] for feature j
            n_ij = np.sum(self.data[:, j] == x[j])
            if n_ij == 0 and x[j] not in self.data[:, j]:
                # Value only appears in test set, not in training data for this class
                # Check if it appears anywhere in the full dataset for this feature
                if x[j] not in self.dataset[:, j]:
                    likelihood *= EPSILLON
                else:
                    likelihood *= (0 + 1) / (self.n_i + self.V[j])
            else:
                likelihood *= (n_ij + 1) / (self.n_i + self.V[j])
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.class0 = ccd0
        self.class1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        post_0 = self.class0.get_instance_posterior(x)
        post_1 = self.class1.get_instance_posterior(x)
        pred = 0 if post_0 > post_1 else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        X = test_set[:, :-1]
        y = test_set[:, -1]
        y_pred = np.array([self.predict(xi) for xi in X])
        acc = np.mean(y == y_pred)
        return acc

