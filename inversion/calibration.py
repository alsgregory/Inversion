from inversion import *

class calibrate:
    
    def __init__(self, priorPPF, sigmaY):
		
        # prior quantile functions - returns N draws of theta from prior
        self.priorPPF = priorPPF
		
        # measurement error standard deviation
        self.sigmaY = sigmaY
	
    def normal_prior(self, means, sds):
	
        # demo of prior quantile function

        cov = np.diag(sds ** 2)
		
        return(norm.ppf(np.random.uniform(0, 1), means, cov))
	
    def __drawFromPrior(self, N=1):
	    
        draws = []
        for i in range(N):
            draws.append(list(self.priorPPF()))
		
        return(np.array(draws))
	
    def metropolisHastings(self, niter, beta, logConstraint=None, burn=1):
	    
        # temperature=0 used for mcmc via metropolis hastings
		
        self.posteriorSamples = self.simAnnealing(0, niter, beta, logConstraint, burn)
	
    def simAnnealing(self, T, niter, beta, logConstraint=None, burn=1):
	    
        # simulated annealing with temperature plan alpha^(T * i)
        # log constraint can be array of booleans, signalling a component of theta (e.g. lengthscale) needs to be lnorm
		
        theta = np.zeros((niter + 1, np.shape(self.__drawFromPrior())[1]))
        theta[0, :] = self.__drawFromPrior()
		
        for i in range(1, (niter + 1)):
		    
            if len(logConstraint) == 0:
                prop = theta[i - 1, :] + np.random.normal(0, beta, len(theta[i - 1, :]))
            else:
                prop = np.zeros(len(logConstraint))
                prop[np.where(logConstraint == 1)[0].astype(int)] = np.exp(np.log(theta[i - 1, np.where(logConstraint == 1)[0].astype(int)]) + np.random.normal(0, beta, len(theta[i - 1, np.where(logConstraint == 1)[0].astype(int)])))
                prop[np.where(logConstraint == 0)[0].astype(int)] = theta[i - 1, np.where(logConstraint == 0)[0].astype(int)] + np.random.normal(0, beta, len(theta[i - 1, np.where(logConstraint == 0)[0].astype(int)]))
			
            alpha = utils.LikelihoodRatio(self.xModel, self.xData,
                                          self.yModel, self.yData,
                                          self.tModel, theta[i - 1, :], prop,
                                          self.sigmaY)
			
            accept = np.log(np.random.uniform(0, 1)) <= np.min([0, ((1 / np.exp(-(T * ((i + 1) / niter)))) * alpha)])
			
            if accept:
                theta[i, :] = prop
            else:
                theta[i, :] = theta[i - 1, :]
		
        return(theta[burn:, :])
	
    def updateCoordinates(self, xModel, xData):
	    
        # setup training data - xmodel and xdata are dimensions N x d arrays
        # N being number of input sets
        # d being the dimension of each input set
        self.xModel = xModel
        self.xData = xData

    def updateTrainingData(self, tModel, yModel, yData):
	
        # update the data used to train the calibration
        self.tModel = tModel
        self.yModel = yModel
        self.yData = yData
		
        # assertions
        assert(np.shape(self.yModel)[0] == np.shape(self.tModel)[0])
        assert(np.shape(self.yModel)[1] == len(self.xModel))
        assert(np.shape(self.yData)[1] == len(self.xData))
		
    def sequentialUpdate(self, N, beta, logConstraint=None):
	
        # carries out a sequential calibration update, using previous self.posteriorSamples as the prior
        if hasattr(self, 'posteriorSamples'):
            # use previous self.posteriorSamples
            particles = self.posteriorSamples
        else:
            # use prior samples
            particles = self.__drawFromPrior(N)

        # find marginal likelihood of each particle
        ml = np.zeros(np.shape(particles)[0])
		
        for i in range(np.shape(particles)[0]):
		    
            gp = utils.fitGP(self.xModel, self.xData,
                             self.yModel, self.yData,
                             self.tModel, particles[i, :], self.sigmaY)
		    
            ml[i] = np.exp(gp.log_marginal_likelihood_value_)
		
        # resample
        w = ml / np.sum(ml)
        inds = np.random.choice(np.linspace(0, np.shape(particles)[0] - 1, np.shape(particles)[0]),
                                N, p = w).astype(int)
        self.posteriorSamples = particles[inds, :]
		
        # rejuvinate variation in ensemble
        if len(logConstraint) == 0:
            self.posteriorSamples = self.posteriorSamples + np.random.normal(0, beta, np.shape(self.posteriorSamples))
        else:
            self.posteriorSamples[:, np.where(logConstraint == 1)[0].astype(int)] = np.exp(np.log(self.posteriorSamples[:, np.where(logConstraint == 1)[0].astype(int)]) + np.random.normal(0, beta, np.shape(self.posteriorSamples[:, np.where(logConstraint == 1)[0].astype(int)])))
            self.posteriorSamples[:, np.where(logConstraint == 0)[0].astype(int)] = self.posteriorSamples[:, np.where(logConstraint == 0)[0].astype(int)] + np.random.normal(0, beta, np.shape(self.posteriorSamples[:, np.where(logConstraint == 0)[0].astype(int)]))
			
			