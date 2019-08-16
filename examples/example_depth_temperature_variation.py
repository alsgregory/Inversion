from inversion import *
from mpl_toolkits.mplot3d import Axes3D

# model and data generating process - with temporally varying effects
def model(x, delta_t, omega, k=1):
    y = delta_t * (np.exp(-(np.sqrt(omega / (2 * k))) * x))
    return(y)

def data_generating_process(x, delta_t, omega, sigmaY):
    return(model(x, delta_t, omega) + np.random.normal(0, sigmaY, len(x)))

# params
ts = np.linspace(1, 40, 40)
true_omega = 5 + (1 * np.sin(ts * math.pi * 2 / 20))
true_delta_t = 10
sigmaY = 1.0

# coordinates
ns = 5
xData = np.linspace(0, 1, ns)
xModel = np.linspace(0, 1, ns)

# calibration parameters
n = 10
tModel = np.array([np.random.normal(6, 3, n),
                   np.random.normal(8, 2, n)]).T

# model outputs
yModel = np.zeros((n, len(xModel)))
for i in range(n):
    yModel[i, :] = model(xModel, *tModel[i, :])

# data outputs
m = len(ts)
yData = np.zeros((m, len(xData)))
for i in range(m):
    yData[i, :] = data_generating_process(xData, true_omega[i], true_delta_t, sigmaY)

# define prior for shift and lengthscale
def priorPPF():
    u = np.random.uniform(0, 1, 3)
    omega = norm.ppf(u[0], 8, 6)
    delta_t = norm.ppf(u[1], 8, 2)
    l = np.exp(norm.ppf(u[2], 0, 1))
    return(np.array([omega, delta_t, l]))

### implement sequential calibration

# regularization
nugget = 0.5
lambda_e = 1.0

# initialize class
cal = calibration.calibrate(priorPPF, sigmaY, nugget, lambda_e)

# load coordinates and data
cal.updateCoordinates(xModel, xData)

# particle filter over data outputs
nparticles = 400
beta = 0.5
posteriors = np.zeros((m, nparticles))
for i in range(m):
    cal.updateTrainingData(tModel, yModel, np.reshape(yData[i, :], ((1, ns))))
    cal.sequentialUpdate(nparticles, beta, logConstraint=np.array([0, 1]))
    posteriors[i, :] = cal.posteriorSamples[:, 0]
    print('filtered in ' + str(i) + ' observation sets')

# posterior
plot.plot(cal.posteriorSamples[:, 0])
plot.plot(true_omega[-1] * np.ones(nparticles), '--')
plot.xlabel("sample")
plot.ylabel("omega parameter")
plot.show()

# posterior
plot.plot(cal.posteriorSamples[:, 1])
plot.plot(np.ones(nparticles) * true_delta_t, '--')
plot.xlabel("sample")
plot.ylabel("delta t parameter")
plot.show()

# posterior
plot.plot(cal.posteriorSamples[:, 2])
plot.xlabel("sample")
plot.ylabel("lengthscale parameter")
plot.show()

# posterior over time
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')
nbins = 30
for z in range(m):
    hist, edges = np.histogram(posteriors[z, :], bins=nbins)
    c = 'r'
    xs = edges[:-1]
    ax.bar(xs, hist, width=np.diff(edges), zs=z, zdir='y', color=[0.1, 1 - np.sqrt(z / m), np.sqrt(z / m)], alpha=np.sqrt(z / m))

ax.set_xlabel('omega parameter')
ax.set_ylabel('iteration')
ax.set_zlabel('frequency density')
plot.show()

plot.plot(np.mean(posteriors, axis=1))
plot.plot(true_omega, '--')
plot.xlabel("iteration")
plot.ylabel("omega parameter posterior mean")
plot.show()

### implement MCMC calibration

# initialize class
calmcmc = calibration.calibrate(priorPPF, sigmaY, nugget)

# load coordinates and data
calmcmc.updateCoordinates(xModel, xData)
calmcmc.updateTrainingData(tModel, yModel, yData)

# mcmc
niter = 500
burn = 50
beta = 0.0
calmcmc.metropolisHastings(niter, beta, logConstraint=np.array([0, 0, 1]), burn=burn)

posterior = calmcmc.posteriorSamples[-nparticles:, 0]

# posterior
plot.plot(calmcmc.posteriorSamples[:, 0])
plot.plot(true_omega[-1] * np.ones(niter - burn + 1), '--')
plot.xlabel("sample")
plot.ylabel("omega parameter")
plot.show()

# posterior
plot.plot(calmcmc.posteriorSamples[:, 1])
plot.xlabel("sample")
plot.ylabel("lengthscale parameter")
plot.show()

### comparison between calibrations at final time point

# compare posteriors from different calibration techniques
plot.hist(posterior)
plot.hist(posteriors[-1, :])
plot.plot(np.ones(nparticles) * true_omega[-1], np.linspace(0, nparticles, nparticles), '--')
plot.xlabel("omega")
plot.ylabel("posterior frequency density")
plot.ylim([0, np.max(plot.hist(calmcmc.posteriorSamples[-nparticles:, 0])[0])])
plot.show()