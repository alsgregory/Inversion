from inversion import *
from mpl_toolkits.mplot3d import Axes3D

# model and data generating process
def model(x, shift):
    y = np.sin(shift + (x * 2 * math.pi))
    return(y)

def data_generating_process(x, shift, sigmaY):
    return(model(x, shift) + np.random.normal(0, sigmaY, len(x)))

# params
true_shift = 0.5
sigmaY = 0.025

# coordinates
ns = 5
xData = np.linspace(0, 1, ns)
xModel = np.linspace(0, 1, ns)

# calibration parameters
n = 5
tModel = np.random.uniform(0, 1, ((n, 1)))

# model outputs
yModel = np.zeros((n, len(xModel)))
for i in range(n):
    yModel[i, :] = model(xModel, tModel[i])

# data outputs
m = 30
yData = np.zeros((m, len(xData)))
for i in range(m):
    yData[i, :] = data_generating_process(xData, true_shift, sigmaY)

# define prior for shift and lengthscale
def priorPPF():
    u = np.random.uniform(0, 1, 2)
    shift = norm.ppf(u[0], 1.0, 0.5)
    l = np.exp(norm.ppf(u[1], -2, 0.25))
    return(np.array([shift, l]))

### implement MCMC calibration

# initialize class
cal = calibration.calibrate(priorPPF, sigmaY)

# load coordinates and data
cal.updateCoordinates(xModel, xData)
cal.updateTrainingData(tModel, yModel, yData)

# mcmc
niter = 2000
burn = 1000
beta = 0.01
cal.metropolisHastings(niter, beta, logConstraint=np.array([0, 1]), burn=burn)

# posterior
plot.plot(cal.posteriorSamples[:, 0])
plot.plot(np.ones(niter - burn + 1) * true_shift, '--')
plot.xlabel("sample")
plot.ylabel("shift parameter")
plot.show()

# posterior
plot.plot(cal.posteriorSamples[:, 1])
plot.xlabel("sample")
plot.ylabel("lengthscale parameter")
plot.show()


### implement sequential calibration

# initialize class
cal = calibration.calibrate(priorPPF, sigmaY)

# load coordinates and data
cal.updateCoordinates(xModel, xData)

# particle filter over data outputs
nparticles = 500
beta = 0.02
posteriors = np.zeros((m, nparticles))
for i in range(m):
    cal.updateTrainingData(tModel, yModel, np.reshape(yData[i, :], ((1, ns))))
    cal.sequentialUpdate(nparticles, beta, logConstraint=np.array([0, 1]))
    posteriors[i, :] = cal.posteriorSamples[:, 0]
    print('filtered in ' + str(i) + ' observation sets')

# posterior
plot.plot(cal.posteriorSamples[:, 0])
plot.plot(np.ones(nparticles) * true_shift, '--')
plot.xlabel("sample")
plot.ylabel("shift parameter")
plot.show()

# posterior
plot.plot(cal.posteriorSamples[:, 1])
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

ax.set_xlabel('shift parameter')
ax.set_ylabel('iteration')
ax.set_zlabel('frequency density')
plot.show()