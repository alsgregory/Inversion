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
true_delta_t = 100
sigmaY = 0.5

# coordinates
ns = 5
xData = np.linspace(0, 10, ns)
xModel = np.linspace(0, 10, ns)

# calibration parameters
n = 10
tModel = np.array([np.random.normal(7, 1, n),
                   np.random.normal(100, 20, n)]).T

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
    omega = norm.ppf(u[0], 8, 3)
    delta_t = norm.ppf(u[1], 80, 15)
    l = np.exp(norm.ppf(u[2], 1, 1))
    return(np.array([omega, delta_t, l]))

### implement sequential calibration

# initialize class
cal = calibration.calibrate(priorPPF, sigmaY)

# load coordinates and data
cal.updateCoordinates(xModel, xData)

# particle filter over data outputs
nparticles = 600
beta = 1
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

ax.set_xlabel('shift parameter')
ax.set_ylabel('iteration')
ax.set_zlabel('frequency density')
plot.show()

plot.plot(np.mean(posteriors, axis=1))
plot.plot(true_omega, '--')
plot.show()