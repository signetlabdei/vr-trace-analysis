import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import utils

### Simulation parameters ###
video = 'vp'
rate = 40
fps = 60
filename = video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng'

### Regression options ###
reg_method = 'Robust'       # Type of regression
max_steps = 0               # Maximum number of taps in the ARMA filter
steps_ahead = 1             # Number of steps for the lookahead
future_shift = 0            # Shift to the future (0 = next step)
quantile = 0.95             # Quantile to predict (only meaningful for quantile regression)
residue_quantile = 0.95     # Quantile of the residue to consider
guard = 50                  # Number of samples to discard at the beginning and end of the video (must be larger than 0!)
lags = 20                   # Lags for autocorrelation

### Import trace data ###
dataframe = pd.read_csv('../traces/' + filename + '.csv', names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
dataframe = dataframe.drop(['idx'], axis = 1)
index = int(np.ceil((dataframe.shape[0] - 2 * guard) * residue_quantile / steps_ahead))
dataframe['time'] = dataframe['time'].diff(-1)
dataframe['size'] = dataframe['size'] / 1000
for step in range(1, max_steps + 1):
    name = 'size_' + str(step)
    dataframe[name] = dataframe['size'].shift(step + future_shift)
    dataframe[name] = dataframe[name].fillna(0)
    
for futurestep in range(1, steps_ahead):
    name = 'futuresize_' + str(futurestep)
    dataframe[name] = dataframe['size'].shift(-futurestep)
    dataframe[name] = dataframe[name].fillna(0)
    dataframe['size'] = dataframe['size'] + dataframe[name]
    
dataframe['size'] = dataframe['size'] / steps_ahead
size = dataframe['size'].to_numpy()
print(np.mean(size))

residue_tail = np.zeros(max_steps + 1)
residue_var = np.zeros(max_steps + 1)

f1 = plt.figure()
plt.yscale('log')
f2 = plt.figure()

### Solve the regression problem and get residue percentile ###
for step in range(max_steps + 1):
    if (step > 0):
        problem = 'size ~ '
        for i in range(step):
            problem = problem + 'size_' + str(i + 1)
            if (step > i + 1):
                problem = problem + ' + '
        model = []
        if reg_method == 'Linear':
            model = smf.ols(problem, dataframe).fit()
        if reg_method == 'Robust':
            model = smf.rlm(problem, dataframe,M=sm.robust.norms.HuberT(t=(np.mean(size) / 4))).fit()
        if reg_method == 'Quantile':
            model = smf.quantreg(problem, dataframe).fit(q=quantile)
        size = dataframe['size'].to_numpy()
        residue = size[np.arange(guard, len(size) - guard + 1, steps_ahead)] - model.params['Intercept']
        for i in range(step):
            name = 'size_' + str(i + 1)
            pastsize = dataframe[name].to_numpy()
            residue = residue - model.params[name] * pastsize[np.arange(guard, len(size) - guard + 1, steps_ahead)]
        print(step, model.summary())
    else:
        residue = size[np.arange(guard, len(size) - guard + 1, steps_ahead)] - np.mean(size[np.arange(guard, len(size) - guard + 1, steps_ahead)])
    y, x = np.histogram(residue, bins=400)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    plt.figure(f1.number)
    plt.plot(x, np.cumsum(y) / np.sum(y),label=str(step) + ' taps')
    plt.figure(f2.number)
    rescorr = utils.autocorr(residue, range(lags + 1))
    plt.plot(range(lags + 1), rescorr, label=str(step) + ' taps')
    residue_var[step] = np.std(residue)
    residue_tail[step] = np.sort(residue)[index]

plt.figure(f1.number)
plt.xlabel('Estimate difference')
plt.ylabel('CDF')
plt.title(reg_method)
plt.legend()

plt.figure(f2.number)
plt.plot(range(lags + 1), np.ones(lags + 1) * 0.05, linestyle='dashed', color='black')
plt.plot(range(lags + 1), np.ones(lags + 1) * -0.05, linestyle='dashed', color='black')
plt.yscale('linear')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title(reg_method)
plt.legend()
plt.show()
    
plt.figure()
plt.plot(range(max_steps + 1), residue_var, label='Standard deviation')
plt.plot(range(max_steps + 1), residue_tail, label=str(residue_quantile * 100) + 'th percentile')
plt.legend()
plt.xlabel('Number of taps')
plt.ylabel('kB')
plt.title(reg_method)
plt.show()
