import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import arviz as az
import pymc3 as pm
import tikzplotlib as tikz
import utils

### Simulation parameters ###
video = 'vp'
rate = 30
fps = 60
filename = video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng'
step_size = 60
window_size = 600
lags = 61
maxwin = 300

### Options ###
run_corr = True
run_bayes = False
run_quantile = False
run_robust = False
run_linear = False
run_residue = False # Only relevant if run_bayes, run_quantile, or run_robust is true
run_sizedist = False


### Import trace data ###
dataframe = pd.read_csv('../traces/' + filename + '.csv', names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
dataframe = dataframe.drop(['idx'], axis = 1)
dataframe['time'] = dataframe['time'].diff(-1)
dataframe['deltasize'] = dataframe['size'].diff(-1)
dataframe['time'] = dataframe['time'].fillna(0)
dataframe['deltasize'] = dataframe['deltasize'].fillna(0)
dataframe['pastdelta'] = dataframe['deltasize'].shift(1)
dataframe['pastdelta'] = dataframe['pastdelta'].fillna(0)
dataframe['twopast'] = dataframe['deltasize'].shift(2)
dataframe['twopast'] = dataframe['twopast'].fillna(0)
dataframe['threepast'] = dataframe['deltasize'].shift(3)
dataframe['threepast'] = dataframe['threepast'].fillna(0)
dataframe['fourpast'] = dataframe['deltasize'].shift(4)
dataframe['fourpast'] = dataframe['fourpast'].fillna(0)
dataframe['fivepast'] = dataframe['deltasize'].shift(5)
dataframe['fivepast'] = dataframe['fivepast'].fillna(0)
dataframe['sixpast'] = dataframe['deltasize'].shift(6)
dataframe['sixpast'] = dataframe['sixpast'].fillna(0)

print(dataframe)


### Correlation analysis ###
if (run_corr):
    sizes = dataframe['size'].tolist()
    delta_size = dataframe['deltasize'].tolist()
    lag_range = np.arange(0, lags, 1)
    auto_size = utils.autocorr(sizes - np.mean(sizes), lag_range)
    auto_delta_size = utils.autocorr(delta_size, lag_range)

    #plt.figure()
    #plt.plot(lag_range, auto_size, label='Size correlation')
    #plt.plot(lag_range, auto_delta_size, label='Delta correlation')
    #plt.plot(lag_range, np.zeros(lags), color='black')
    #plt.plot(lag_range, np.ones(lags) * 0.05, linestyle='dashed', color='black')
    #plt.plot(lag_range, np.ones(lags) * -0.05, linestyle='dashed', color='black')
    #plt.xlabel('Lag')
    #plt.ylabel('Autocorrelation')
    #plt.legend()
    #tikz.save('size_autocorr_' + video + '_' + str(rate) + '_' + str(fps) + '.tex')
    #plt.show()

    ifi = dataframe['time'].tolist()
    for i in range(len(ifi) - 1):
        if ifi[i] > 0.1:
            ifi[i] = 0.1
    auto_ifi = utils.autocorr(ifi[: -1], lag_range)

    #plt.figure()
    #plt.plot(lag_range, auto_ifi, label='IFI correlation')
    #plt.plot(lag_range, np.zeros(lags), color='black')
    #plt.plot(lag_range, np.ones(lags) * 0.05, linestyle='dashed', color='black')
    #plt.plot(lag_range, np.ones(lags) * -0.05, linestyle='dashed', color='black')
    #plt.xlabel('Lag')
    #plt.ylabel('Autocorrelation')
    #plt.legend()
    #plt.show()


    #rss = utils.windowed_crosscorr(dataframe, 'size', 'size', lags, step_size, window_size)
    #f,ax = plt.subplots(figsize=(10,10))
    #sbn.heatmap(rss,cmap='RdBu_r',ax=ax, vmin=-1, vmax=1)
    #ax.set(title=f'Rolling Windowed Time Lagged Size Autocorrelation', xlabel='Offset',ylabel='Epochs')
    #plt.show()
    rss = utils.windowed_crosscorr(dataframe, 'deltasize', 'deltasize', lags, step_size, window_size)
    f,ax = plt.subplots(figsize=(10,10))
    xticks = np.arange(1, 61, 5)
    # the content of labels of these yticks
    xticklabels = np.arange(0, 60, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    #pal = sbn.diverging_palette(10, 240, s=80, l=40, n=13)
    pal = sbn.color_palette('inferno_r', 20)
    sbn.color_palette(pal)
    sbn.heatmap(rss,ax=ax,cmap=pal, vmin=-1, vmax=1)
    ax.set(xlabel='Offset',ylabel='Epochs')
    tikz.save('size_rolling_autocorr_' + video + '_' + str(rate) + '_' + str(fps) + '.tex')
    plt.show()

    rss = utils.windowed_crosscorr(dataframe, 'time', 'time', lags, step_size, window_size)
    f,ax = plt.subplots(figsize=(10,10))
    sbn.heatmap(rss,cmap='RdBu_r',ax=ax, vmin=-1, vmax=1)
    ax.set(title=f'Rolling Windowed Time Lagged IFI Autocorrelation', xlabel='Offset',ylabel='Epochs')
    plt.show()

    rss = utils.windowed_crosscorr(dataframe, 'size', 'time', lags, step_size, window_size)
    f,ax = plt.subplots(figsize=(10,10))
    sbn.heatmap(rss,cmap='RdBu_r',ax=ax, vmin=-1, vmax=1)
    ax.set(title=f'Rolling Windowed Time Lagged Size/IFI Crosscorrelation', xlabel='Offset',ylabel='Epochs')
    plt.show()

    rss = utils.windowed_crosscorr(dataframe, 'deltasize', 'time', lags, step_size, window_size)
    f,ax = plt.subplots(figsize=(10,10))
    sbn.heatmap(rss,cmap='RdBu_r',ax=ax, vmin=-1, vmax=1)
    ax.set(title=f'Rolling Windowed Time Lagged Delta Size/IFI Crosscorrelation', xlabel='Offset',ylabel='Epochs')
    plt.show()
    
### Posterior analysis and fitting ###
if (run_bayes or run_robust or run_quantile):
    model = pm.Model()
    delta_size = np.asarray(dataframe['deltasize'].tolist())
    
    if (run_bayes):
        with model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=0, sigma=50)
            beta = pm.Normal("beta", mu=0, sigma=5)
            gamma = pm.Normal("gamma", mu=0, sigma=5)
            #delta = pm.Normal("delta", mu=0, sigma=5)
            #epsilon = pm.Normal("epsilon", mu=0, sigma=5)
            #llambda = pm.Normal("lambda", mu=0, sigma=5)
            #mu = pm.Normal("mu", mu=0, sigma=5)
            sigma = pm.HalfNormal("sigma", sigma=1)
            # Expected value of outcome
            mu = alpha + beta * delta_size[5:-5] + gamma * delta_size[4:-6] #+ delta * delta_size[3:-7] + epsilon * delta_size[2:-8]# + llambda * delta_size[1:-9] + mu * delta_size[:-10]
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=delta_size[6:-4])
            # Draw 10000 posterior samples
            trace = pm.sample(10000, return_inferencedata=False, tune=1000)
            az.plot_trace(trace)
            tikz.save('bayes_' + video + '_' + str(rate) + '_' + str(fps) + '.tex')
            plt.show()
            summary = az.summary(trace, round_to=2)
            print(summary)
            
    if (run_linear):
        lin_model = smf.ols('deltasize ~ pastdelta + twopast + threepast + fourpast + fivepast + sixpast', dataframe).fit()
        print(lin_model.summary())

    if (run_robust):
        rlm_model = smf.rlm('deltasize ~ pastdelta + twopast + threepast + fourpast + fivepast + sixpast', dataframe).fit()
        print(rlm_model.summary())
    
    if(run_quantile):
        qmodel = smf.quantreg('deltasize ~ pastdelta + twopast + threepast + fourpast + fivepast + sixpast', dataframe).fit(q=0.95)
        print(qmodel.summary())

    if (run_residue):
        if (run_bayes):
            residue = delta_size[6:-4] - summary['mean']['alpha'] - summary['mean']['beta'] * delta_size[5:-5] - summary['mean']['gamma'] * delta_size[4:-6]# - summary['mean']['delta'] * delta_size[3:-7] - summary['mean']['epsilon'] * delta_size[2:-8] #- summary['mean']['lambda'] * delta_size[1:-9] - - summary['mean']['mu'] * delta_size[:-10]
            utils.residue_analysis(residue, True, lags, step_size, window_size, 'bayes3')
        if (run_linear):
            residue = delta_size[6:-4] - lin_model.params['Intercept'] - lin_model.params['pastdelta'] * delta_size[5:-5] - lin_model.params['twopast'] * delta_size[4:-6] - lin_model.params['threepast'] * delta_size[3:-7] - lin_model.params['fourpast'] * delta_size[2:-8] - lin_model.params['fivepast'] * delta_size[1:-9] - lin_model.params['sixpast'] * delta_size[:-10]
            utils.residue_analysis(residue, True, lags, step_size, window_size, 'bayes3')
        if (run_robust):
            print(delta_size[6:10])
            residue = delta_size[6:-4] - rlm_model.params['Intercept'] - rlm_model.params['pastdelta'] * delta_size[5:-5] - rlm_model.params['twopast'] * delta_size[4:-6] - rlm_model.params['threepast'] * delta_size[3:-7] - rlm_model.params['fourpast'] * delta_size[2:-8] - rlm_model.params['fivepast'] * delta_size[1:-9] - rlm_model.params['sixpast'] * delta_size[:-10]
            utils.residue_analysis(residue / 1000, True, lags, step_size, window_size, 'robust')        
        if (run_quantile):
            residue = delta_size[6:-4] - qmodel.params['Intercept'] - qmodel.params['pastdelta'] * delta_size[5:-5] - qmodel.params['twopast'] * delta_size[4:-6] - qmodel.params['threepast'] * delta_size[3:-7] - qmodel.params['fourpast'] * delta_size[2:-8] - qmodel.params['fivepast'] * delta_size[1:-9] - qmodel.params['sixpast'] * delta_size[:-10]
            utils.residue_analysis(residue / 1000, True, lags, step_size, window_size, 'quantile')
        
### Size distribution analysis with rolling windows ###
if (run_sizedist):
    sizes = dataframe['size'].tolist()
    print(np.mean(sizes))
    f1 = plt.figure()
    f2 = plt.figure()
    valuerange = np.arange(-0.05, 60.05, 0.1)
    for win in [1, 2, 5, 10, 30, 60, 120, 300]:
        size_mva = utils.rolling_average(sizes, win) / 125000 * fps  # Convert from frame size in B to rate in Mb/s
        plt.figure(f1.number)
        size_hist = np.histogram(size_mva, bins=valuerange)
        plt.plot(size_mva, label='Window: ' + str(win) + ' frames')
        plt.figure(f2.number)
        plt.xlabel('Rate (Mb/s)')
        plt.ylabel('Probability')
        plt.plot(np.arange(0, 60, 0.1), np.cumsum(size_hist[0])/np.sum(size_hist[0]), label='Window: ' + str(win) + ' frames')
        plt.legend()
    tikz.save('window_dist_' + video + '_' + str(rate) + '_' + str(fps) + '.tex',strict=True)
    plt.figure(f1.number)
    plt.plot(range(len(sizes)), np.ones(len(sizes)) * rate, label='CBR rate')
    plt.xlabel('Time')
    plt.ylabel('Rate (Mb/s)')
    plt.legend()
    
    f3 = plt.figure()
    size_std = np.zeros(maxwin)
    size_95 = np.zeros(maxwin)
    size_99 = np.zeros(maxwin)
    
    
    sizecdf = np.zeros((len(valuerange), maxwin))
    
    for win in range(1, maxwin + 1):
        size_mva = utils.rolling_average(sizes, win) / 125000 * fps
        size_std[win - 1] = np.std(size_mva)
        sorted_size = np.sort(size_mva)
        ecdf = sm.distributions.empirical_distribution.ECDF(sorted_size)
        sizecdf[:, win - 1] = ecdf(valuerange / rate)
        size_95[win - 1] = sorted_size[int(len(size_mva) * 0.95)]
        size_99[win - 1] = sorted_size[int(len(size_mva) * 0.99)]
    plt.plot(range(1, maxwin + 1), size_std, label='Rate standard deviation')
    plt.plot(range(1, maxwin + 1), size_95 - rate, label='95th percentile overflow')
    plt.plot(range(1, maxwin + 1), size_99 - rate, label='99th percentile overflow')
    plt.xlabel('Window')
    plt.ylabel('Rate (Mb/s)')
    tikz.save('window_deviation_' + video + '_' + str(rate) + '_' + str(fps) + '.tex')
    plt.legend()
    

    f,ax = plt.subplots(figsize=(10,10))
    sbn.heatmap(pd.DataFrame(sizecdf),cmap='RdBu_r',ax=ax, vmin=0, vmax=1)
    ax.set(title=f'Averaged rate', xlabel='Window',ylabel='Relative rate')
    plt.show()
