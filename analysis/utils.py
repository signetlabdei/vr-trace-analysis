import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.stats as sps
import bottleneck as bn
import tikzplotlib as tikz
import warnings

def autocorr(x,lags):
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)         

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
    
def windowed_crosscorr(df, col1, col2, lags, step_size, window_size):
    
    """ Lag-N windowed cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    df: pandas data object
    col1: name of the first column to correlate
    col1: name of the first column to correlate
    lags: number of lags (from 0 to max value)
    step_size: step in rolling window
    window_size: length of rolling window
    
    Returns
    ----------
    crosscorr : float
    """
    rss=np.zeros(((len(df[col1]) - window_size) // step_size + 1, lags))
    t_start = 0
    t_end = window_size
    i = 0
    while t_end < len(df[col1]):
        d1 = df[col1].iloc[t_start:t_end]
        d2 = df[col2].iloc[t_start:t_end]
        rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(0, lags)]
        rss[i, :] = rs
        i += 1
        t_start = t_start + step_size
        t_end = t_end + step_size
    return rss 

def rolling_average(a,n):
    'bottleneck.move_mean'
    return bn.move_mean(a, window=n, min_count=1)

def residue_analysis(residue, plotting, lags, step_size, window_size, name):
    if (plotting):
        plt.figure()
        plt.plot(residue)
        plt.show()

        plt.figure()
        plt.hist(residue, density=True,bins=400)
        plt.show()
        
        plt.figure()
        plt.specgram(residue, Fs = 60)
        plt.show()
        
        plt.figure
        lag_range = np.arange(0, lags, 1)
        auto_res = autocorr(residue[1: -1], lag_range)
        plt.plot(lag_range, auto_res, label='Residue correlation')
        plt.plot(lag_range, np.zeros(lags), color='black')
        plt.plot(lag_range, np.ones(lags) * 0.05, linestyle='dashed', color='black')
        plt.plot(lag_range, np.ones(lags) * -0.05, linestyle='dashed', color='black')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        tikz.save(name + '_residue_autocorr.tex')
        plt.show()
    
    # Fit distributions with Scipy Stats

    y, x = np.histogram(residue, bins=400, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    if (plotting):
        plt.figure()
        plt.plot(x,y,label='Data')
    
    sse = np.zeros(4)
    ii = 0
    for distname in ['cauchy','laplace', 'norm', 't']:

        distribution = getattr(sps, distname)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                # fit dist to data
                params = distribution.fit(residue)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                if (distname=='t'):
                    perc = sps.t.ppf(0.99, params[0], params[1], params[2])
                
                # Calculate fitted PDF and error with fit in distribution
                fit = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse[ii] = np.sum(np.power(y - fit, 2.0))
                if (plotting):
                    plt.plot(x, fit, label=distname)
                ii += 1   
                print(distname, sse)
                
        except Exception as e:
            print(e)
            print('missed ' + distname)
    
    if (plotting):
        plt.xlabel('Residue error')
        plt.ylabel('PDF')
        plt.legend()       
        tikz.save(name + '_residue_fitdist.tex')
        plt.show()
        
        plt.figure()
        

    
    if (plotting):
        print(params, ' 99th percentile: ', perc)
        
        residueframe = pd.DataFrame(residue, columns = ['residue'])
        rss = windowed_crosscorr(residueframe, 'residue', 'residue', lags, step_size, window_size)
        f,ax = plt.subplots(figsize=(10,10))
        sbn.heatmap(rss,cmap='RdBu_r',ax=ax, vmin=-1, vmax=1)
        ax.set(title=f'Rolling Windowed Time Lagged Residue Autocorrelation', xlabel='Offset',ylabel='Epochs')
        tikz.save(name + '_residue_rolling_autocorr.tex')
        plt.show()
    
    return params
