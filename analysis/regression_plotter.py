import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as sps
import statsmodels.formula.api as smf
import tikzplotlib as tikz
import csv
import warnings

class RegressionPlotter:
    """
    A class that imports trace data and handles the modeling and prediction
    of future frame size with different parameters and regression models.

    Attributes
    ----------
    dataframe : pandas dataframe
        a dataframe containing the trace, divided frame by frame (with fields
        idx, time, and size)
    shifted_size : numpy array
        array containing the average frame size for the next T steps, with a
        predetermined delay
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    __init__(self, videos, rates, fps)
        Constructor. Builds the object by concatenating traces into a single
        dataframe and normalizing frame sizes
        
    regress(self, method, past_steps, future_steps, future_shift, percentile, 
            guard)
        Computes the regression model parameters with the given settings
        
    residue(self, model, past_steps, future_steps, future_shift, percentile, 
            guard, rate, fps)
        Returns the residual noise in bits for the prediction for a given model
        
    get_index(self, percentile, guard, future_steps)
        Gets the index in the sorted residual vector for the given percentile
        
    predict(past_samples, model, rate, fps)
        Predicts a single sample (in bits) from a linear model and returns it
        
    fit_residue(residue, bins)
        Fits the residue to a Laplace distribution and returns its parameters
        
    fit_residue_plot(residue, bins, show=False, folder='', savename='')
        Plots the residue distribution, fitted to various common long-tailed
        distributions, and shows or saves the figure
        
    histogram_plot(residues, labels, bins, logscale, complementary, show=False,
                   folder='', savename='')
        Plots the empirical CDF or CCDF of the residual error vector
        
    autocorr(x, lags)
        Computes the autocorrelation for a given vector
        
    autocorr_plot(residues, labels, lags, show=False, folder='', savename='')
        Plots the autocorrelation of the residual error signal and shows it or
        saves it to file
    """


    def __init__(self, videos, rates, fps):
        """
        Parameters
        ----------
        videos : list (str)
            The list of video contents to import
        rates : list (int)
            The list of bitrate levels to import
        fps : list (int)
            The list of framerates to import
        """
        
        frames = []
        ### Import trace data ###
        for f in fps:
            for v in videos:
                for r in rates:
                    # Load the relevant elements of the trace
                    filename = '../traces/' + v + '_' + str(r) + 'mbps_' + str(f) + 'fps.pcapng' + '.csv'
                    frame = pd.read_csv(filename, names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
                    frame = frame.drop(['idx'], axis = 1)
                    frame = frame.drop(['time'], axis = 1)
                    # Normalize frame size to compare different traces fairly
                    frame['size'] *=  f / r / 125000
                    frames.append(frame)
        self.dataframe = pd.concat(frames)

    def __load_data(self, past_steps, future_steps, future_shift):
        """
        Parameters
        ----------
        past_steps : int
            The memory of the linear model (N in the paper)
        future_steps : int
            How many steps to average (T in the paper)
        future_shift : int
            The delay in the prediction (tau in the paper), starting from 0
        """
        
        # Create fields for the past N frames
        for step in range(1, past_steps + 1):
            name = 'pastsize_' + str(step)
            self.dataframe[name] = self.dataframe['size'].shift(step + future_shift)
            self.dataframe[name] = self.dataframe[name].fillna(0)
            
        # Compute average of the future T frames (shifted by tau)
        self.dataframe['shifted_size'] = self.dataframe['size']
        for step in range(1, future_steps):
            name = 'futuresize_' + str(step)
            self.dataframe[name] = self.dataframe['size'].shift(-step)
            self.dataframe[name] = self.dataframe[name].fillna(0)
            self.dataframe['shifted_size'] = self.dataframe['shifted_size'] + self.dataframe[name]
            
        self.dataframe['shifted_size'] = self.dataframe['shifted_size'] / future_steps
        self.shifted_size = self.dataframe['shifted_size'].to_numpy()

    def regress(self, method, past_steps, future_steps, future_shift, percentile):
        """
        Parameters
        ----------
        method : str
            The regression method (linear, robust or quantile)
        past_steps : int
            The memory of the linear model (N in the paper)
        future_steps : int
            How many steps to average (T in the paper)
        future_shift : int
            The delay in the prediction (tau in the paper), starting from 0
        percentile: double
            The quantile to use for quantile regression (0 to 1)
            
        Returns
        -------
        dict
            A dictionary containing the model parameters
        """
        
        self.__load_data(past_steps, future_steps, future_shift)
        if (past_steps > 0):
            # Define the regression problem and run the model
            problem = 'shifted_size ~ '
            for i in range(past_steps):
                problem = problem + 'pastsize_' + str(i + 1)
                if (past_steps > i + 1):
                    problem = problem + ' + '
            model = []
            if method == 'Linear':
                model = smf.ols(problem, self.dataframe).fit()
            if method == 'Robust':
                model = smf.rlm(problem, self.dataframe,M=sm.robust.norms.HuberT(t=0.25)).fit()
            if method == 'Quantile':
                model = smf.quantreg(problem, self.dataframe).fit(q=percentile)
            return model.params
        else:
            # If N=0, the model is a constant value
            if (method == 'Quantile'):
                q = np.sort(self.shifted_size)[int(len(self.shifted_size) * percentile)]
                return {'Intercept':q}
            else:
                return {'Intercept':np.mean(self.shifted_size)}
    
    def residue(self, model, past_steps, future_steps, future_shift, percentile, guard, rate, fps):
        """
        Parameters
        ----------
        model : dict
            The regression model
        past_steps : int
            The memory of the linear model (N in the paper)
        future_steps : int
            How many steps to average (T in the paper)
        future_shift : int
            The delay in the prediction (tau in the paper), starting from 0
        percentile: double
            The quantile to use for quantile regression (0 to 1)
        guard: int
            The number of steps to discard at the beginning and end of the
            dataframe to avoid 
        rate: int
            The bitrate of the video
        fps: int
            The framerate of the video
            
        Returns
        -------
        numpy array
            The residual error vector (in bits)
        """
        
        self.__load_data(past_steps, future_steps, future_shift)
        indices = np.arange(guard, len(self.shifted_size) - guard + 1, future_steps)
        residue = self.shifted_size[indices] - model['Intercept']
        for i in range(past_steps):
            name = 'pastsize_' + str(i + 1)
            pastsize = self.dataframe[name].to_numpy()
            residue = residue - model[name] * pastsize[indices]
        return residue * rate * 125000 / fps
    
    def get_index(self, percentile, guard, future_steps):
        """
        Parameters
        ----------
        percentile: double
            The quantile to use for quantile regression (0 to 1)
        guard: int
            The number of steps to discard at the beginning and end of the
            dataframe to avoid 
        future_steps : int
            How many steps to average (T in the paper)
            
        Returns
        -------
        int
            The desired index in the residue vector
        """
        
        return int(np.ceil((self.dataframe.shape[0] - 2 * guard) * percentile / future_steps))
    
    def predict(past_samples, model, rate, fps):
        """
        Parameters
        ----------
        past_samples: list (int)
            List of past samples to use for the prediction
        model : dict
            The regression model
        rate: int
            The bitrate of the video
        fps: int
            The framerate of the video
                        
        Returns
        -------
        double
            The predicted frame size in bits
        """
        
        predicted = model['Intercept'] * rate * 125000 / fps
        for i in range(len(past_samples)):
            name = 'pastsize_' + str(i + 1)
            predicted += model[name] * past_samples[i]
        return predicted
    
    def fit_residue(residue, bins):
        """
        Parameters
        ----------
        residue: numpy array
            Residual error of the prediction on a trace
        bins: int
            The number of bins to use for the histogram
                        
        Returns
        -------
        list
            A dictionary containing the distribution parameters
        """
        
        y, x = np.histogram(residue, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        distribution = getattr(sps, 'laplace')
        params = distribution.fit(residue)
        return params

    def fit_residue_plot(residue, bins, show=False, folder='', savename=''):
        """
        Parameters
        ----------
        residue: numpy array
            Residual error of the prediction on a trace
        bins: int
            The number of bins to use for the histogram
        show: bool, optional
            True to show the plot, false to save it
        folder: str, optional
            If show is false, the folder to save the plot into
        savename: str, optional
            If show is false, the name to give the file
        """
        
        # Create empirical CDF
        y, x = np.histogram(residue, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        plt.figure()
        plt.plot(x, y, label='Empirical distribution')
        
        sse = np.zeros(4)
        ii = 0
        for distname in ['cauchy','laplace', 'norm', 't']:
            distribution = getattr(sps, distname)
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fitted
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # Fit distribution to data
                    params = distribution.fit(residue)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    fit = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse[ii] = np.sum(np.power(y - fit, 2.0))
                    plt.plot(x, fit, label=distname)
                    ii += 1   
                    
            except Exception as e:
                print(e)
        plt.xlabel('Residue error')
        plt.ylabel('PDF')
        plt.legend()       
        if (show):
            plt.show()
        else:
            tikz.save(folder + '/residue_dist_' + savename + '.tex')
            plt.close()
        
    def histogram_plot(residues, labels, bins, logscale, complementary, show=False, folder='', savename=''):
        """
        Parameters
        ----------
        residues: list (numpy array)
            List of residual errors for different traces
        labels: list (str)
            The names for the traces
        bins: int
            The number of bins to use for the histogram
        logscale: bool
            True to set the plot to use log scale on the y axis
        complementary: bool
            True to plot the CCDF instead of the CDF
        show: bool, optional
            True to show the plot, false to save it
        folder: str, optional
            If show is false, the folder to save the plot into
        savename: str, optional
            If show is false, the name to give the file
        """
        
        plt.figure()
        if (logscale):
            plt.yscale('log')
        for i in range(len(residues)):
            y, x = np.histogram(residues[i], bins=bins)
            x = (x + np.roll(x, -1))[:-1] / 2.0
            cdf = np.cumsum(y) / np.sum(y)
            if (complementary):
                cdf = 1 - cdf
            plt.plot(x, cdf, label=labels[i])
        plt.xlabel('Residual error w (kB)')
        plt.legend()
        if (complementary):
            plt.ylabel('CCDF')
        else:
            plt.ylabel('CDF')
            
        if (show):
            plt.show()
        else:
            if (complementary):
                tikz.save(folder + '/ccdf_' + savename + '.tex')
            else:
                tikz.save(folder + '/cdf_' + savename + '.tex')
            plt.close()
            
    def autocorr(x, lags):
        """
        Parameters
        ----------
        x: numpy array
            The data vector
        lags: int
            The number of lags to compute the autocorrelation for
                        
        Returns
        -------
        numpy array
            The autocorrelation vector
        """
        
        corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in range(lags + 1)]
        return np.array(corr)  
            
    def autocorr_plot(residues, labels, lags, show=False, folder='', savename=''):
        """
        Parameters
        ----------
        residues: list (numpy array)
            List of residual errors for different traces
        labels: list (str)
            The names for the traces
        lags: int
            The number of lags to show in the plot
        show: bool, optional
            True to show the plot, false to save it
        folder: str, optional
            If show is false, the folder to save the plot into
        savename: str, optional
            If show is false, the name to give the file
        """
        
        plt.figure()
        for i in range(len(residues)):
            rescorr = RegressionPlotter.autocorr(residues[i], lags)
            plt.plot(range(lags + 1), rescorr, label=labels[i])
        plt.legend()
        plt.plot(range(lags + 1), np.ones(lags + 1) * 0.05, linestyle='dashed', color='black')
        plt.plot(range(lags + 1), np.ones(lags + 1) * -0.05, linestyle='dashed', color='black')
        plt.xlabel('Lag (frames)')
        plt.ylabel('Autocorrelation')
        if (show):
            plt.show()
        else:
            tikz.save(folder + '/autocorr_' + savename + '.tex')
            plt.close()
    

def set_box_color(bp, color):
    """
    Sets the color of a boxplot
    
    Parameters
    ----------
    bp: boxplot object
        The boxplot to recolor
    color: str
        Hex representation of the color
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

if __name__ == "__main__":
    # Simulation parameters
    videos = ['mc', 'ge_cities', 'ge_tour', 'vp']
    rates = np.arange(10, 51, 10)
    fpss = [30, 60]
    video = 'vp'
    rate = 30
    fps = 60
    
    # Regression options
    reg_methods = ['Linear', 'Quantile']
    quantile = 0.95
    guard = 50
    lags = 20
    future = [1, 6]
    bins = 500
    shifts = 10
    max_taps = 10
    future_steps = 1
    single = False
    
    if (single):
        # The prediction and results are for a single trace
        plotter = RegressionPlotter([video], [rate], [fps])
        
        for method in reg_methods:
            # Run colormap for single-frame prediction
            residue_tail = []
            residue_std = []
            residue_tail.append(['x','y','z'])
            residue_std.append(['x','y','z'])
            for shift in range(shifts + 1):
                for past_steps in range(max_taps + 1):
                    model = plotter.regress(method, past_steps, future_steps, shift, quantile, guard)
                    residue = plotter.residue(model, past_steps, future_steps, shift, quantile, guard, rate, fps)
                    residue_tail.append([past_steps, shift, np.sort(residue)[plotter.get_index(quantile, guard, future_steps)] / 1000])
                    residue_std.append([past_steps, shift, np.std(residue) / 1000])
            with open('figures/' + method + '_95.csv', 'w') as f:
                writer = csv.writer(f, delimiter = " ")
                writer.writerows(residue_tail)
            with open('figures/' + method + '_std.csv', 'w') as f:
                writer = csv.writer(f, delimiter = " ")
                writer.writerows(residue_std)
                                    
            # Autocorrelation and CDF for future prediction
            for future_steps in future:
                residues = []
                labels = []
                for past_steps in range(max_taps + 1):
                    model = plotter.regress(method, past_steps, future_steps, 0, quantile, guard)
                    print(method,future_steps,model)
                    residue = plotter.residue(model, past_steps, future_steps, 0, quantile, guard, rate, fps)
                    if (past_steps == 6):
                        RegressionPlotter.fit_residue_plot(residue, bins, folder='figures/', savename=method + '_' + str(future_steps) + '_frames')
                    residues.append(residue / 1000)
                    labels.append('N: ' + str(past_steps))
                RegressionPlotter.histogram_plot(residues, labels, bins, True, True, folder='figures/', savename=method + '_' + str(future_steps) + '_frames')
                RegressionPlotter.autocorr_plot(residues, labels, lags, folder='figures/', savename=method + '_' + str(future_steps) + '_frames')
    else:
        # The models are over multiple traces
        for method in reg_methods:
            for future_steps in future:
                plt.figure()
                residue_single = []
                residue_rate = []
                residue_gen = []
                labels = []
                past_steps = 6
                shift = 0
                gen_files = []
                gen_plotter = RegressionPlotter(videos, rates, fpss)
                gen_model = gen_plotter.regress(method, past_steps, future_steps, shift, quantile, guard)
                # Analyze residue for all traces
                for video in videos:
                    rate_files = []
                    rate_plotter = RegressionPlotter([video], rates, fpss)
                    rate_model = rate_plotter.regress(method, past_steps, future_steps, shift, quantile, guard)
                    for rate in rates:
                        plotter = RegressionPlotter([video], [rate], [fps])
                        single_model = plotter.regress(method, past_steps, future_steps, shift, quantile, guard)
                        residue_single.append(plotter.residue(single_model, past_steps, future_steps, shift, quantile, guard, rate, fps) * fps / rate / 125000)
                        residue_rate.append(plotter.residue(rate_model, past_steps, future_steps, shift, quantile, guard, rate, fps) * fps / rate / 125000)
                        residue_gen.append(plotter.residue(gen_model, past_steps, future_steps, shift, quantile, guard, rate, fps) * fps / rate / 125000)
                        labels.append(video + '_' + str(rate))
                
                bpg = plt.boxplot(residue_gen, positions=np.arange(0, len(labels), 1) * 3.0 - 0.8, sym='', widths=0.6)
                bpr = plt.boxplot(residue_rate, positions=np.arange(0, len(labels), 1) * 3.0, sym='', widths=0.6)
                bps = plt.boxplot(residue_single, positions=np.arange(0, len(labels), 1) * 3.0 + 0.8, sym='', widths=0.6)
                set_box_color(bpg, '#FFD700')
                set_box_color(bpr, '#EA5F94')
                set_box_color(bps, '#0000FF')
                
                plt.xlim(-3, len(labels)*3)
                plt.ylim(-1, 1)
                plt.xticks(np.arange(0, len(labels) * 3, 3), labels)
                plt.tight_layout()
                tikz.save('figures/gen_boxplot_' + method + '_' + str(future_steps) + '.tex')
                plt.show()
