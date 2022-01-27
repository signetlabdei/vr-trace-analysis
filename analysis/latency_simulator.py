import numpy as np
import regression_plotter as reg 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import tikzplotlib as tikz

class Simulator:
     """
    A class that exploits the regression models to simulate a network slicing
    scenario that can exploit future frame prediction.

    Attributes
    ----------
    guard: int
        The number of steps to discard at the beginning and end of the
        dataframe to avoid 
    method : str
        The regression method (linear, robust or quantile)
    past_steps : int
        The memory of the linear model (N in the paper)
    future_steps : int
        How many steps to average (T in the paper)
    feedback_latency : int
        The delay in the prediction (tau in the paper), starting from 0
    percentile: double
        The quantile to use for quantile regression (0 to 1)
    regressor: RegressionPlotter
        The regression helper object

    Methods
    -------
    __init__(self, videos, rates, fpss, past_steps, feedback_latency, method, 
             quantile, guard)
        Constructor. Builds the object and initializes the regression model
        
    set_quantile(self, quantile)
        Changes the prediction quantile
        
    simulate(self, period, frame_by_frame, video, rate, fps)
        Main simulation function
    """
    
    def __init__(self, videos, rates, fps, past_steps, feedback_latency, method, quantile, guard):
        """
        Parameters
        ----------
        videos : list (str)
            The list of video contents to import
        rates : list (int)
            The list of bitrate levels to import
        fps : list (int)
            The list of framerates to import
        past_steps : int
            The memory of the linear model (N in the paper)
        feedback_latency : int
            The delay in the prediction (tau in the paper), starting from 0
        method : str
            The regression method (linear, robust or quantile)
        quantile: double
            The quantile to use for quantile regression (0 to 1)
        guard: int
            The number of steps to discard at the beginning and end of the
            dataframe to avoid 
        """
        
        self.guard = guard
        self.method = method
        self.past_steps = past_steps
        self.feedback_latency = feedback_latency
        self.quantile = quantile
        self.regressor = reg.RegressionPlotter(videos, rates, fps)

        
    def __setup(self, period, frame_by_frame):
        """
        Parameters
        ----------
        period: int
            The interval (in frames) between scheduling decisions 
        frame_by_frame: bool
            True for frame by frame scheduling, false for constant scheduling
        
        Returns
        -------
        list
            A list containing the prediction models for the given scheduler settings
        """
        
        # Generate prediction model list
        models = []
        if (frame_by_frame):
            for i in range(period):
                models.append(self.regressor.regress(self.method, self.past_steps, 1, i + self.feedback_latency, self.quantile, self.guard))
        else:
            models.append(self.regressor.regress(self.method, self.past_steps, period, self.feedback_latency, self.quantile, self.guard))
        return models
    
    def set_quantile(self, quantile):
        """
        Parameters
        ----------
        quantile: double
            The quantile to use for quantile regression (0 to 1)
        """
        
        self.quantile = quantile
        
    def simulate(self, period, frame_by_frame, video, rate, fps):
        """
        Parameters
        ----------
        period: int
            The interval (in frames) between scheduling decisions 
        frame_by_frame: bool
            True for frame by frame scheduling, false for constant scheduling
        video : str
            The video content to simulate
        rate : int
            The bitrate level of the trace
        fps : int
            The framerate of the trace
        
        Returns
        -------
        tuple
            A tuple containing the latency (in ms) and scheduled capacity
            allocation (in b/s) numpy arrays
        """
        
        # Import trace and generate models
        filename = '../traces/' + video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng' + '.csv'
        df = pd.read_csv(filename, names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
        frame_size = df['size'].to_numpy()[self.guard : -self.guard]
        models = self.__setup(period, frame_by_frame)
        
        # Estimate residue quantiles for static allocation (non-quantile regression methods)
        dist = getattr(sps, 'laplace')
        residue_q = []
        if (method != 'Quantile'):
            reg_res = reg.RegressionPlotter([video], [rate], [fps])
            if (frame_by_frame):
                for i in range(period):
                    params = reg.RegressionPlotter.fit_residue(reg_res.residue(models[i], self.past_steps, 1, i + self.feedback_latency, self.quantile, self.guard, rate, fps), 500)
                    residue_q.append(dist.ppf(self.quantile, loc=params[-2], scale=params[-1]))
            else:
                params = reg.RegressionPlotter.fit_residue(reg_res.residue(models[0], self.past_steps, period, self.feedback_latency, self.quantile, self.guard, rate, fps), 500)
                residue_q.append(dist.ppf(self.quantile, loc=params[-2], scale=params[-1]))
                
        schedule = np.zeros(len(frame_size))
        latency = np.zeros(len(frame_size))
        queue = np.zeros(len(frame_size))
        
        # Main scheduling loop: keep track of the queue and determine schedule
        for i in range(len(frame_size)):
            if (np.mod(i, period) == period - 1):
                if (frame_by_frame):
                    for j in range(i, i + period):
                        if (j < len(schedule)):
                            schedule[j] = reg.RegressionPlotter.predict(frame_size[i - self.past_steps : i], models[j - i], rate, fps)
                            if (method != 'Quantile'):
                                schedule[j] += residue_q[j - i]
                    schedule[i] += queue[i - 1]
                else:
                    schedule[i : i + period] = reg.RegressionPlotter.predict(frame_size[i - self.past_steps : i], models[0], rate, fps) + queue[i - 1] / period
                    if (method != 'Quantile'):
                        schedule[i : i + period] += residue_q[0]
            queue[i] = np.max([0, queue[i - 1] + frame_size[i] - schedule[i]])
            
        # Latency calculation loop: compute latency for individual frames
        j = 0
        spent = 0
        for i in range(len(frame_size)):
            if (j < i):
                j = i
                spent = 0
            remaining = frame_size[i]
            while (j < len(schedule) - 1 and remaining > schedule[j] * (1 - spent)):
                remaining -= schedule[j] * (1 - spent)
                spent = 0
                j += 1
            if (j == len(schedule) - 1):
                break
            latency[i] = (j - i) + spent + remaining / schedule[j]
            spent = remaining / schedule[j] + spent
                    
        latency *= 1000 / fps
        schedule *= fps / 125000
        return latency, schedule
        
        
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
    
def boxplot(data, palette, ticks, groups, xlabel, ylabel, ymax, line=-1, show=False, folder='', savename=''):
    """
    Plots the data as a boxplot
    
    Parameters
    ----------
    data: list(numpy array)
        Lists of data sets
    palette: list(str)
        Hex representation of the color
    ticks:
        Names of the data ticks
    groups:
        Names of the data groups
    xlabel: str
        Label on x axis
    ylabel: str
        Label on y axis
    ymax: double
        Maximum value on y axis
    show: bool, optional
        True to show the plot, false to save it
    folder: str, optional
        If show is false, the folder to save the plot into
    savename: str, optional
        If show is false, the name to give the file
    """
    
    plt.figure()
    N = len(groups)
    for i in range(N):
        bp = plt.boxplot(data[i], positions=np.arange(0, len(ticks), 1) * N + (i + 0.5 - N / 2) * 0.8, sym='', widths=0.6)
        set_box_color(bp, palette[i])
        plt.plot([], c=palette[i], label=groups[i])
    plt.legend()
    plt.xlim(-N, len(ticks)*N)
    plt.ylim(0, ymax)
    if (line > 0):
        plt.plot(np.arange(-N, (len(ticks) + 1) * N, 1), np.ones((len(ticks) + 2) * N) * line, linestyle='dashed', color='black')
    plt.xticks(np.arange(0, len(ticks) * N, N), ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (show):
        plt.show()
    else:
        tikz.save(folder + savename + '.tex')
        plt.close()
    
    

if __name__ == "__main__":
    # Main simulation parameters
    video = 'vp'
    rate = 30
    fps = 60    
    method = 'Quantile'
    quantile = 0.95
    past_steps = 6
    feedback_latency = 0
    guard = 50
    bins = 500
    max_granularity = 10
    palette = ['#cd34b5', '#0000ff']
    groups = ['Frame', 'Overall']
    percentiles = [0.5, 0.95, 1 - 1 / 60]
    rates = np.arange(10, 51, 10)
    videos = ['vp']
    fpss = [30, 60]
    futures = [6]
    
    latencies = [[], []]
    schedules = [[], []]
    labels = []
    
    # Initialize simulator and run boxplot code as a function of the scheduler period
    sim = Simulator(videos, rates, fpss, past_steps, feedback_latency, method, quantile, guard)
    for i in range(max_granularity):
        labels.append(str(i + 1))
        latency, schedule = sim.simulate(i + 1, True, video, rate, fps)
        latencies[0].append(latency)
        schedules[0].append(schedule)
        latency, schedule = sim.simulate(i + 1, False, video, rate, fps)
        latencies[1].append(latency)
        schedules[1].append(schedule)
        
    boxplot(latencies, palette, labels, groups, 'Granularity', 'Latency (ms)', 30, line=1000 / fps, folder='figures/', savename='schedule_' + method + '_latency_granularity')
    boxplot(schedules, palette, labels, groups, 'Granularity', 'Scheduled capacity (Mb/s)', 50, folder='figures/', savename='schedule_' + method + '_rate_granularity')
    
    percentiles = np.arange(0.9, 1, 0.005)
    
    # Simulate the scheduler as a function of the desired quantile
    perc_line = [0.9, 0.95, 0.99]
    for future in futures:
        latencies = [[], []]
        schedules = [[], []]
        for perc in percentiles:
            sim.set_quantile(perc)
            latency, schedule = sim.simulate(future, True, video, rate, fps)
            latencies[0].append(latency)
            schedules[0].append(schedule)
            latency, schedule = sim.simulate(future, False, video, rate, fps)
            latencies[1].append(latency)
            schedules[1].append(schedule)
        for i in range(2):
            perc_latency = np.zeros((len(perc_line), len(latencies[i])))
            perc_schedule = np.zeros((len(perc_line), len(schedules[i])))
            av_latency = np.zeros(len(latencies[i]))
            av_schedule = np.zeros(len(schedules[i]))
            for q in range(len(latencies[i])):
                av_schedule[q] = np.mean(schedules[i][q])
                av_latency[q] = np.mean(latencies[i][q])
                for p in range(len(perc_line)):
                    idx = int(np.ceil(perc_line[p] * len(latencies[i][q])))
                    perc_latency[p, q] = np.sort(latencies[i][q])[idx]
                    perc_schedule[p, q] = np.sort(schedules[i][q])[idx]
            
            # Latency vs quantile
            plt.figure()
            plt.plot(percentiles, av_latency, marker='o', label='Mean')
            for p in range(len(perc_line)):
                plt.plot(percentiles, perc_latency[p, :], marker='o', label=str(perc_line[p]))
            plt.xlabel('p_s')
            plt.ylabel('Latency')
            plt.title(groups[i])
            plt.legend()
            tikz.save('figures/perc_latency_' + groups[i] + '_' + str(future) + '.tex')
            plt.show()
            
            # Schedule vs quantile
            plt.figure()            
            plt.plot(percentiles, av_schedule, marker='o', label='Mean')
            for p in range(len(perc_line)):
                plt.plot(percentiles, perc_schedule[p, :], marker='o', label=str(perc_line[p]))
            plt.xlabel('p_s')
            plt.ylabel('Average schedule')
            plt.title(groups[i])
            plt.legend()
            tikz.save('figures/perc_schedule_' + groups[i] + '_' + str(future) + '.tex')
            plt.show()
            
