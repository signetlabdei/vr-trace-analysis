import numpy as np
import regression_plotter as reg 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
import tikzplotlib as tikz

class Simulator:
    
    def __init__(self, videos, rates, fpss, past_steps, feedback_latency, method, quantile, guard):
        self.videos = videos
        self.rates = rates
        self.guard = guard
        self.method = method
        self.past_steps = past_steps
        self.feedback_latency = feedback_latency
        self.quantile = quantile
        self.regressor = reg.RegressionPlotter(videos, rates, fpss)

        
    def __setup(self, frequency, frame_by_frame):
        models = []
        if (frame_by_frame):
            for i in range(frequency):
                models.append(self.regressor.regress(self.method, self.past_steps, 1, i + self.feedback_latency, self.quantile, self.guard))
        else:
            models.append(self.regressor.regress(self.method, self.past_steps, frequency, self.feedback_latency, self.quantile, self.guard))
        return models
    
    def set_quantile(self, quantile):
        self.quantile = quantile
        
    def simulate(self, frequency, frame_by_frame, video, rate, fps):
        filename = '../traces/' + video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng' + '.csv'
        df = pd.read_csv(filename, names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
        frame_size = df['size'].to_numpy()[self.guard : -self.guard]
        models = self.__setup(frequency, frame_by_frame)
        
        dist = getattr(sps, 'laplace')
        residue_q = []
        if (method != 'Quantile'):
            reg_res = reg.RegressionPlotter([video], [rate], [fps])
            if (frame_by_frame):
                for i in range(frequency):
                    params = reg.RegressionPlotter.fit_residue(reg_res.residue(models[i], self.past_steps, 1, i + self.feedback_latency, self.quantile, self.guard, rate, fps), 500)
                    residue_q.append(dist.ppf(self.quantile, loc=params[-2], scale=params[-1]))
            else:
                params = reg.RegressionPlotter.fit_residue(reg_res.residue(models[0], self.past_steps, frequency, self.feedback_latency, self.quantile, self.guard, rate, fps), 500)
                residue_q.append(dist.ppf(self.quantile, loc=params[-2], scale=params[-1]))
                
        schedule = np.zeros(len(frame_size))
        latency = np.zeros(len(frame_size))
        queue = np.zeros(len(frame_size))
        
        for i in range(len(frame_size)):
            if (np.mod(i, frequency) == frequency - 1):
                if (frame_by_frame):
                    for j in range(i, i + frequency):
                        if (j < len(schedule)):
                            schedule[j] = reg.RegressionPlotter.predict(frame_size[i - self.past_steps : i], models[j - i], rate, fps)
                            if (method != 'Quantile'):
                                schedule[j] += residue_q[j - i]
                    schedule[i] += queue[i - 1]
                else:
                    schedule[i : i + frequency] = reg.RegressionPlotter.predict(frame_size[i - self.past_steps : i], models[0], rate, fps) + queue[i - 1] / frequency
                    if (method != 'Quantile'):
                        schedule[i : i + frequency] += residue_q[0]
            queue[i] = np.max([0, queue[i - 1] + frame_size[i] - schedule[i]])
            
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
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)        
        
    def boxplot(data, palette, ticks, groups, xlabel, ylabel, ymax, line=-1, show=False, folder='', savename=''):
        # Latency boxplot
        plt.figure()
        N = len(groups)
        for i in range(N):
            bp = plt.boxplot(data[i], positions=np.arange(0, len(ticks), 1) * N + (i + 0.5 - N / 2) * 0.8, sym='', widths=0.6)
            Simulator.set_box_color(bp, palette[i])
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
    ### Main simulation parameters ###
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
    
    latencies = [[], []]
    schedules = [[], []]
    labels = []
    
    sim = Simulator(videos, rates, fpss, past_steps, feedback_latency, method, quantile, guard)
    #for i in range(max_granularity):
        #labels.append(str(i + 1))
        #latency, schedule = sim.simulate(i + 1, True, video, rate, fps)
        #latencies[0].append(latency)
        #schedules[0].append(schedule)
        #latency, schedule = sim.simulate(i + 1, False, video, rate, fps)
        #latencies[1].append(latency)
        #schedules[1].append(schedule)
        
    #Simulator.boxplot(latencies, palette, labels, groups, 'Granularity', 'Latency (ms)', 30, line=1000 / fps, folder='figures/', savename='schedule_' + method + '_latency_granularity')
    #Simulator.boxplot(schedules, palette, labels, groups, 'Granularity', 'Scheduled capacity (Mb/s)', 50, folder='figures/', savename='schedule_' + method + '_rate_granularity')
    
    percentiles = np.arange(0.9, 1, 0.005)
    

    perc_line = [0.9, 0.95, 0.99]
    for future in [6]:
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
            
