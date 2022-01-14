import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import utils

def get_residue_parameters(delta_size, params, perc):

    residue = delta_size[6:-4] - params[0] - params[1] * delta_size[5:-5]
    respar = list(utils.residue_analysis(residue, False, lags, step_size, window_size))
    respar.append(sp.stats.t.ppf(perc, respar[0], respar[1], respar[2]))
    respar.append(np.sort(residue)[int(perc * len(residue))])
    return respar

### Main parameters ###
analyzed = True # Set to true to skip model derivation and load it from file
elements = 5 # Number of residue elements to save
perc = 0.9

### Videos and parameters ###
videos = ['mc', 'ge_cities', 'ge_tour', 'vp']
rates = np.arange(10, 51, 10)
fpss = [30, 60]
step_size = 20
window_size = 200
lags = 21

### Model derivation for each trace ###
m_const = np.zeros((len(videos), len(rates), 2))
m_slope = np.zeros((len(videos), len(rates), 2))
q_const = np.zeros((len(videos), len(rates), 2))
q_slope = np.zeros((len(videos), len(rates), 2))
if (not analyzed):
    vv = 0
    for video in videos:
        rr = 0
        for rate in rates:
            ff = 0
            for fps in fpss:
                filename = video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng'
                print(video, rate, fps)
                
                ### Import trace data ###
                dataframe = pd.read_csv('../traces/' + filename + '.csv', names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
                dataframe = dataframe.drop(['idx'], axis = 1)
                dataframe['time'] = dataframe['time'].diff(-1)
                dataframe['deltasize'] = dataframe['size'].diff(-1)
                dataframe['time'] = dataframe['time'].fillna(0)
                dataframe['deltasize'] = dataframe['deltasize'].fillna(0)
                dataframe['pastdelta'] = dataframe['deltasize'].shift(1)
                dataframe['pastdelta'] = dataframe['pastdelta'].fillna(0)
                
                ### Posterior analysis and fitting ###
                delta_size = np.asarray(dataframe['deltasize'].tolist()) / 1000
                
                indep = sm.add_constant(delta_size[5:-5])
                rlm_model = sm.RLM(delta_size[6:-4], indep, M=sm.robust.norms.HuberT())
                rlm_results = rlm_model.fit()
                m_const[vv, rr, ff] = rlm_results.params[0]
                m_slope[vv, rr, ff] = rlm_results.params[1]        
                
                qmodel = smf.quantreg('deltasize ~ pastdelta', dataframe).fit(q=perc)
                q_const[vv, rr, ff] = qmodel.params['Intercept'] / 1000 
                q_slope[vv, rr, ff] = qmodel.params['pastdelta']
                    
                    
                ff += 1
            rr += 1
        vv += 1
        
    np.savez('videoresults.npz', m_const, m_slope, q_const, q_slope)

param = np.load('videoresults.npz')
m_const = param['arr_0']
m_slope = param['arr_1']
q_const = param['arr_2']
q_slope = param['arr_3']

m_videodist = np.zeros((len(videos), len(rates), 2, elements))
m_ratedist = np.zeros((len(videos), len(rates), 2, elements))
m_singledist = np.zeros((len(videos), len(rates), 2, elements))
m_resdist = np.zeros((len(videos), len(rates), 2, elements))
m_framedist = np.zeros((len(videos), len(rates), 2, elements))
q_videodist = np.zeros((len(videos), len(rates), 2, elements))
q_ratedist = np.zeros((len(videos), len(rates), 2, elements))
q_singledist = np.zeros((len(videos), len(rates), 2, elements))
q_resdist = np.zeros((len(videos), len(rates), 2, elements))
q_framedist = np.zeros((len(videos), len(rates), 2, elements))
nodist = np.zeros((len(videos), len(rates), 2, elements))

m_singlepar = [np.mean(m_const), np.mean(m_slope)]
q_singlepar = [np.mean(q_const), np.mean(q_slope)]

### Analysis of model residues ###
vv = 0
for video in videos:
    m_videopar = [np.mean(m_const[vv, :, :]), np.mean(q_slope[vv, :, :])]
    q_videopar = [np.mean(q_const[vv, :, :]), np.mean(q_slope[vv, :, :])]
    rr = 0
    for rate in rates:
        m_ratepar = [np.mean(m_const[:, rr, :]), np.mean(m_slope[:, rr, :])]
        q_ratepar = [np.mean(q_const[:, rr, :]), np.mean(q_slope[:, rr, :])]
        ff = 0
        for fps in fpss:
            m_framepar = [np.mean(m_const[:, :, ff]), np.mean(m_slope[:, :, ff])]
            q_framepar = [np.mean(q_const[:, :, ff]), np.mean(q_slope[:, :, ff])]
            
            filename = video + '_' + str(rate) + 'mbps_' + str(fps) + 'fps.pcapng'
            print(video, rate, fps)
            
            ### Import trace data ###
            dataframe = pd.read_csv('../traces/' + filename + '.csv', names=['idx', 'size', 'time'], skiprows=0, delimiter=',')
            dataframe = dataframe.drop(['idx'], axis = 1)
            dataframe['time'] = dataframe['time'].diff(-1)
            dataframe['deltasize'] = dataframe['size'].diff(-1)
            dataframe['time'] = dataframe['time'].fillna(0)
            dataframe['deltasize'] = dataframe['deltasize'].fillna(0)
            
            delta_size = np.asarray(dataframe['deltasize'].tolist()) / 1000
            
            ### Residue distribution analysis (trace) ###
            m_resdist[vv, rr, ff, :] = get_residue_parameters(delta_size, [m_const[vv, rr, ff], m_slope[vv, rr, ff]], perc)     
            m_singledist[vv, rr, ff, :] = get_residue_parameters(delta_size, m_singlepar, perc)             
            m_videodist[vv, rr, ff, :] = get_residue_parameters(delta_size, m_videopar, perc)                   
            m_ratedist[vv, rr, ff, :] = get_residue_parameters(delta_size, m_ratepar, perc)                     
            m_framedist[vv, rr, ff, :] = get_residue_parameters(delta_size, m_framepar, perc)                     
            q_resdist[vv, rr, ff, :] = get_residue_parameters(delta_size, [q_const[vv, rr, ff], q_slope[vv, rr, ff]], perc)     
            q_singledist[vv, rr, ff, :] = get_residue_parameters(delta_size, q_singlepar, perc)             
            q_videodist[vv, rr, ff, :] = get_residue_parameters(delta_size, q_videopar, perc)                   
            q_ratedist[vv, rr, ff, :] = get_residue_parameters(delta_size, q_ratepar, perc)                     
            q_framedist[vv, rr, ff, :] = get_residue_parameters(delta_size, q_framepar, perc)                     
            nodist[vv, rr, ff, :] = get_residue_parameters(delta_size, [0, 0], perc)    
                            
            ff += 1
        rr += 1
    vv += 1


np.savez('fullresults.npz', m_const, m_slope, m_resdist, m_singledist, m_videodist, m_ratedist, m_framedist, q_resdist, q_singledist, q_videodist, q_ratedist, q_framedist, nodist)
