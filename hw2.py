# Bobby Saba - plot data from in-class flux exercise 

# import required packages
import numpy as np 
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import metpy.constants as constants 
from matplotlib.offsetbox import AnchoredText

# define path 
path = '/Users/robert.saba/OneDrive - University of Oklahoma/METR_5603/'

# constants 
rho = 1.13
cp = 1004
L = constants.Lv.magnitude
#%%
# set a list of headers for the non-sm dataset
headers = ['date', 'time', 'u', 'v', 'w', 'T', 'co_2', 'q']

# read in the data 
df = pd.read_csv(f'{path}HW 2/data/data20200810.dat', delimiter = ' ', names = headers)

df_dt = []

for i in range(len(df)):
    try: 
        df_dt.append(dt.datetime.strptime(df['date'][i] + df['time'][i], '%Y-%m-%d%H:%M:%S.%f')) 
    except:
        df_dt.append(dt.datetime.strptime(df['date'][i] + df['time'][i], '%Y-%m-%d%H:%M:%S'))


df = pd.DataFrame({'u': df['u'].values, 'v': df['v'].values, 'w': df['w'].values, 'co_2': df['co_2'].values, 'q': df['q'].values, 'T': df['T'].values,
                   'stamps': [i.timestamp() for i in df_dt]}, 
                  index = pd.to_datetime(df_dt))

df_time_seconds = [(df_dt[i] - df_dt[0]).total_seconds() for i in range(len(df_dt))]
#%%
# q2 data analysis
sm142 = pd.read_csv(f'{path}HW 2/data/SM_142_2020-08-10_0001.dat', header = None, delimiter = ',')
sm143 = pd.read_csv(f'{path}HW 2/data/SM_143_2020-08-10_0001.dat', header = None, delimiter = ',')

# array of dts
sm142_dt = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in sm142[0]]

sm142_seconds = [(sm142_dt[i] - sm142_dt[0]).total_seconds() for i in range(len(sm142_dt))]


# interpolate the first file times to the second file times 
stamps_1 = [i.timestamp() for i in df_dt]
stamps_2 = [i.timestamp() for i in sm142_dt]

# plot radiative budget 
fig = plt.figure(figsize = (12, 8))

# total 
tot = sm142[1] - sm142[2] - sm142[12] + sm142[11]

plt.plot(sm142_dt, sm142[1], label = 'indident shortwave', c = 'orange')
plt.plot(sm142_dt, -sm142[2], label = 'reflected shortwave', c = 'b')
plt.plot(sm142_dt, -sm142[12], label = 'terrestrial longwave', c = 'r')
plt.plot(sm142_dt, sm142[11], label = 'atmospheric longwave', c = 'lime')
plt.plot(sm142_dt, tot, label = 'net radiation')

plt.title('Radiative Budget Observations')

plt.xlabel('time (local)')
plt.ylabel('$Wm^2$')

plt.xlim(sm142_dt[0], sm142_dt[-1])

plt.grid(ls = '--', color = 'gray')

# format x axis
myFmt = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)

plt.legend(loc = 'upper left')

plt.hlines(0, sm142_dt[0], sm142_dt[-1], color = 'k', ls = '--')

plt.savefig(f'{path}HW 2/radiative_budget.png', dpi = 250)
#%% compute resample averages 
# loop through the following for each time 
for i in [5, 30]:
    # # set the timedelta in seconds 
    # interval = i * 60 * 10 
    
    # int_start = np.arange(0, len(df), interval)
    
    # int_end = int_start + interval
    
    # dts = [df_dt[0] + dt.timedelta(seconds = i * interval/10) for i in range(len(int_start))]
    
    # # create arrays to store variables 
    # tke = np.ones_like(int_start) * np.nan
    
    # uw = np.ones_like(int_start) * np.nan
    
    # vw = np.ones_like(int_start) * np.nan
    
    # tw = np.ones_like(int_start) * np.nan
    
    # qw = np.ones_like(int_start) * np.nan
    
    # co2w = np.ones_like(int_start) * np.nan
    
    # for idx in range(len(tke)):
    #      # find where the time (seconds) is in our window 
    #      idxx = np.arange(int_start[idx], int_end[idx])
                  
    #      u_prime = (df['u'][idxx] - df['u'][idxx].mean())
    #      w_prime = (df['w'][idxx] - df['w'][idxx].mean())
    #      v_prime = (df['v'][idxx] - df['v'][idxx].mean())
    #      t_prime = (df['T'][idxx] - df['T'][idxx].mean())
    #      q_prime = (df['q'][idxx] - df['q'][idxx].mean())
    #      co2_prime = (df['co_2'][idxx] - df['q'][idxx].mean())
         
    #      # calculate variables above 
    #      uw[idx] = np.nanmean(u_prime * w_prime)
    #      vw[idx] = np.nanmean(v_prime * w_prime)
    #      tw[idx] = np.nanmean(t_prime * w_prime)
    #      qw[idx] = np.nanmean(q_prime * w_prime)
    #      co2w[idx] = np.nanmean(co2_prime * w_prime)
         
    #      # tke 
    #      tke[idx] = 0.5 * (np.nanmean(u_prime ** 2) * np.nanmean(v_prime ** 2) * np.nanmean(w_prime ** 2))

    # # tke plot 
    # fig, ax = plt.subplots(3, 2, figsize = (24, 16), sharex = True, constrained_layout = True)
    
    # # plot each variable 
    # ax[0,0].plot(dts, tke, c = 'k')
    # ax[0,1].plot(dts, uw, c = 'b')
    # ax[1,0].plot(dts, vw, c = 'b')
    # ax[1,1].plot(dts, tw, c = 'r')
    # ax[2,0].plot(dts, qw, c = 'g')
    # ax[2,1].plot(dts, co2w, c = 'orange')
    
    # # add titles 
    # ax[0,0].set_title('Turbulent Kinetic Energy')
    # ax[0,1].set_title('Vertical Kinematic Flux of Horizontal Momentum: U')
    # ax[1,0].set_title('Vertical Kinematic Flux of Horizontal Momentum: V')
    # ax[1,1].set_title('Vertical Kinematic Temperature Flux')
    # ax[2,0].set_title('Vetical Kinematic Moisture Flux')
    # ax[2,1].set_title('Vertical Kinematic CO$_2$ Flux')
    
    # # add grid 
    # for k in range(3):
    #     for j in range(2):
    #         ax[k,j].grid(ls = '--', color = 'gray')
    #         ax[k,j].hlines(0, dts[0], dts[-1], ls = '--', color = 'k')
            
    # # format x axis
    # myFmt = mdates.DateFormatter('%H:%M')
    # ax[0,0].xaxis.set_major_formatter(myFmt)
    
    # # add x labels 
    # ax[2,0].set_xlabel('time (local)')
    # ax[2,1].set_xlabel('time (local)')
    
    # # add y labels 
    # ax[0,0].set_ylabel('$m^2s^{-2}$')
    # ax[0,1].set_ylabel('$ms^{-1}$')
    # ax[1,0].set_ylabel('$ms^{-1}$')
    # ax[1,1].set_ylabel('˚C')
    # ax[2,0].set_ylabel('$g/kg$')
    # ax[2,1].set_ylabel('ppm')
    
    # # set ylims 
    # ax[0,0].set_ylim(-0.5,3.5)
    # ax[0,1].set_ylim(-0.25, 0.25)
    # ax[1,0].set_ylim(-0.6, 0.6)
    # ax[1,1].set_ylim(-0.2, 0.2)
    # ax[2,0].set_ylim(-0.25, 0.25)
    # ax[2,1].set_ylim(-2, 2)
    
    # # set xlim 
    # ax[0,0].set_xlim(dts[0], dts[-1])
    
    # # add plot labels
    # ax[0,0].add_artist(AnchoredText('A', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    # ax[0,1].add_artist(AnchoredText('B', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    # ax[1,0].add_artist(AnchoredText('C', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    # ax[1,1].add_artist(AnchoredText('D', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    # ax[2,0].add_artist(AnchoredText('E', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    # ax[2,1].add_artist(AnchoredText('F', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    
    # plt.savefig(f'{path}HW 2/tke_panels_{i}.png', dpi = 250)
    
    # prep energy budget data - set new interval
    interval = i
    
    int_start = np.arange(0, len(sm142), interval)
    
    # ending interval 
    int_end = int_start + interval
    
    # datetimes for plotting
    dts = np.array([sm142_dt[0] + dt.timedelta(minutes = i * interval) for i in range(len(int_start))])
    
    # interpolate necessary variables
    w_interp = np.interp(stamps_2, stamps_1, df['w'].values)
    q_interp = np.interp(stamps_2, stamps_1, df['q'].values)
    t_interp = np.interp(stamps_2, stamps_1, df['T'].values)
    
    # pull the soil moisture temp 
    sm = sm143[14].values
    
    # create arrays for H and LE 
    H = np.ones_like(int_start) * np.nan
    LE = np.ones_like(int_start) * np.nan
    G = np.ones_like(int_start) * np.nan
    
    # create net radiation array
    net_rad = np.ones_like(int_start) * np.nan
    
    # loop and create arrays 
    for idx in range(len(int_start)):
        # find indices to use 
        idxx = np.arange(int_start[idx], int_end[idx], 1)
        
        # compute prime values 
        t_prime = (t_interp[idxx] - t_interp[idxx].mean())
        q_prime = (q_interp[idxx] - q_interp[idxx].mean())
        w_prime = (w_interp[idxx] - w_interp[idxx].mean())
        
        # add final computed values to arrays 
        H[idx] = rho * cp * np.nanmean(w_prime * t_prime)
        
        LE[idx] = rho * L/1000 * np.nanmean(w_prime * q_prime)
        
        G[idx] = np.nanmean(sm[idxx])
        
        net_rad[idx] = np.nanmean(sm142[10][idxx])

    # close any previously used figure 
    plt.close() 
    
    # plot up the data 
    plt.figure(figsize = (12, 8))
    
    plt.plot(dts, H, label = 'sensible heat flux', c = 'r')
    plt.plot(dts, LE, label = 'latent heat flux', c = 'b')
    plt.plot(dts, G, label = 'soil heat flux', c = 'lime')
    plt.plot(dts, G + LE + H, label = 'net heat flux', c = 'orange')
    
    plt.grid(color = 'gray', ls = '--')
    
    plt.title('Surface Energy Budget')
    
    plt.xlabel('time (local)')
    plt.ylabel('$Wm^{-2}$')
    
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.legend(loc = 'upper left')
    
    plt.xlim(dts[0], dts[-1])
    
    plt.savefig(f'{path}HW 2/energy_budget_{i}.png', dpi = 250)
    
    # break up day and night array
    day_i = np.where((dts >= dt.datetime(2020, 8, 10, 8)) & (dts <= dt.datetime(2020, 8, 10, 18)))[0]
    
    # plot them up 
    fig, ax = plt.subplots(1, 2, figsize = (14, 8), sharex = False, sharey = True, constrained_layout = True)
    
    ax[0].plot(dts[day_i], H[day_i], label = 'sensible heat flux', c = 'r')
    ax[0].plot(dts[day_i], LE[day_i], label = 'latent heat flux', c = 'b')
    ax[0].plot(dts[day_i], G[day_i], label = 'soil heat flux', c = 'lime')
    ax[0].plot(dts[day_i], G[day_i] + LE[day_i] + H[day_i], label = 'net heat flux', c = 'orange')
    ax[0].plot(dts[day_i], net_rad[day_i], c = 'k', label = 'observed net heat flux')
    
    # mask the arrays for night
    H[day_i] = np.nan
    LE[day_i] = np.nan
    G[day_i] = np.nan
    net_rad[day_i] = np.nan
    
    ax[1].plot(dts, H, label = 'sensible heat flux', c = 'r')
    ax[1].plot(dts, LE, label = 'latent heat flux', c = 'b')
    ax[1].plot(dts, G, label = 'soil heat flux', c = 'lime')
    ax[1].plot(dts, G + LE + H, label = 'net heat flux', c = 'orange')
    ax[1].plot(dts, net_rad, c = 'k', label = 'observed net heat flux')
    
    ax[0].set_title('Daytime Energy Budget Components')
    ax[1].set_title('Nighttime Energy Budget Components')
    
    ax[0].set_xlim(dts[day_i][0], dts[day_i][-1])
    ax[1].set_xlim(dts[0], dts[-1])
    
    for k in range(2):
        ax[k].grid(ls = '--', color = 'gray')
    
        ax[k].xaxis.set_major_formatter(myFmt)
        
        ax[k].set_xlabel('time (local)')
        
    ax[0].set_ylabel('$Wm^{-2}$')
    
    # add plot labels
    ax[0].add_artist(AnchoredText('A', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    ax[1].add_artist(AnchoredText('B', loc = 'upper right', frameon = False, prop = {'fontsize': 14, 'weight': 'bold'}))
    
    ax[0].legend(loc = 'upper left')
    ax[1].legend(loc = 'upper left')
    
    plt.savefig(f'{path}HW 2/day_night_{i}.png', dpi = 250)























