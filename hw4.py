# Bobby Saba - Obs HW 4

# import packages 
import numpy as np 
import netCDF4 as nc
import datetime as dt 
from datetime import timezone
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm

# create path 
path = '/Users/robert.saba/OneDrive - University of Oklahoma/METR_5603/'

# question 1a
# create the figure
fig, ax = plt.subplots(3, 1, figsize = (12, 15), sharex = True, sharey = True)

# loop through the ax and plot the data 
for i in range(3):
    # read in the dataset 
    nc_data = nc.Dataset(f'{path}HW 4/Problem 1/dltruckdlcsmwindsDL1.a{i + 1}.20241028.000000.cdf')
    
    # pull the times 
    t = np.array([dt.datetime.fromtimestamp(int(x) + nc_data['base_time'][:], tz = timezone.utc) for x in nc_data['time_offset'][:]])
    
    # pull the data
    wspd = nc_data['wspd'][:]
    h = nc_data['height'][:]
    
    # plot the data 
    plot = ax[i].pcolormesh(t, h, wspd.T, vmin = 0, vmax = 35, cmap = 'magma_r')
    
    # add a title 
    ax[i].set_title(f'A{i+1} Horizontal Wind Speed')
    
    # add y ticks 
    ax[i].set_ylim(0, 6)
    ax[i].set_ylabel('height (km)')

    # close the dataset
    nc_data.close()
    
# format the x ticks 
myFmt = mdates.DateFormatter('%H:%M')
ax[0].xaxis.set_major_formatter(myFmt)

# add an axes for a master colorbar 
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.03, 0.77])

# add a colorbar 
cbar = fig.colorbar(plot, cax = cbar_ax, ticks = np.arange(0, 36, 5), extend = 'max')
cbar.set_label('[ms$^{-1}$]')

# add axis label 
ax[2].set_xlabel('time (utc)')

# save figure 
plt.savefig(f'{path}HW 4/q1.png', dpi = 250)
#%%
# question 1b

# create the figure
fig, ax = plt.subplots(3, 1, figsize = (12, 15), sharex = True, sharey = True)

# loop through the ax and plot the data 
for i in range(3):
    # read in the dataset 
    nc_data = nc.Dataset(f'{path}HW 4/Problem 1/dltruckdlcsmwindsDL1.a{i + 1}.20241028.000000.cdf')
    
    # pull the times 
    t = np.array([dt.datetime.fromtimestamp(int(x) + nc_data['base_time'][:], tz = timezone.utc) for x in nc_data['time_offset'][:]])
    
    # pull the data
    wspd = nc_data['w'][:]
    h = nc_data['height'][:]
    
    # plot the data 
    plot = ax[i].pcolormesh(t, h, wspd.T, vmin = -4, vmax = 4, cmap = 'RdBu_r')
    
    # add a title 
    ax[i].set_title(f'A{i+1} Vertical Wind Speed')
    
    # add y ticks 
    ax[i].set_ylim(0, 6)
    ax[i].set_ylabel('height (km)')

    # close the dataset
    nc_data.close()
    
# format the x ticks 
myFmt = mdates.DateFormatter('%H:%M')
ax[0].xaxis.set_major_formatter(myFmt)

# add an axes for a master colorbar 
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.03, 0.77])

# add a colorbar 
cbar = fig.colorbar(plot, cax = cbar_ax, ticks = np.arange(-4, 5, 1), extend = 'both')
cbar.set_label('[ms$^{-1}$]')

# add axis label 
ax[2].set_xlabel('time (utc)')

# save figure 
plt.savefig(f'{path}HW 4/q1_w.png', dpi = 250)
#%%
# question 2a 

# read in the data 
nc_data = nc.Dataset(f'{path}HW 4/clampstropoe10.aeri.v0.C2.20241103.002005.nc')

# pull the times/heights
t = np.array([dt.datetime.fromtimestamp(int(x) + nc_data['base_time'][:], tz = timezone.utc) for x in nc_data['time_offset'][:]])

# create plot bounds 
start = dt.datetime(2024, 11, 3, 0, tzinfo = timezone.utc)
end = dt.datetime(2024, 11, 3, 7, tzinfo = timezone.utc)

h = nc_data['height'][:]

T, H = np.meshgrid(t, h)

# pull the other variables 
temp = nc_data['temperature'][:]
sig_temp = nc_data['sigma_temperature'][:]

wvmr = nc_data['waterVapor'][:]
sig_wvmr = nc_data['sigma_waterVapor'][:]

# create the figure 
fig, ax = plt.subplots(2, 1, figsize = (12, 10), sharex = True, sharey = True)

# plot the data 
temp_plt = ax[0].pcolormesh(t, h, temp.T, vmin = 0, vmax = 20, cmap = 'autumn_r')

# add colorbar 
temp_cbar = plt.colorbar(temp_plt, ax = ax[0], ticks = np.arange(0, 21, 4), extend = 'both')
temp_cbar.set_label('[˚C]')

# add sigma contour 
temp_cont = ax[0].contour(T, H, sig_temp.T, [2], c = 'k')
ax[0].clabel(temp_cont, inline = True)

# plot the data 
wvmr_plt = ax[1].pcolormesh(t, h, wvmr.T, vmin = 0, vmax = 10, cmap = 'summer_r')

# add colorbar 
wvmr_cbar = plt.colorbar(wvmr_plt, ax = ax[1], ticks = np.arange(0, 11, 2), extend = 'max')
wvmr_cbar.set_label('[g/kg]')

# add sigma contour 
wvmr_cont = ax[1].contour(T, H, sig_wvmr.T, [2], c = 'k')
ax[1].clabel(wvmr_cont, inline = True)

# axes limits
ax[0].set_ylim(0, 4)
ax[0].set_xlim(start, end)

# format x axis 
myFmt = mdates.DateFormatter('%H:%M')
ax[0].xaxis.set_major_formatter(myFmt)

# axis lables 
ax[0].set_ylabel('height (km)')
ax[1].set_ylabel('height (km)')

ax[1].set_xlabel('time (utc)')

# titles 
ax[0].set_title('TROPoe Temperature')
ax[1].set_title('TROPoe Water Vapor Mixing Ratio')

# close the dataset
nc_data.close()

# save fig 
plt.savefig(f'{path}HW 4/q2_trop.png', dpi = 250)
#%%
# question 2b

# create wdir colorbar 
cmap = plt.colormaps['hsv']
norm = BoundaryNorm(np.arange(0, 361, 10), ncolors = cmap.N, clip = True)

# read in the data 
nc_data = nc.Dataset(f'{path}HW 4/clampsdlvadC2.c1.20241103.000000.cdf')

# pull the times/heights
t = np.array([dt.datetime.fromtimestamp(int(x) + nc_data['base_time'][:], tz = timezone.utc) for x in nc_data['time_offset'][:]])

h = nc_data['height'][:]

T, H = np.meshgrid(t, h)

# pull the other variables 
wspd = nc_data['wspd'][:]
wdir = nc_data['wdir'][:]

snr = nc_data['intensity'][:]

# create the figure 
fig, ax = plt.subplots(2, 1, figsize = (12, 10), sharex = True, sharey = True)

# plot the data 
wspd_plt = ax[0].pcolormesh(t, h, wspd.T, vmin = 0, vmax = 35, cmap = 'magma_r')
wdir_plt = ax[1].pcolormesh(t, h, wdir.T, cmap = cmap, norm = norm)

# add colorbar 
wspd_cbar = plt.colorbar(wspd_plt, ax = ax[0], ticks = np.arange(0, 36, 5), extend = 'max')
wspd_cbar.set_label('[m/s]')

# add colorbar 
wdir_cbar = plt.colorbar(wdir_plt, ax = ax[1], ticks = np.arange(0, 361, 90))
wdir_cbar.set_label('[˚]')

# add sigma contour 
wspd_cont = ax[0].contour(T, H, snr.T, [1.01], c = 'k')
ax[0].clabel(wspd_cont, inline = True)

wdir_cont = ax[1].contour(T, H, snr.T, [1.01], c = 'k')
ax[1].clabel(wspd_cont, inline = True)

# axes limits
ax[0].set_ylim(0, 2)
ax[0].set_xlim(start, end)

# format x axis 
myFmt = mdates.DateFormatter('%H:%M')
ax[0].xaxis.set_major_formatter(myFmt)

# axis lables 
ax[0].set_ylabel('height (km)')
ax[1].set_ylabel('height (km)')

ax[1].set_xlabel('time (utc)')

# titles 
ax[0].set_title('VAD Horizontal Wind Speed')
ax[1].set_title('VAD Horizontal Wind Direction')

# close the dataset
nc_data.close()

# save fig 
plt.savefig(f'{path}HW 4/q2_vad.png', dpi = 250)





















