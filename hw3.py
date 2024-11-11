# Bobby Saba - hw 3 code 

# import required packages 
import numpy as np
import netCDF4 as nc 
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

# create path 
path = '/Users/robert.saba/OneDrive - University of Oklahoma/METR_5603/'

# open the proper files for the first comparison 
p, z, T, Td, wdir, wspd = np.loadtxt(f'{path}HW 3/Data/radiosondes/sharppyv2_20240807_1506.txt', skiprows = 8, delimiter = ',', unpack = True)
p2, z2, T2, Td2, wdir2, wspd2 = np.loadtxt(f'{path}HW 3/Data/radiosondes/sharppyv2_20240807_1556.txt', skiprows = 8, delimiter = ',', unpack = True)

# compute u and v from the sonde 
u = -wspd * np.sin(wdir * np.pi/180)
v = -wspd * np.cos(wdir * np.pi/180)

u2 = -wspd2 * np.sin(wdir2 * np.pi/180)
v2 = -wspd2 * np.cos(wdir2 * np.pi/180)

cs3 = nc.Dataset(f'{path}HW 3/Data/CS3D/KAEFS5N938UACMTascent.c1.20240807.150706.cdf')
cs3_2 = nc.Dataset(f'{path}HW 3/Data/CS3D/KAEFS5N938UACMTascent.c1.20240807.155632.cdf')

cswx = nc.Dataset(f'{path}HW 3/Data/SWX/KAEFS5FA3KEC4HPRCMTascent.c1.20240807.150704.cdf')
cswx2 = nc.Dataset(f'{path}HW 3/Data/SWX/KAEFS5FA3KEC4HPRCMTascent.c1.20240807.155631.cdf')

# plot the comparison of thermo 
fig, ax = plt.subplots(1, 2, figsize = (10, 12), sharex = True, sharey = True, constrained_layout = True)

ax[0].plot(T[z <= cs3['alt'][-1]], z[z <= cs3['alt'][-1]], c = 'r', label = 'radiosonde - T')
ax[0].plot(cs3['tdry'][:] - 273.15, cs3['alt'][:], c = 'r', ls = '--', label = 'coptersonde 3D - T')
ax[0].plot(cswx['tdry'][:] - 273.15, cswx['alt'][:], c = 'r', ls = ':', label = 'coptersonde swx - T')

ax[0].plot(Td[z <= cs3['alt'][-1]], z[z <= cs3['alt'][-1]], c = 'g', label = 'radiosonde - Td')
ax[0].plot(cs3['Td'][:], cs3['alt'][:], c = 'g', ls = '--', label = 'coptersonde 3D - Td')
ax[0].plot(cswx['Td'][:], cswx['alt'][:], c = 'g', ls = ':', label = 'coptersonde swx - Td')

ax[1].plot(T2[z2 <= cs3_2['alt'][-1]], z2[z2 <= cs3_2['alt'][-1]], c = 'r')
ax[1].plot(cs3_2['tdry'][:] - 273.15, cs3_2['alt'][:], c = 'r', ls = '--')
ax[1].plot(cswx2['tdry'][:] - 273.15, cswx2['alt'][:], c = 'r', ls = ':')

ax[1].plot(Td2[z2 <= cs3_2['alt'][-1]], z2[z2 <= cs3_2['alt'][-1]], c = 'g')
ax[1].plot(cs3_2['Td'][:], cs3_2['alt'][:], c = 'g', ls = '--')
ax[1].plot(cswx2['Td'][:], cswx2['alt'][:], c = 'g', ls = ':')

# add plot details 
for i in range(0, 2):
    ax[i].grid(ls = '--', c = 'gray')
    ax[i].set_xlabel('T/T$_d$ (˚C)')


# legend lines - tornadic/non-tornadic
legend_lines = [Line2D([0], [0], color = 'black', lw = 2),
                Line2D([0], [0], color = 'black', lw = 2, linestyle = '--'),
                Line2D([0], [0], color = 'black', lw = 2, linestyle = ':'),
                Line2D([0], [0], color = 'red', lw = 2),
                Line2D([0], [0], color = 'green', lw = 2)]

legend_labels = ['radiosonde', 'coptersonde 3D', 'coptersonde swx', 'T', 'Td']

# add legend
fig.legend(legend_lines, legend_labels, ncols = 5, loc = 'lower center', bbox_to_anchor = (0.5, 0.05))

# axis labels 
ax[0].set_ylabel('zagl (m)')

# titles 
ax[0].set_title('Radiosonde/Copter Comparison\n20240807 @ 1506')
ax[1].set_title('Radiosonde/Copter Comparison\n20240807 @ 1556')

plt.savefig(f'{path}HW 3/sonde_comp.png', dpi = 250)


# plot the comparison of kinematic
fig, ax = plt.subplots(1, 2, figsize = (10, 12), sharex = True, sharey = True, constrained_layout = True)

ax[0].plot(u[z <= cs3['alt'][-1]], z[z <= cs3['alt'][-1]], c = 'orange')
ax[0].plot(cs3['wind_u'][:], cs3['alt'][:], c = 'orange', ls = '--')
ax[0].plot(cswx['wind_u'][:], cswx['alt'][:], c = 'orange', ls = ':')

ax[0].plot(v[z <= cs3['alt'][-1]], z[z <= cs3['alt'][-1]], c = 'b')
ax[0].plot(cs3['wind_v'][:], cs3['alt'][:], c = 'b', ls = '--')
ax[0].plot(cswx['wind_v'][:], cswx['alt'][:], c = 'b', ls = ':')

ax[1].plot(u2[z2 <= cs3_2['alt'][-1]], z2[z2 <= cs3_2['alt'][-1]], c = 'orange')
ax[1].plot(cs3_2['wind_u'][:], cs3_2['alt'][:], c = 'orange', ls = '--')
ax[1].plot(cswx2['wind_u'][:], cswx2['alt'][:], c = 'orange', ls = ':')

ax[1].plot(v2[z2 <= cs3_2['alt'][-1]], z2[z2 <= cs3_2['alt'][-1]], c = 'b')
ax[1].plot(cs3_2['wind_v'][:], cs3_2['alt'][:], c = 'b', ls = '--')
ax[1].plot(cswx2['wind_v'][:], cswx2['alt'][:], c = 'b', ls = ':')

# add plot details 
for i in range(0, 2):
    ax[i].grid(ls = '--', c = 'gray')
    ax[i].set_xlabel('u/v (m/s)')


# legend lines - tornadic/non-tornadic
legend_lines = [Line2D([0], [0], color = 'black', lw = 2),
                Line2D([0], [0], color = 'black', lw = 2, linestyle = '--'),
                Line2D([0], [0], color = 'black', lw = 2, linestyle = ':'),
                Line2D([0], [0], color = 'orange', lw = 2),
                Line2D([0], [0], color = 'b', lw = 2)]

legend_labels = ['radiosonde', 'coptersonde 3D', 'coptersonde swx', 'u-component', 'v-component']

# add legend
fig.legend(legend_lines, legend_labels, ncols = 5, loc = 'lower center', bbox_to_anchor = (0.5, 0.05))

# axis labels 
ax[0].set_ylabel('zagl (m)')

# titles 
ax[0].set_title('Radiosonde/Copter Comparison\n20240807 @ 1506')
ax[1].set_title('Radiosonde/Copter Comparison\n20240807 @ 1556')

plt.savefig(f'{path}HW 3/sonde_comp2.png', dpi = 250)

# 
# close the files 
cs3.close()
cs3_2.close()

cswx.close()
cswx2.close()