# Bobby Saba - advanced obs hw1

# import required packages
import pyart
import warnings
import tempfile
import numpy as np
import netCDF4 as nc
import datetime as dt
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.units import units
from datetime import timezone
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from metpy.plots import SkewT, Hodograph
import cartopy.io.shapereader as shpreader
from boto.s3.connection import S3Connection
from siphon.simplewebservice.wyoming import WyomingUpperAir
from sharppy.sharptab import profile, interp, winds, utils, params

# function for wind params
def wind_params(u, v, storm_motion, z):
    # u: array of u-component of ground relative winds (from data array w/metpy units)
    # v: array of v-component of ground relative winds (from data array w/metpy units)
    # storm_motion: array of right moving storm motion 
        # storm_motion[0]: u-component
        # storm_motion[1]: v-component
    # returns:
        # vert_helicity: vertical helicity profile (m^2s^-2)
        # dvdz: vertial wind shear (ms^-1)
        # bs: bulk shear at each level (knots)
        
    u_mps = u / 1.944
    v_mps = v / 1.944

    # start by defining your arrays to store vars
    dvdz = np.zeros_like(z) * np.nan
    bs = np.zeros_like(z) * np.nan
    
    # loop through the heighs to calculate sequential SRH in the profile
    for i in range(1, len(z)):
        
        # compute vertical shear 
        dvdz[i] = (np.sqrt(((u_mps[i] - u_mps[i-1]) ** 2) + ((v_mps[i] - v_mps[i-1]) ** 2))/(z[i] - z[i-1]))
        
        # compute bulk shear
        bs[i] = np.sqrt(((u[i] - u[0]) ** 2) + ((v[i] - v[0]) ** 2))
    
    return dvdz, bs

# madke a function to qc radiosonde launch ------- not my proudest work, but it gets the job done
def gosh(arr, arr2):
    unique_indices = {}
    
    # Iterate through the array and store the index of the first occurrence of each unique element
    for i, value in enumerate(arr):
        if value not in unique_indices:
            if np.isnan(arr2[i]) == False:
                unique_indices[value] = i
    
    # Extract the values in sorted order (increasing)
    sorted_unique_values = sorted(unique_indices.keys())
    
    # Return the indices corresponding to the increasing, unique values
    return np.array([unique_indices[value] for value in sorted_unique_values])

# Helper function for the search
def _nearestDate(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))

# func to pull the data
def get_radar_data(site, datetime_t):
    # site : string of four letter radar designation (ex: 'KTLX')
    # datetime_t : datetime in which you'd like the closest scan
    # returns:
    # radar : Py-ART Radar Object of scan closest to the queried datetime

    #First create the query string for the bucket knowing
    #how NOAA and AWS store the data
    my_pref = datetime_t.strftime('%Y/%m/%d/') + site

    #Connect to the bucket
    conn = S3Connection(anon = True)
    bucket = conn.get_bucket('noaa-nexrad-level2')

    #Get a list of files
    bucket_list = list(bucket.list(prefix = my_pref))

    #we are going to create a list of keys and datetimes to allow easy searching
    keys = []
    datetimes = []

    #populate the list
    for i in range(len(bucket_list)):
        this_str = str(bucket_list[i].key)
        if 'gz' in this_str:
            endme = this_str[-22:-4]
            fmt = '%Y%m%d_%H%M%S_V0'
            date_t = dt.datetime.strptime(endme, fmt).replace(tzinfo = timezone.utc)
            datetimes.append(date_t)
            keys.append(bucket_list[i])

        if this_str[-3::] == 'V06':
            endme = this_str[-19::]
            fmt = '%Y%m%d_%H%M%S_V06'
            date_t = dt.datetime.strptime(endme, fmt).replace(tzinfo = timezone.utc)
            datetimes.append(date_t)
            keys.append(bucket_list[i])

    #find the closest available radar to your datetime
    closest_datetime = _nearestDate(datetimes, datetime_t)
    index = datetimes.index(closest_datetime)

    localfile = tempfile.NamedTemporaryFile()
    keys[index].get_contents_to_filename(localfile.name)
    radar = pyart.io.read(localfile.name)
    return radar

# hide warnings
warnings.filterwarnings('ignore')

# set path 
path = '/Users/robert.saba/OneDrive - University of Oklahoma/METR_5603/'

# read in file
nc_data = nc.Dataset(f'{path}hw1_sounding.cdf')

# read in height
h = nc_data['Height'][:]
t = nc_data['Temperature'][:]

# find indices to use
idx = gosh(h, t)

# pull the variables at the right heights
h = h[idx]
t = t[idx]
p = nc_data['Pressure'][idx]
rh = nc_data['RH'][idx]
wspd = nc_data['Wspd'][idx]
wdir = nc_data['Wdir'][idx]

# convert humidity to dewpoint
td = mpcalc.dewpoint_from_relative_humidity(units.Quantity(t, 'degC'), units.Quantity(rh, 'percent'))

# create sharppy profile - requested time
prof = profile.create_profile(profile = 'convective', pres = p, hght = h, tmpc = t, dwpc = td.magnitude, wspd = wspd * 1.944, wdir = wdir, 
                              strictQC = False, date = dt.datetime(2020, 8, 28, 23, 4, 27))
#%%
# log launch date 
launch_dt = dt.datetime(2024, 9, 11, 12).replace(tzinfo = timezone.utc)

# pull sounding from Wyo 
oun_launch = WyomingUpperAir.request_data(launch_dt, 'OUN')

# create profile from OUN sounding 
prof = profile.create_profile(profile = 'convective', pres = oun_launch['pressure'], hght = oun_launch['height'], tmpc = oun_launch['temperature'], 
                              dwpc = oun_launch['dewpoint'], u = oun_launch['u_wind'], v = oun_launch['v_wind'], strictQC = False, 
                              date = launch_dt)
#%%
######################### CREATE AXES OBJECTS (excluding radar) #########################
# skew-t
fig = plt.figure(figsize = (21, 14))
skew = SkewT(fig, rotation = 45, rect = (0.05, 0.05, 0.50, 0.90))

# hodograph
hodo_ax = plt.axes((0.5, 0.6, 0.35, 0.35))
h = Hodograph(hodo_ax, component_range = 30)

# vertical helicity
helicity_ax = plt.axes((0.58, 0.27, 0.08, 0.28))

# vertical shear
dvdz_ax = plt.axes((0.71, 0.27, 0.08, 0.28))

# bulk shear
bulk_shear_ax = plt.axes((0.86, 0.27, 0.08, 0.28))

# theta/theta_e
theta_ax = plt.axes((1.01, 0.27, 0.08, 0.28))

# create an axis for the text tables to go 
parcel_ax = plt.axes((0.55, 0.03, 0.45, 0.2))

# hide the text axis (we just use this for positioning)
parcel_ax.axis('off')
######################### END OF CREATING AXES OBJECTS #########################

print('Formatting plot....\n')
######################### FORMAT AXES #########################
# AXES TITLES
#####------------------------------------------------#####
# skew-t
skew.ax.set_title('Skew-T Log-P', fontsize = 16)

# hodograph
hodo_ax.set_title('0-6km Ground Relative Wind Hodograph')

# vertical helicity
helicity_ax.set_title('Vertical Helicity Profile')

# vertical shear
dvdz_ax.set_title('Vertical Wind Shear')

# bulk shear
bulk_shear_ax.set_title('Bulk Shear')

# theta/theta_e
theta_ax.set_title(r'$\theta$' + '(red) and ' + r'$\theta_e$' + '(green)')


# AXES LIMITS
#####------------------------------------------------#####
# skew-t
skew.ax.set_adjustable('datalim')
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-30, 45)

# hodograph - incremented grid
h.add_grid(increment = 5, ls = '--', lw = 1, alpha = 0.2)

# vertical helicity
helicity_ax.set_xlim(-100, 1500)
helicity_ax.set_ylim(0,6)

# vertical shear
dvdz_ax.set_xlim(-2e-2, 8e-2)
dvdz_ax.set_ylim(0,6)

# bulk shear
bulk_shear_ax.set_xlim(0, 120)
bulk_shear_ax.set_ylim(0,6)

# theta/theta_e
theta_ax.set_ylim(0,6)
theta_ax.set_xlim(250, 400)

# AXES LABELS
#####------------------------------------------------#####
# skew-t
skew.ax.set_xlabel('Temperature (˚C)')
skew.ax.set_ylabel('Pressure (hPa)')

# vertical helicity
helicity_ax.set_ylabel('height (km)')
helicity_ax.set_xlabel('srh ($m^2s^{-2}$)')

# vertical shear
dvdz_ax.set_xlabel('shear ($s^{-1}$)')

# bulk shear
bulk_shear_ax.set_xlabel('shear (knots)')

# theta/theta_e
theta_ax.set_xlabel('kelvin')

# AXES TICKS
#####------------------------------------------------#####
# hodograph
h.ax.set_yticklabels([])
h.ax.set_xticklabels([])
h.ax.set_xticks([])
h.ax.set_yticks([])
h.ax.set_xlabel(' ')
h.ax.set_ylabel(' ')
plt.xticks(np.arange(0, 0, 1))
plt.yticks(np.arange(0, 0, 1))
for i in range(5, 41, 5):
    h.ax.annotate(str(i), (i, 0), xytext = (0, 2), textcoords = 'offset pixels',
                  clip_on = True, fontsize = 10, alpha = 0.3, zorder = 0)
for i in range(5, 41, 5):
    h.ax.annotate(str(i), (0, -i), xytext = (0, 2), textcoords = 'offset pixels',
                  clip_on = True, fontsize = 10, alpha = 0.3, zorder = 0)

# vertical helicity
helicity_ax.set_xticks(np.arange(0, 1501, 500))
helicity_ax.set_xticklabels(np.arange(0, 1501, 500))

# vertical shear
dvdz_ax.set_xticks(np.arange(-2e-2, 8.1e-2, 2e-2))
dvdz_ax.set_xticklabels(['$-2^{-2}$', '$0$', '$2^{-2}$', '$4^{-2}$', '$6^{-2}$', '$8^{-2}$'])

# bulk shear
bulk_shear_ax.set_xticks(np.arange(0, 121, 30))
bulk_shear_ax.set_xticklabels(np.arange(0, 121, 30))

# theta/theta_e
theta_ax.set_yticks(np.arange(0,7,1))
theta_ax.set_yticklabels(np.arange(0,7,1))
theta_ax.set_xticks(np.arange(250, 401, 50))
theta_ax.set_xticklabels(np.arange(250, 401, 50))

# AXES GRIDS
#####------------------------------------------------#####
# vertical helicity 
helicity_ax.grid(c = 'gray', ls = '--')

# vertical shear
dvdz_ax.grid(c = 'gray', ls = '--')

# bulk shear
bulk_shear_ax.grid(c = 'gray', ls = '--')

# theta/theta_e
theta_ax.grid(c = 'gray', ls = '--')
# OTHER FORMATTING
#####------------------------------------------------#####
# skew-t: alter facecolor to white
fig.set_facecolor('#ffffff')
skew.ax.set_facecolor('#ffffff')

# skew-t: add isotherms
x1 = np.linspace(-100, 40, 8)
x2 = np.linspace(-90, 50, 8)
y = [1100, 50]
for i in range(0, 8):
    skew.shade_area(y = y, x1 = x1[i], x2 = x2[i], color = 'gray', alpha = 0.02, zorder = 1)
    
# skew - t: highlight 0˚ isotherm
skew.ax.axvline(0 * units.degC, linestyle = '--', color = 'blue', alpha = 0.3)

# skew-t: add dry adiabats
skew.plot_dry_adiabats(lw = 1, alpha = 0.3)

# skew-t: add moist adiabats
skew.plot_moist_adiabats(lw = 1, alpha = 0.3)

# skew-t: add mixing ratio lines
skew.plot_mixing_lines(lw = 1, alpha = 0.3)

# hodograph - set aspect ratio    
h.ax.set_box_aspect(1)

# hodograph: set incremented levels/colors
levels = [0, 100, 250, 500, 1000, 3000, 6000]
colors = ['red', 'orange', 'g', 'b', 'magenta', 'k']

######################### END OF FORMATTING AXES #########################

print('Plotting data....\n')
######################### PLOT DATA #########################
# SKEW - T
#####------------------------------------------------#####
# temperature 
skew.plot(prof.pres * units.hPa, prof.tmpc * units.degC, 'r', lw = 1.5, label = 'T')

# dewpoint 
skew.plot(prof.pres * units.hPa, prof.dwpc * units.degC, 'g', lw = 1.5, label = 'Td')

# sfc temp
sfc_t = np.round((prof.tmpc[0] * units.degC).to(units.degF).magnitude, 1)

# sfc dewpoint
sfc_td = np.round((prof.dwpc[0] * units.degC).to(units.degF).magnitude, 1)

# wind barbs
interval = np.logspace(2, 3, 40) * units.hPa
idx = mpcalc.resample_nn_1d(prof.pres, interval)
skew.plot_barbs(pressure = prof.pres[idx], u = prof.u[idx], v = prof.v[idx])

# HODOGRAPH
#####------------------------------------------------#####
# find where we have data above 10km
sfc_10 = np.where(prof.hght <= 10000)[0]

# wind profile
h.plot_colormapped(prof.u[sfc_10], prof.v[sfc_10], c = prof.hght[sfc_10], 
                    intervals = levels, colors = colors, linewidth = 3)

# pull storm motions
rm_u, rm_v, lm_u, lm_v = prof.bunkers

# plot the storm motion 
h.ax.text(rm_u, rm_v, 'RM', weight = 'bold')
h.ax.text(lm_u, lm_v, 'LM', weight = 'bold')

# calculate wind params 
dvdz, bs = wind_params(prof.u, prof.v, [rm_u, rm_v], prof.hght)

vert_h = [winds.helicity(prof, 0, i, rm_u, rm_v)[0] for i in prof.hght]

# VERTICAL HELICITY
#####------------------------------------------------#####
# vertical helicity profile
helicity_ax.plot(vert_h, prof.hght/1000, c = 'r')

# VERTICAL SHEAR
#####------------------------------------------------#####
# vertical shear
dvdz_ax.plot(dvdz, prof.hght/1000, c = 'b')

# BULK SHEAR
#####------------------------------------------------#####
# bulk shear
bulk_shear_ax.plot(bs, prof.hght/1000, c = 'g')

# THETA/THETA_E
#####------------------------------------------------#####
# theta
theta_ax.plot(prof.theta, prof.hght/1000, c = 'r')

# theta_e
theta_ax.plot(prof.thetae, prof.hght/1000, c = 'g')
######################### END OF PLOTTING DATA #########################

# pull eil vars 
eil_pbot = prof.ebottom * units.hPa
eil_ptop = prof.etop * units.hPa
eil_hbot = prof.ebotm
eil_htop = prof.etopm
######################### PLOT EIL (if applicable) #########################
if eil_pbot is not np.ma.masked:
    # fill eil (if applicable)
    helicity_ax.fill_between([-200, 1500], eil_hbot/1000, eil_htop/1000, color = 'dimgray', alpha = 0.25)
    dvdz_ax.fill_between([-2e-2, 8e-2], eil_hbot/1000, eil_htop/1000, color = 'dimgray', alpha = 0.25)
    bulk_shear_ax.fill_between([0, 120], eil_hbot/1000, eil_htop/1000, color = 'dimgray', alpha = 0.25)
    theta_ax.fill_between([250, 400], eil_hbot/1000, eil_htop/1000, color = 'dimgray', alpha = 0.25)
    
    # add eil to skew t
    eil_line = plt.Line2D([0.1], (eil_pbot, eil_ptop), color = 'dimgray', linewidth = 3, transform = skew.ax.get_yaxis_transform())
    eil_line_end1 = plt.Line2D([0.09, 0.11], [eil_pbot], color = 'dimgray', linewidth = 3, transform = skew.ax.get_yaxis_transform())
    eil_line_end2 = plt.Line2D([0.09, 0.11], [eil_ptop], color = 'dimgray', linewidth = 3, transform = skew.ax.get_yaxis_transform())
    
    skew.ax.add_artist(eil_line)
    skew.ax.add_artist(eil_line_end1)
    skew.ax.add_artist(eil_line_end2)
######################### END OF PLOT EIL (if applicable) #########################

######################### LEGENDS #########################
# skew - t
skew.ax.legend(loc='upper left', framealpha = 0.9)

# legend lines/labels
lines = [Line2D([0], [0], color = 'lime', lw = 2),
         Line2D([0], [0], color = 'red', lw = 2),
         Line2D([0], [0], color = 'orange', lw = 2),
         Line2D([0], [0], color = 'green', lw = 2),
         Line2D([0], [0], color = 'blue', lw = 2),
         Line2D([0], [0], color = 'magenta', lw = 2)]

labels = ['0 - 100m', '100 - 250m', '250 - 500m', '500m - 1km', '1 - 3km', '3 - 6km']

# hodograph 
h.ax.legend(lines, labels, loc = 'lower center', ncol = 3)
######################### END OF LEGENDS #########################

######################### ANNOTATIONS #########################
# SKEW - T
#####------------------------------------------------#####
# surface temperature unit
sfc_t_label = f'{int(sfc_t)}°F' 
skew.ax.annotate(sfc_t_label, (prof.tmpc[0] * units.degC, prof.pres[0] * units.hPa), textcoords = "offset points", xytext = (16,-13), fontsize = 12, 
                 color = 'red', weight = 'bold', alpha = 0.7, ha = 'center')

# surface dewpoint unit
sfc_td_label = f'{int(sfc_td)}°F'                       
skew.ax.annotate(sfc_td_label,(prof.dwpc[0] * units.degC, prof.pres[0] * units.hPa), textcoords = "offset points", xytext = (-16,-13), fontsize = 12, 
                 color = 'g', weight = 'bold', alpha = 0.7, ha = 'center') 

# effective inflow layer label
skew.ax.text(0.1, eil_ptop - (5 * units.hPa), 'EIL', transform = skew.ax.get_yaxis_transform(), ha = 'center', 
          color='dimgray', fontsize = 10, weight = 'bold', clip_on = True)

# add height lines (km)
for i in [1, 3, 6, 8, 10, 12]:
    # find where the height is our desired label height
    pres = interp.pres(prof, i * 1000) * units.hPa
        
    # plot the height at the associated pressure level 
    skew.ax.text(0.02, pres, f'-{i}km-', color='k', fontsize = 11, weight = 'bold',
              transform = skew.ax.get_yaxis_transform(), clip_on = True)
    
# plot the surface marker 
skew.ax.text(0.02, prof.pres[0] * units.hPa, '-sfc-', color='k', fontsize = 11, weight = 'bold',
          transform = skew.ax.get_yaxis_transform(), clip_on = True)

# HODOGAPH
#####------------------------------------------------#####
# NOTE: 
    # storm_motion[0]: RM u,v
    # storm_motion[1]: LM u,v
    # storm_motion[2]: 0-6km shear vector u,v

# compute spd/dir of storms 
# right mover speed/dir
rm_deg = np.round((np.arctan2(rm_u, rm_v) * (180/np.pi)) + 180)
rm_spd = np.round(np.sqrt(((rm_u ** 2) + (rm_v ** 2))))

# left mover speed/dir
lm_deg = np.round((np.arctan2(lm_u, lm_v) * (180/np.pi)) + 180)
lm_spd = np.round(np.sqrt((lm_u ** 2) + (lm_v ** 2)))

# storm motions 
h.ax.text(-20, 23, r'$\bf{Storm}$' + ' ' + r'$\bf{Motion}$' + f'\nRM: {int(rm_deg)}˚ @ {int(rm_spd)}kts\nLM: {int(lm_deg)}˚ @ {int(lm_spd)}kts',
          bbox = dict(facecolor = 'white', edgecolor = 'k', alpha = 0.9), ha = 'center')


# eil variables (if applicable)
if eil_htop is np.ma.masked:
    eil_srh = '--'
    srw_eil = '--'
    shr_eil = '--'
    lr_eil = '--'
else:
    eil_srh = prof.esrh[0]
    srw_eil = utils.comp2vec(*prof.right_srw_eff)[1]
    shr_eil = utils.comp2vec(*prof.eff_shear)[1]
    lr_eil = params.lapse_rate(prof, eil_pbot.magnitude, eil_ptop.magnitude)
    
print('Writing parcel info table....\n')
# OTHER
#####------------------------------------------------#####
### start with parcel variables ###
# table rows
sb_title_str = r'$\bf{Surface}$' + ' ' + r'$\bf{Based}$'
ml_title_str = r'$\bf{Mixed}$' + ' ' + r'$\bf{Layer}$'
mu_title_str = r'$\bf{Most}$' + ' ' + r'$\bf{Unstable}$'

m500_title_str= r'$\bf{0-500 m}$'
km1_title_str = r'$\bf{0-1 km}$'
km3_title_str = r'$\bf{0-3 km}$'
km6_title_str = r'$\bf{0-6 km}$'
eil_title_str = r'$\bf{EIL}$'

rows = [sb_title_str, ml_title_str, mu_title_str, '', m500_title_str, km1_title_str, km3_title_str, km6_title_str, eil_title_str]

# table columns
cape_col_str = r'$\bf{CAPE}$'
cin_col_str = r'$\bf{CIN}$'
lcl_col_str = r'$\bf{LCL}$'
lfc_col_str = r'$\bf{LFC}$'

srh_title_str = r'$\bf{SRH}$'
srw_title_str = r'$\bf{SRW}$'
shr_title_str = r'$\bf{Shear}$'
lr_title_str = r'$\bf{\Gamma}$'

cols = [cape_col_str, cin_col_str, lcl_col_str, lfc_col_str]

# table values
# cape/cin
sb_cape_str = f'{int(np.round(prof.sfcpcl.bplus))} ' + r'$Jkg^{-1}$'
sb_cin_str = f'{int(np.round(prof.sfcpcl.bminus))} ' + r'$Jkg^{-1}$'

ml_cape_str = f'{int(np.round(prof.mlpcl.bplus))} ' + r'$Jkg^{-1}$'
ml_cin_str = f'{int(np.round(prof.mlpcl.bminus))} ' + r'$Jkg^{-1}$'

mu_cape_str = f'{int(np.round(prof.mupcl.bplus))} ' + r'$Jkg^{-1}$'
mu_cin_str = f'{int(np.round(prof.mupcl.bminus))} ' + r'$Jkg^{-1}$'

# lcl/el
sb_lcl_str = f'{int(np.round(prof.sfcpcl.lclhght))} ' + r'$m$'
sb_lfc_str = f'--N/A--'

ml_lcl_str = f'{int(np.round(prof.mlpcl.lclhght))} ' + r'$m$'
ml_lfc_str = f'--N/A--'

mu_lcl_str = f'{int(np.round(prof.mupcl.lclhght))} ' + r'$m$'
mu_lfc_str = f'--N/A--'

# wind params
srh_500m_str = f'{int(np.round(winds.helicity(prof, 0, 500, rm_u, rm_v)[0]))} ' + r'$ m^2s^{-2}$'
srh_1km_str = f'{int(np.round(prof.srh1km[0]))} ' + r'$ m^2s^{-2}$'
srh_3km_str = f'{int(np.round(prof.srh3km[0]))} ' + r'$ m^2s^{-2}$'
srh_6km_str = f'{int(np.round(winds.helicity(prof, 0, 6000, rm_u, rm_v)[0]))} ' + r'$ m^2s^{-2}$'
srh_eil_str = f'{int(np.round(eil_srh))} ' + r'$ m^2s^{-2}$'

shr_500m_str = f'{int(np.round(utils.comp2vec(*winds.wind_shear(prof, pbot = prof.pres[0], ptop = prof.pres[np.argmin(abs(prof.hght - 500))]))[1]))} ' + r'$ kts$'
shr_1km_str = f'{int(np.round(utils.comp2vec(*prof.sfc_1km_shear)[1]))} ' + r'$ kts$'
shr_3km_str = f'{int(np.round(utils.comp2vec(*prof.sfc_3km_shear)[1]))} ' + r'$ kts$'
shr_6km_str = f'{int(np.round(utils.comp2vec(*prof.sfc_6km_shear)[1]))} ' + r'$ kts$'
shr_eil_str = f'{int(np.round(shr_eil))} ' + r'$ kts$'

srw_500m_str = f'{int(np.round(utils.comp2vec(*winds.sr_wind(prof, pbot = prof.pres[0], ptop = prof.pres[np.argmin(abs(prof.hght - 500))]))[1]))} ' + r'$ kts$'
srw_1km_str = f'{int(np.round(prof.right_srw_1km[1]))} ' + r'$ kts$'
srw_3km_str = f'{int(np.round(prof.right_srw_3km[1]))} ' + r'$ kts$'
srw_6km_str = f'{int(np.round(prof.right_srw_6km[1]))} ' + r'$ kts$'
srw_eil_str = f'{int(np.round(srw_eil))} ' + r'$ kts$'

# lapse rates
lr_500m_str = f'{np.round(params.lapse_rate(prof, 0, 500, pres = False), 1)} ' + r'$ ˚Ckm^{-1}$'
lr_1km_str = f'{np.round(params.lapse_rate(prof, 0, 1000, pres = False), 1)} ' + r'$ ˚Ckm^{-1}$'
lr_3km_str = f'{np.round(prof.lapserate_3km, 1)} ' + r'$ ˚Ckm^{-1}$'
lr_6km_str = f'{np.round(prof.lapserate_3_6km, 1)} ' + r'$ ˚Ckm^{-1}$' + ' (3-6km)'
lr_eil_str = f'{np.round(lr_eil, 1)} ' + r'$ ˚Ckm^{-1}$'

# create array of table values 
table_data = [[mu_cape_str, mu_cin_str, mu_lcl_str, mu_lfc_str],
              [ml_cape_str, ml_cin_str, ml_lcl_str, ml_lfc_str],
              [sb_cape_str, sb_cin_str, sb_lcl_str, sb_lfc_str],
              [srh_title_str, srw_title_str, shr_title_str, lr_title_str],
              [srh_500m_str, srw_500m_str, shr_500m_str, lr_500m_str],
              [srh_1km_str, srw_1km_str, shr_1km_str, lr_1km_str],
              [srh_3km_str, srw_3km_str, shr_3km_str, lr_3km_str],
              [srh_6km_str, srw_6km_str, shr_6km_str, lr_6km_str],
              [srh_eil_str, srw_eil_str, shr_eil_str, lr_eil_str]]

# plot the table
data_table = parcel_ax.table(cellText = table_data, colLabels = cols, rowLabels = rows, cellLoc = 'center', loc = 'center', edges = 'open')

# scale table size
data_table.scale(0.7, 1.5)

# fetch the closest scan to the time of the data
radar = get_radar_data("KTLX", prof.date.replace(tzinfo = timezone.utc))

# pull radar variables 
ele = str(np.round(radar.elevation["data"][0], 1))
rad_time = dt.datetime.strptime(radar.time['units'][14:], '%Y-%m-%dT%H:%M:%SZ')

print('Radar data has been pulled....plotting.....')

# set extent
min_lat = 35.181 - 1.5
max_lat = 35.181 + 1.5
min_lon = -97.439 - 1.5
max_lon = -97.439 + 1.5

# set transform
transform = ccrs.PlateCarree()

# set projections
projection = ccrs.PlateCarree(central_longitude = -97.439)

# pull radar variables 
ele = str(np.round(radar.elevation["data"][0], 1))
rad_time = dt.datetime.strptime(radar.time['units'][14:], '%Y-%m-%dT%H:%M:%SZ')

# add axis
rad_ax = plt.axes((0.75, 0.6, 0.45, 0.35), projection = projection)

# read in the county data
reader = shpreader.Reader(f'{path}County/countyl010g.shp')
counties = list(reader.geometries())

# read in the highway data
reader = shpreader.Reader(f'{path}tl_2019_us_primaryroads/tl_2019_us_primaryroads.shp')

names = []
geoms = []

for rec in reader.records():
    if (rec.attributes['FULLNAME'][0]=='I'):
        names.append(rec)
        geoms.append(rec.geometry)

# make features to plot
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
highways = cfeature.ShapelyFeature(geoms, ccrs.PlateCarree())

# add cartopy features to rad/ref axis
rad_ax.add_feature(COUNTIES, facecolor = 'none', edgecolor = 'black', linewidth = 0.5)
rad_ax.add_feature(highways, edgecolor = 'darkblue',facecolor = 'none')
rad_ax.add_feature(cfeature.STATES, facecolor = 'none', edgecolor = 'k', linewidth = 2)
rad_ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs = transform)

# plot radar data
display = pyart.graph.RadarMapDisplay(radar)
display.plot_ppi_map('reflectivity', 0, projection = projection, ax = rad_ax, title = f'KTLX {ele}˚ Reflectivity on {rad_time.strftime("%Y%m%d @ %H:%M:%S")}Z', 
                      colorbar_flag = False, add_grid_lines = False, min_lat = min_lat, max_lat = max_lat, min_lon = min_lon, max_lon = max_lon)

# plot point where our sounding is valid
display.plot_point(-97.439, 35.181, markersize = 10, symbol = 'X')

# save fig
plt.savefig(f'{path}hw1_quicklook.png', dpi = 350, bbox_inches = 'tight')

