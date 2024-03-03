"""
Analyse the radar data
"""
import commonLib
import edinburghRainLib
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from R_filter import gev_r

recreate_fit  =False # if true create radar fit data *even* if file exists
file = edinburghRainLib.dataDir/'radar_precip/summary_1km_15min.nc'

radar_fit_dir = edinburghRainLib.dataDir/'radar_fits'
radar_fit_dir.mkdir(exist_ok=True) # create data for radar data
radar_fit_file = radar_fit_dir/(file.stem+"_fit.nc")

## get in the topography

ed_rgn = {k:slice(v-50e3,v+50e3) for k,v in edinburghRainLib.edinburgh_castle.items()}
radar_precip=xarray.open_dataset(file).sel(**ed_rgn)
radar_time = 'Rx15min'
if file.name.endswith('15Min.nc'): # 15min data.
    radar_precip.monthlyMean /= 4. # FIXME. Fix mean in Jasmin computation.
    radar_time= 'Rx15min'
if '5km' in file.name:
    topog_grid=55
else:
    topog_grid=11

topogSS = edinburghRainLib.read_90m_topog(ed_rgn,resample=topog_grid)
top_fit_grid = topogSS.interp_like(radar_precip.monthlyMax.isel(time=0).squeeze())
msk=(top_fit_grid > 0 ) & (top_fit_grid < 200)

summer = radar_precip.time.dt.month.isin([6,7,8]) # summer.
mean = radar_precip.monthlyMean.sel(time=summer).load()
mx_seas_rain = radar_precip.monthlyMax.sel(time=summer).load()
mx_seas_rain =mx_seas_rain.resample(time='QS-Jun').max().dropna('time') # max rain in a season
cet = commonLib.read_cet()
cet = cet.sel(time=(cet.time.dt.month==7)).sel(time=slice('2005','2021'))

## plot the data for QC  purposes
cbar_kwrds=dict(orientation='horizontal',fraction=0.1,pad=0.1)
projGB = ccrs.OSGB()
fig,axes = plt.subplots(nrows=1,ncols=2,num='Radar_QC',clear=True,subplot_kw=dict(projection=projGB),figsize=[11,8])
mean.median('time').plot(robust=True,ax=axes[0],cbar_kwargs=cbar_kwrds,levels=[1,1.5,2,2.5,3,3.5,4,4.5,5,10,20])
mx_seas_rain.median('time').plot(robust=True,ax=axes[1],cbar_kwargs=cbar_kwrds)
rgn = []
for v in edinburghRainLib.edinburgh_castle.values():
    rgn.extend([v-50e3,v+50e3])
for ax,title in zip(axes,['Monthly Mean Summer Rainfall (mm/day)',f'Summer Mean Monthly {radar_time}']):
    ax.set_extent(rgn,crs=projGB)
    edinburghRainLib.std_decorators(ax,showregions=True)
    c=top_fit_grid.plot.contour(ax=ax,levels=[200],colors='black',transform=ccrs.PlateCarree())
    ax.set_title(title)

    #ax.clabel(c, c.levels, inline=True, fontsize=10)

plt.show()
## do fit
cet['time']=mx_seas_rain.time
fit = gev_r.xarray_gev(mx_seas_rain, dim='time', file=radar_fit_file, recreate_fit=recreate_fit, verbose=True, name=If)
today = gev_r.param_cov(fit, cet.mean())

##and plot it
fig_params,axes = plt.subplots(nrows=2,ncols=2,num='Radar_params',clear=True,subplot_kw=dict(projection=projGB),figsize=[11,9])
for ax,title in zip(axes[0].flatten(),['location','scale']):
    ax.set_extent(rgn,crs=projGB)
    today.sel(parameter=title).plot(robust=True,cbar_kwargs=cbar_kwrds,ax=ax)
for ax,title in zip(axes[1],['AIC','nll']):
    ax.set_extent(rgn,crs=projGB)
    fit[title].plot(robust=True,cbar_kwargs=cbar_kwrds,ax=ax)
for ax, title in zip(axes.flatten(), ['location', 'scale','AIC','nll']): #std stuff
    edinburghRainLib.std_decorators(ax,showregions=True)
    c=top_fit_grid.plot.contour(ax=ax,levels=[200],colors='black',transform=ccrs.PlateCarree())
    ax.set_title(title)

fig_params.tight_layout()
fig_params.show()


## and plot it
rgn = []
rgn_dict={}
for k,v in edinburghRainLib.edinburgh_castle.items():
    val=[v-50e3,v+50e3]
    rgn.extend(val)
    rgn_dict[k]=slice(val[0],val[1])

topog_lev = [-75,-50, -25, 0, 25, 50, 75, 100, 150, 200, 300, 400, 500]
#Cite as: Pope, Addy. (2017). GB SRTM Digital Elevation Model (DEM) 90m, [Dataset]. EDINA. https://doi.org/10.7488/ds/1928.
fig, ax = plt.subplots(subplot_kw=dict(projection=projGB),clear=True,num='topog')
ax.set_extent(rgn,crs=projGB)
cm=topog.sel(**rgn_dict).plot(ax=ax,robust=True,cmap='terrain',levels=topog_lev,add_colorbar=False)
edinburghRainLib.std_decorators(ax)
fig.colorbar(cm,ax=ax,orientation='horizontal',
             label='Height/Baythmetry from ASL (m)',fraction=0.1,pad=0.1,aspect=30,ticks=topog_lev,spacing='proportional')
fig.show()







