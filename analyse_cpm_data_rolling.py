"""
Fit covariate dependent GEV to extremes from CPM and saves them.  Data should contain time/rolling period/ensemble_id, grid_lat & grid_long
Then compute bootstrap uncertainties on average fits when hts between 0 and 200 m.

Data is saved and subsequently used by plot_intensity_risk_ratios.py

Then produce figure for SI.
Will show:
1) Scale and location plots around Edinburgh
2) Scatter plot coloured by ht of Dscale & Dlocation
3) QQ-plots for some samples. Ideal would be one plot but can't do that!
This code will generate the fits if data does not exist or recreate_fit is True. There are used
 to compute changes in risk and intensity.
"""
import pathlib

import commonLib
import edinburghRainLib
import xarray
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors
import patch_mpl_mosaic # patch plt.subplot_mosaic.
import gev_r # needs to be imported before any r done as sets up r-environment
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as rpandas2ri

refresh = False  # if true create  fit data & bootstrap samples *even* if files exist



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    From https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def qsat(temperature):
    """
    Saturated humidity from temperature.
    :param temperature: temperature (in degrees c)
    :return: saturated humidity
    """
    es = 6.112 * np.exp(17.6 * temperature / (temperature + 243.5))
    return es


# where all the processed data goes.
outdir_gev = edinburghRainLib.dataDir / 'gev_fits_roll'
outdir_gev.mkdir(parents=True, exist_ok=True)  # create the directory

## read in the data we need
covariate_dir = edinburghRainLib.dataDir / "covariates"
cet = xarray.load_dataset(covariate_dir/ 'cet_tas.nc').tas
cpm = xarray.load_dataset(covariate_dir/ 'cpm_reg_tas.nc').tas

ed_temp = xarray.load_dataset(covariate_dir / 'ed_reg_tas.nc').tas
ed_precip = xarray.load_dataset(covariate_dir / 'ed_reg_pr.nc').pr
ed_reg_hum=qsat(ed_temp)
cpm_files=list((edinburghRainLib.dataDir/'CPM_data').glob("**/*.nc"))
ed_extreme_precip = xarray.open_mfdataset(cpm_files).seasonalMax
# now extract roughly 100x100 km box centred on Edinburgh and only for summer
ed_rot = edinburghRainLib.rotated_coords['Edinburgh']
ed_rgn_rot=dict()
for c,name in zip(ed_rot,['grid_longitude','grid_latitude']):
    ed_rgn_rot[name]= slice(c-0.5,c+0.5)

ed_extreme_precip = ed_extreme_precip.sel(time=(ed_extreme_precip.time.dt.season=='JJA'))
ed_extreme_precip= ed_extreme_precip.sel(**ed_rgn_rot).load()


# get in the topographic info for the CPM. Note coords differ slightly from ed_extreme prob due to FP changes.
cpm_ht = xarray.load_dataset(edinburghRainLib.dataDir / 'orog_land-cpm_BI_2.2km.nc', decode_times=False).squeeze()
cpm_ht = cpm_ht.sel(
    longitude=slice(ed_extreme_precip.grid_longitude.min() - 0.01, ed_extreme_precip.grid_longitude.max() + 0.01),
    latitude=slice(ed_extreme_precip.grid_latitude.min() - 0.01, ed_extreme_precip.grid_latitude.max() + 0.01))
cpm_ht = cpm_ht.rename(longitude='grid_longitude', latitude='grid_latitude')
# need to fix the longitudes & latitudes...
cpm_ht = cpm_ht.assign(grid_longitude=ed_extreme_precip.grid_longitude,
                       grid_latitude=ed_extreme_precip.grid_latitude).ht
msk = (cpm_ht > 0) & (cpm_ht < 200)

# get in observed seasonal CET
obs_cet = commonLib.read_cet()
t_today = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2012', '2021')).mean())
#t_today = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time='2020'))
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())
temp_p2k = 2 * 0.94 + t_pi # from Ed

## Do the fit
stack_dims = ['time', 'ensemble_member']
file = outdir_gev / f"constant_fit.nc"
fit_nocov = gev_r.xarray_gev(ed_extreme_precip.stack(time_ensemble=stack_dims),
                                   dim='time_ensemble', file=file, recreate_fit=refresh, verbose=True)
xfit = dict()
xfit_shape = dict()
covariates = dict()
for ts, title in zip([cet, cpm, ed_temp, ed_reg_hum], ['CET', 'CPM_region', 'Edinburgh_region', 'Ed_Humidity']):
    file = outdir_gev / f"cov_{title}.nc"
    ts_summer = ts.resample(time='QS-DEC').mean().dropna('time')
    ts_summer = ts_summer.sel(time=(ts_summer.time.dt.month == 6))
    ts_summer['time'] = ed_extreme_precip.time  # make sure times are the same
    covariates[title] = ts_summer
    xfit[title] = gev_r.xarray_gev(ed_extreme_precip.stack(time_ensemble=stack_dims),
                                   cov=ts_summer.stack(time_ensemble=stack_dims),
                                   dim='time_ensemble', file=file, recreate_fit=refresh, verbose=True)
    file = outdir_gev / f"cov_{title}_shape.nc"
    xfit_shape[title] = gev_r.xarray_gev(ed_extreme_precip.stack(time_ensemble=stack_dims),
                                         cov=ts_summer.stack(time_ensemble=stack_dims), shape_cov=True,
                                         dim='time_ensemble', file=file, recreate_fit=refresh, verbose=True)

#
# want ratios of Dlocation/location & Dscale/scale

def comp_bootstrap(ratio_Dparam,cpm_ht,direct=None,filename_start=None,refresh=False,
                   nboot=1000,fract_sample=1e-2):
    """
    Compute mean and bootstrap samples for ratios.
    :param ratio_Dparam:
    :param fract_sample: how many samples to take each bs -- crude hack for auto-correlation
    :param nboot: how many bootstrap samples to take
    :return:
    """
    if direct is None:
        direct= pathlib.Path.cwd()
    if filename_start is None:
        filename_start="Dparam"
    bs_file = direct/f"{filename_start}_bootstrap.nc"
    mn_file = direct/f"{filename_start}_mean.nc"
    if refresh or (not bs_file.exists()):
        print("Making mean and boostrap of params")
        L = (cpm_ht > 0) & (cpm_ht < 200)
        ratio_Dparam_flatten = ratio_Dparam.where(L).stack(horizontal=edinburghRainLib.cpm_horizontal_coords)
        ratio_Dparam_flatten = ratio_Dparam_flatten.where(np.isfinite(ratio_Dparam_flatten), drop=True)
        nCPM = ratio_Dparam_flatten.sel(parameter='Dlocation').sel(rolling=1).size
        mn_ratio = ratio_Dparam_flatten.mean('horizontal')
        bootstrap_ratio = []
        rng = np.random.default_rng()
        for indx in range(0, nboot):
            if ((indx + 1) % 20 == 0):
                print(f"{indx}", end=' ')
            if ((indx + 1) % 200 == 0):
                print(">")

            CPM_sample = rng.integers(0, nCPM, int(nCPM * fract_sample))
            bs_Dparam = ratio_Dparam_flatten.isel(horizontal=CPM_sample).mean('horizontal')
            bootstrap_ratio.append(bs_Dparam)
        bootstrap_ratio = xarray.concat(bootstrap_ratio, dim='sample')
        # save the data
        mn_ratio.to_netcdf(mn_file)
        bootstrap_ratio.to_netcdf(bs_file)
    else:
        mn_ratio = xarray.load_dataarray(mn_file, decode_times=False)
        bootstrap_ratio = xarray.load_dataarray(bs_file, decode_times=False)
    return mn_ratio,bootstrap_ratio
rgn='CET'
fit_var = xfit[rgn]
fit_var_shape  = xfit_shape[rgn]


D=fit_var.Parameters.sel(parameter=['Dlocation','Dscale','Dshape'])#.where(msk)
params_today = gev_r.param_at_cov(fit_var.Parameters,t_today)
ratio_Dparam = D/params_today.sel(parameter=['location','scale','shape']).assign_coords(parameter=['Dlocation','Dscale','Dshape'])
mn_ratio, bootstrap_ratio = comp_bootstrap(ratio_Dparam,cpm_ht,direct=outdir_gev,filename_start=rgn,refresh=refresh)

D_shape=fit_var_shape.Parameters.sel(parameter=['Dlocation','Dscale','Dshape'])#.where(msk)
params_today_shape = gev_r.param_at_cov(fit_var_shape.Parameters,t_today)
ratio_Dparam_shape = D_shape/params_today_shape.sel(parameter=['location','scale','shape']).assign_coords(parameter=['Dlocation','Dscale','Dshape'])
mn_ratio_shape, bootstrap_ratio_shape = comp_bootstrap(ratio_Dparam_shape,cpm_ht,direct=outdir_gev,filename_start=f"{rgn}_shape",refresh=refresh)



## generate scatter diagram of summer CET  vs summer edinburgh humidity
# also generate the sat humidity with CET relationship.
import statsmodels.api as sm
def comp_summer(ts, dim='time'):
    """
    Compute summer average form monthly-mean data
    :param ts: timeseries to compute summer means from
    :param dim: dimension to be resampled over
    :return: summer mean values
    """

    ts_seas = ts.resample({dim:'QS-DEC'}).mean().dropna(dim)
    L = (ts_seas[dim].dt.season == 'JJA')
    summer = ts_seas.sel({dim:L})
    return summer
ed_qsat=qsat(ed_temp)
ed_temp_summer  = comp_summer(ed_temp)
ed_precip_summer = comp_summer(ed_precip)
cet_summer = comp_summer(cet)
obs_cet_summer = comp_summer(obs_cet.sel(time=slice('1850',None)))
ed_qsat_summer = comp_summer(ed_qsat)
median_max_Rx1hr=ed_extreme_precip.where(msk,np.nan).median(edinburghRainLib.cpm_horizontal_coords)
fig_scatter,axes=plt.subplots(nrows=2,ncols=2,num='cet_scatter',figsize=[8,5],clear=True)

def plot_regress(x,y,ax):
    X = x.values.flatten()
    X = sm.add_constant(X)
    Y = y.values.flatten()
    ens_m = y.ensemble_member.broadcast_like(y).values.flatten()
    model = sm.OLS(Y, X)
    fit = model.fit()

    ax.scatter(X[:,1],Y,s=10,marker='o',c=ens_m,cmap='tab20')
    indx = [np.argmin(X[:,1]),np.argmax(X[:,1])]
    ax.plot(X[indx, 1], fit.fittedvalues[indx], color='black')
    return fit

label=commonLib.plotLabel()
linear_fit=dict()
min_obs=obs_cet_summer.min()
max_obs=obs_cet_summer.max()
median_obs = obs_cet_summer.median()
xerr=np.array([[median_obs-min_obs,max_obs -median_obs]]).T
for ax,y,name in zip(axes.flatten(),[ed_qsat_summer,ed_temp_summer,ed_precip_summer,median_max_Rx1hr.sel(rolling=1)],
                     ['qsat','Ed_Temp','Ed_Precip','Rx1hr']):
    linear_fit[name] = plot_regress(cet_summer,y,ax)

    label.plot(ax)
    ax.set_title(f'CET vs {name} $R^2$:{linear_fit[name].rsquared_adj*100:2.0f} %')
    ax.set_xlabel("CET (C)")
    ax.set_ylabel(name)
    for t in [t_pi,t_today,temp_p2k]:
        ax.axvline(t,color='black',linestyle='dotted')
    # plot range of CET
    today_v = float(linear_fit[name].get_prediction([1, t_today]).predicted_mean)
    if name != 'Ed_Temp':
        text=f"{100*float(linear_fit[name].params[1]/today_v):3.1f} $\pm$ {100*float(linear_fit[name].HC3_se[1]/today_v):4.2f} %"

    else:
        text=f"{float(linear_fit[name].params[1]):3.2f} $\pm$ {float(linear_fit[name].HC3_se[1]):4.3f}"

    print(name, text)
    ax.text(0.1,0.85,text,transform=ax.transAxes,bbox=dict(boxstyle="square",fc='lightgrey'))
    for t in [min_obs,max_obs]:
        ax.axvline(t,color='red',linestyle='dotted')
    #ax.errorbar(median_obs,today_v,xerr=xerr,linewidth=2,color='red',capsize=3,capthick=2)
    #ax.scatter(temp_p2k,today_v,color='red',marker='o',s=20)

fig_scatter.tight_layout()
fig_scatter.show()
commonLib.saveFig(fig_scatter)


# use the fit information to compute the expected change in extreme rainfall
f=linear_fit['qsat'].get_prediction([1,t_today])
cc_scale = 100*(linear_fit['qsat'].params[1]/f.predicted_mean)

# and the decline in precipit.
f=linear_fit['Ed_Precip'].get_prediction([1,t_today])
rain_change = 100*float(linear_fit['Ed_Precip'].params[1]/f.predicted_mean)
# save the qsat fit
linear_fit['qsat'].save(outdir_gev/'cet_qsat_fit.sav')
## and make plot for SI
projRot=ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
plot_kws=dict(projection=projRot)
rgn= []
for k,v in edinburghRainLib.edinburgh_region.items():
    rgn.extend([v.start,v.stop])

kw_cbar=dict(orientation='horizontal',label='',fraction=0.1,pad=0.15,extend='both')

label=commonLib.plotLabel()
fig,ax_dict=plt.subplot_mosaic([['location','scale'],
                                ['location','scale'],
                                ['scatter','scatter_shape'],
                                ['r_gev_fit_ed','r_gev_fit_w_ed']],
                               #gridspec_kw=dict(height_ratios=[2,1,1]),
                               num='cpm_gev_fit',clear=True,figsize=[9,10],
                               subplot_kw_mosaic=dict(location=plot_kws,scale=plot_kws))
for p in ['location','scale']:
    ax= ax_dict[p]
    ax.set_extent(rgn,crs=ccrs.OSGB())
    params_today.sel(parameter=p,rolling=1).plot(ax=ax,robust=True,cbar_kwargs=kw_cbar,cmap='Blues')
    c=cpm_ht.plot.contour(levels=[200,400],colors=['black'],linewidths=2,linestyles=['solid','dashed'],ax=ax)
    ax.set_title(p.capitalize())
    edinburghRainLib.std_decorators(ax)
    ax.plot(*edinburghRainLib.edinburgh_castle.values(), transform=ccrs.OSGB(),
            marker='o', color='purple', ms=9, alpha=0.7)
    label.plot(ax)
# scatter plot



def scatter_dloc_scale(ax,ratio_Dparam,ratio_mean,ratio_bootstrap,cpm_ht,topog_levels):
    """
    Plot scatter plot of dLocation vs dScale
    :param ax: axis
    :param ratio_mean: Mean ratios
    :param ratio_bootstrap: Bootstraped mean ratios.
    :return: colour map
    """
    norm = matplotlib.colors.BoundaryNorm(topog_levels, ncolors=255, extend='both')  # as topography plots.
    s= np.where(cpm_ht < 0.01, 30,15)
    s = np.where(cpm_ht > 200,30,s)
    mn_ratio = ratio_mean * 100
    bootstrap_ratio = ratio_bootstrap*100
    rr=100*ratio_Dparam.sel(rolling=1)
    rolling=1
    cm=ax.scatter(rr.sel(parameter='Dlocation'),rr.sel(parameter='Dscale'),
                  c=cpm_ht,cmap='terrain',s=s,norm=norm,marker='.') # plot all points
    ax.scatter(mn_ratio.sel(parameter='Dlocation',rolling=rolling),mn_ratio.sel(parameter='Dscale',rolling=rolling),s=60,marker='o',color='red',label=f'Rx{rolling}hr')
    e=confidence_ellipse(bootstrap_ratio.sel(parameter='Dlocation',rolling=rolling).values,
                         bootstrap_ratio.sel(parameter='Dscale',rolling=rolling).values,ax,n_std=2,edgecolor='red',linewidth=4)
    # Only keep hts where between 0 and 200.
    L = (cpm_ht > 0) & (cpm_ht < 200)
    rrm=rr.where(L,np.nan)
    rrm=rrm.stack(horizontal=edinburghRainLib.cpm_horizontal_coords)
    rrm=rrm.where(np.isfinite(rrm),drop=True)
    e2=confidence_ellipse(rrm.sel(parameter='Dlocation'),rrm.sel(parameter='Dscale'),ax,n_std=2,
                         edgecolor='black',linewidth=4)

    for rolling, color in zip(mn_ratio['rolling'].values[1:], ['brown', 'purple', 'blue']):
        ax.scatter(mn_ratio.sel(parameter='Dlocation', rolling=rolling),
                   mn_ratio.sel(parameter='Dscale', rolling=rolling),
                   s=40, marker='o', color=color, label=f'Rx{rolling}hr')
        e = confidence_ellipse(bootstrap_ratio.sel(parameter='Dlocation', rolling=rolling).values,
                               bootstrap_ratio.sel(parameter='Dscale', rolling=rolling).values, ax, n_std=2,
                               edgecolor=color, linewidth=4)

    return cm

ax_scatters=[]
topog_levels=[0.01,100,200,300,400,500]
cm = scatter_dloc_scale(ax_dict['scatter'], ratio_Dparam, mn_ratio, bootstrap_ratio, cpm_ht, topog_levels)
cm = scatter_dloc_scale(ax_dict['scatter_shape'], ratio_Dparam_shape, mn_ratio_shape, bootstrap_ratio_shape, cpm_ht, topog_levels)
for name,title in zip(['scatter','scatter_shape'],['Fixed shape','Covariate shape']):
    ax=ax_dict[name]

    ax.axhline(cc_scale,color='black',linestyle='dashed')
    ax.axvline(cc_scale,color='black',linestyle='dashed')
    ax.set_title(title)

    ax.set_xlabel("% d location/dT")
    ax.set_ylabel("% d scale/dT")
    label.plot(ax)
    ax_scatters.append(ax)


fig.colorbar(cm,ax=ax_scatters,orientation='horizontal',label='',fraction=0.075,extend='both',ticks=topog_levels,pad=0.1)


#ax.legend()



# plot qq-plots for covariate fits
## plot the fits for CPM data
cet = xarray.load_dataset(covariate_dir / 'cet_tas.nc').tas
ed=edinburghRainLib.rotated_coords['Edinburgh']
ed_ext=ed_extreme_precip.sel(grid_latitude=ed[1],grid_longitude=ed[0],method='nearest',rolling=1).load() #
ed_west_ext = ed_extreme_precip.sel(grid_latitude=ed[1],grid_longitude=ed[0]-0.25,method='nearest',rolling=1).load() #
ed_north_ext=ed_extreme_precip.sel(grid_latitude=ed[1]+0.25,grid_longitude=ed[0],method='nearest',rolling=1).load() #
ts_summer = cet.resample(time='QS-DEC').mean().dropna('time')
ts_summer = ts_summer.sel(time=(ts_summer.time.dt.month == 6)) # get the summer values
ts_summer['time'] = ed_extreme_precip.time  # make sure times are the same

for data,name in zip( [ed_ext,ed_west_ext],['r_gev_fit_ed','r_gev_fit_w_ed']):
    ax=ax_dict[name]
    df_data = [data.values.flatten()]
    df_data.append(ts_summer.values.flatten())
    cols=['x','cov']
    df=pd.DataFrame(np.array(df_data).T,columns=cols)
    print("Created df")
    with rpy2.robjects.conversion.localconverter(robjects.default_converter + rpandas2ri.converter):
        robjects.globalenv['df'] = df  # push the dataframe with info into R
    print("Pushed df to R and ready to fit")
    r_code = 'result<-fevd(x=x,data=df,location.fun=~cov,scale.fun=~cov)'
    fit = robjects.r(r_code)  # do the fit
    print("Fit data")
    # and plot it. Can we do better and get the qq plot as parameters (want data and model appropriately scaled)
    robjects.r('windows(width=4,height=3)')
    robjects.r(f'plot(result,type="qq")')
    # and save it
    path = rf'figures/rplot_qq_{name}.png'
    robjects.r(f'savePlot(filename="{str(path)}",type="png")')
    print("Saved ",str(path))
    img = plt.imread(path) # read image back in.
    ax.imshow(img)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_title(name)
    label.plot(ax) # add label

#fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

# print out ratios relative to CC
print(mn_ratio.drop_vars(['t','surface'])/float(cc_scale))
## and print out mean AIC values
L = (cpm_ht > 0) & (cpm_ht < 200)
with np.printoptions(precision=0):

    print("All pts")
    print("No varn             ", fit_nocov.AIC.mean(edinburghRainLib.cpm_horizontal_coords).values)
    for k in xfit.keys():
        print("d locn & scale       ", k,xfit[k].AIC.mean(edinburghRainLib.cpm_horizontal_coords).values)
        print("d locn, scale& shape ", k, xfit_shape[k].AIC.mean(edinburghRainLib.cpm_horizontal_coords).values)
    print("Low pts")
    print("No varn             ", fit_nocov.AIC.where(L).mean(edinburghRainLib.cpm_horizontal_coords).values)
    for k in xfit.keys():
        print("d locn & scale       ", k,xfit[k].AIC.where(L).mean(edinburghRainLib.cpm_horizontal_coords).values)
        print("d locn, scale& shape ", k, xfit_shape[k].where(L).AIC.mean(edinburghRainLib.cpm_horizontal_coords).values)





