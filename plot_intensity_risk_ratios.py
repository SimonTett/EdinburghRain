"""
Compute intensity and risk ratios from observed data with scalings from simulations
Makes use of data on fits from plot_edinburgh_geography -- which gives bootstrap estimates for uncertainty in radar distributions.
"""
import pathlib

import commonLib
import gev_r
import xarray
import matplotlib.pyplot as plt
import edinburghRainLib
import scipy.stats
import numpy as np
import statsmodels.api as sm

"""
Very ugly code which larely plots key figure showing chaning intensity and probability ratios as function of time period and rtn period/rainfall
Does the following -- using results produced by:
   analyse_cpm_data_rolling -- data from analysis of cpm data
   comp_radar_fits -- data from analysis of radar data
1) Read in CPM GEV scaling parameters including bootstrap samples of the scalings to apply to radar data
2) Load in the radar data fits and bootstrap fits. 
3) Load radarMax data and find castle radar max for 2020-08 & 2021-07. Add to radar dataset
4) Read in observed CET and generate temperatures for several periods. Estimates CET at +2K plus uncertainty.
5) Computes intensity and probability ratios for various  times relative to PI (1850-1899)
6) Compute monte carlo samples of intensity and probability ratios for time periods. 
7) Compute expected CC scaling based on humidity
8) plot data! (and code for this is fairly ugly)

"""


def scale_fn(cdf_values, params=None, params_ref=None):
    gev = scipy.stats.genextreme(*params)
    gev_ref = scipy.stats.genextreme(*params_ref)
    ratio = gev.ppf(cdf_values) / gev_ref.ppf(cdf_values)
    return ratio


def comp_intense_prob_ratios(radar_fit_all, dScale, rain_values, return_period_values, temperatures,
                             scale_ref='2005-2020', ref_name='PI'):
    """
    Compute intensity and probability ratios for various different time periods using
        parameters from GEV fir to radard regional extremes, scaling from models and temperatures.

    :param radar_fit_all:  Parameters for GEV fits to radar data for all different regional extreme quantiles
    :param dScale: simulated scale data to be used.
    :param rain_values: rain values wanted
    :param return_period_values: return periods wanted
    :param temperatures: temperatures. Dict -- keys are names. Values are temperature differences from present day.
    :param scale_ref: Reference for scaling corresponding to when radar data is for. Is key to temperatures.
    :param ref_name: name of reference for which Prob and intensity ratios are being computed.
    :return: dataset of the intensity, probability ratios relative to PI, scaled radard GEV parameters and scales.
    """
    # setup.
    sf_values = 1.0 / return_period_values
    # need values in order, shape, loc, scale
    param_names = ['shape', 'location', 'scale']
    # Fit the masked observed data to a GEV -- using R
    radar_fit = radar_fit_all.sel(parameter=param_names)
    # compute the rainfall for the  specified sf from the rainfall data
    # generate all distributions at different temperatures by scaling parameters by CPM values
    scale_radar = dict()
    sf = dict()
    isf = dict()
    scales = dict()
    for k in temperatures.keys():
        delta_t = temperatures[k]-temperatures[scale_ref]
        scale = 1+(dScale * delta_t)
        scales[k] = scale  # and store it.
        s = scale.sel(parameter=['Dshape','Dlocation','Dscale']).assign_coords(parameter=['shape','location','scale'])
        radar_fit_scale = s * radar_fit  # compute the scaled distributions.
        sf[k] = gev_r.xarray_sf(rain_values,radar_fit_scale, output_dim_name='Rx15min')  # probability for different rain values
        isf[k] = gev_r.xarray_isf(sf_values,radar_fit_scale)  # rainfall for different p-values
    # now to compute intensity and probability ratios relative to 2005-2020

    ref_sf = sf.pop(ref_name)
    ref_isf = isf.pop(ref_name)
    all_pr = dict()
    all_ir = dict()
    for k in sf.keys():
        all_pr[k] = sf[k] / ref_sf
        all_ir[k] = isf[k] / ref_isf

    all_ir = xarray.Dataset(all_ir)
    all_pr = xarray.Dataset(all_pr)
    scales = xarray.Dataset(scales)
    # convert intensity ratio probabilities to return periods.
    all_ir = all_ir.assign_coords(probability=1.0 / all_ir.probability.values).rename(probability='Return Period')
    return all_ir, all_pr, radar_fit, scales

def pr_ir_monte_carlo(mc_radar_fit,bootstrap_ratio,temperatures,temp_p2k,rain, rtn_period,
                      file_pattern = None,refresh=False, rng=None):
    """
    Monte carlo generate the prob and intensity ratios at specificed rain and return periods
    :param mc_radar_fit: Bootstrap samples (with index sample) of the radar fit parameters
    :param bootstrap_ratio: Bootstrap samples (with index sample) of the GEV scaling parametrs sa a fraction of "today" parameters
    :param temperatures: The temperatures with labels
    :param temp_p2k: Temperatures as a dist for +2K
    :param rain: The rain values at which the prob ratios are to be generated
    :param rtn_period: The return periods at which the prob ratios are to be generated
    :param file_pattern: The file  pattern (as a string). The names for the saved intensity and prob ratios
        are this +"mc_pr.nc" and "mc_ir.nc" respectively. If None then no files will be saved or read from.
        If files exist then data will be read from them rather than being computed.
    :param refresh: If True then data will be generated
    :param rng: random number generator (if not None). If None then np.random.default_rng()   will be used
    :return:intensity and probability ratio samples.
    """

    if file_pattern is not None: # generate file paths
        pr_file = pathlib.Path(file_pattern+"mc_pr.nc")
        ir_file = pathlib.Path(file_pattern+"mc_ir.nc")

    read_data = (not refresh) and (file_pattern is not  None) and ir_file.exists() and pr_file.exists()
    if read_data: # we have data to read  So read it and return it
        mc_all_pr = xarray.load_dataset(pr_file)
        mc_all_ir = xarray.load_dataset(ir_file)
        return mc_all_ir,mc_all_pr

    #  monte-carlo the data
    # setup
    if rng is None:
        rng = np.random.default_rng()  # RNG stuff.
    mc_all_ir = []
    mc_all_pr = []
    temperatures_mc = temperatures.copy()
    n_monte_carlo = bootstrap_ratio.sample.size # how many samples do we have.
    print(f"Bootstrapping {n_monte_carlo} times")
    for indx in range(0, n_monte_carlo):
        # print out a progress marker.
        if ((indx + 1) % 20 == 0):
            print(f"{indx}", end=' ')
        if ((indx + 1) % 200 == 0):
            print(">")

        mc_scale  = bootstrap_ratio.isel(sample=indx) # scaling on the parameters
        mc_radar_params = mc_radar_fit.isel(sample=indx).Parameters # actual paramters
        t = temp_p2k.rvs(random_state=rng) # uncertainty in +2K estimate
        temperatures_mc['PI+2K'] = t
        ir, pr, dist_now_mc, scales_mc = comp_intense_prob_ratios(mc_radar_params, mc_scale, rain, rtn_period,
                                                                  temperatures_mc, scale_ref='2005-2020')

        mc_all_ir.append(ir) # add to the list
        mc_all_pr.append(pr)

    mc_all_ir = xarray.concat(mc_all_ir, dim='sample')# make into a dataset.
    mc_all_pr = xarray.concat(mc_all_pr, dim='sample')
    # save the data
    if file_pattern is not None:
        for v, fname in zip([mc_all_pr, mc_all_ir], [pr_file, ir_file]):
            v.to_netcdf(fname)

    return mc_all_ir,mc_all_pr

## plotting functions


def plot_cc(dT,ax,color,cc_scale,cc_range):
    """
    Plot the CC range
    :param dT: temperature change
    :param ax: axis on which to plot
    :param color: color to use
    :param cc_scale: scaling of CC to temperature
    :param cc_range: lower and upper range of CC scaling

    :return:None

    """
    ax.fill_between(rain,dT*cc_range[0],dT*cc_range[1],alpha=0.2,color=color)
    ax.axhline(dT * cc_scale, color=color, linewidth=3, linestyle='dotted')


def gen_inv_fns(dist_now):
    """
    Generate the functions that relate rain to return period and return period to rain based on current radar distribution.
    :param dist_now: current radar distribution
    :return: rp_to_rain (convert return period to rain) and rain_to_rp (convert rain to return period)
    """
    # inverse functions
    params = dist_now.sel(parameter=['shape', 'location', 'scale'], time_quant=qt).values.tolist()
    gev_now = scipy.stats.genextreme(*params)
    rp_to_rain = lambda x: gev_now.isf(1.0 / x)
    rain_to_rp = lambda x: 1.0 / gev_now.sf(x)

    return rp_to_rain, rain_to_rp

def plot_ir(ax,all_ir,mc_all_ir,plot_styles,dist_now, qt=0.95,q=[0.05,0.95],
            crit_value=None):
    """
    Plot Intensity Ratios & uncertainties therein.

    :param ax: axis on which to plot intensity ratios.
    :param all_ir: All intensity ratios (time periods) as a function of Return Period. Dataset with dataArrays being values for each time period.
    :param mc_all_ir:Monte carlo values. as all_ir with additional axis sample. Used to generate uncertainty estimates.
    :param plot_styles: dictionary of plot styles for each time period. Passed to plotting. color used in uncertainty plot
    :return: secondary axis (of  radar rainfall critical values)
    """
    # add secondary axis showing rainfall for current radar
    rp_to_rain, rain_to_rp = gen_inv_fns(dist_now)
    sec_ax_rain = ax.secondary_xaxis('top', functions=(rp_to_rain, rain_to_rp))
    sec_ax_rain.set_xticks([50, 80, 90, 100, 110, 120, 130, 140, 150, 200])  # rain
    sec_ax_rain.set_xlabel(f'Radar Rx15min (mm/hr)', fontsize='small')

    per = lambda x: (x-1)*100.
    for k, v in plot_styles.items():

        plot_args = v.copy()
        col = plot_args['color']
        ir = per(all_ir[k].sel(time_quant=qt))
        ir.plot(ax=ax, label=k, **plot_args)
        # put the bootstrap empirical uncertainties on...
        mc_all_ir_range = per(mc_all_ir[k].sel(time_quant=qt).quantile(q, 'sample') )
        ax.fill_between(mc_all_ir_range['Return Period'],
                           mc_all_ir_range.isel(quantile=0),
                           mc_all_ir_range.isel(quantile=1),
                           color=col, alpha=0.5)

        if crit_value is not None: #print out values nicely formatted.
            rp = rain_to_rp(crit_value)
            ir = float(all_ir[k].sel(time_quant=qt).interp({'Return Period':rp}))
            irmin=float(mc_all_ir[k].sel(time_quant=qt).quantile(0.05,'sample').interp({'Return Period':rp}))
            irmax = float(mc_all_ir[k].sel(time_quant=qt).quantile(0.95,'sample').interp({'Return Period':rp}))
            print(f"IR {k:12s}: {per(ir):2.0f} ({per(irmin):2.0f}-{per(irmax):2.0f}) " )

    # Axis limits
    ax.set_xlim(5, 150)
    ax.set_ylim(0, 20.)
    ax.yaxis.set_ticks_position('both')

    return sec_ax_rain

def plot_pr(ax_pr,all_pr,mc_all_pr,plot_styles,dist_now, qt=0.95,q=[0.05,0.95],crit_value=None):
    """
    Plot the probability ratios and uncertainties therein.
    :param ax_pr: axis on which to plot the probability ratio
    :param all_pr: all pr values
    :param mc_all_pr: mote carle pr values
    :param plot_styles: plot  styles indexed by name
    :param dist_now: radar distribution for "current" period
    :param q: two element list with min and max quantiles wanted for uncertainty range.
    :param qt: quantile to select for regional extreme.
    :return:secondary axis of return period for current radar data (main axis is rainfall amount)
    """


    for k, plot_Args in plot_styles.items():
        col = plot_Args['color']
        all_pr[k].sel(time_quant=qt).plot(ax=ax_pr, **plot_Args)
        # put the bootstrap empirical uncertainties on...
        mc_all_pr_range = mc_all_pr[k].sel(time_quant=qt).quantile(q, 'sample')
        ax_pr.fill_between(mc_all_pr_range['Rx15min'],
                           mc_all_pr_range.isel(quantile=0),
                           mc_all_pr_range.isel(quantile=1),
                           color=col, alpha=0.5)
        if crit_value is not None:
            pr = float(all_pr[k].sel(time_quant=qt).interp(Rx15min=crit_value))
            prmin = float(mc_all_pr_range.isel(quantile=0).interp(Rx15min=crit_value))
            prmax = float(mc_all_pr_range.isel(quantile=1).interp(Rx15min=crit_value))
            print(f"PR {k:12s}: {pr:3.2f} ({prmin:3.2f}-{prmax:3.2f})")
    # set ticks for axis.
    ax_pr.set_xlim(50, 150)
    ax_pr.set_ylim(1, 2)
    ax_pr.yaxis.set_ticks_position('both')
    # add secondary axis`
    rp_to_rain, rain_to_rp = gen_inv_fns(dist_now) # generate functions needed for secondary axis
    sec_ax_rp = ax_pr.secondary_xaxis('top', functions=(rain_to_rp, rp_to_rain))
    sec_ax_rp.set_xticks([5, 10, 20, 50, 100, 150, 250])
    sec_ax_rp.set_xlabel(f'Radar Return Period (Years)', fontsize='small')

    return sec_ax_rp

def plot_ir_sens(ax,all_ir,dist_now,plot_styles,crit_values,qt=0.95):
    """
    Plot intensity ratios as a function of regional quantile values.
    :param ax: axis on which to plot the intensity  ratios
    :param all_ir: intensity ratio values as a dataset -- each variable being a time period. index is the return period
    :param dist_now: distribution for "current" period
    :param plot_styles: colors and lines styles for each key being plotted.
    :param crit_values: Rainfall values for critical values for all regional extreme quantiles. These converted to return periods and
       used to select the intensity ratios at those return periods.
    :param qt: Plot this regional extreme quantile with a large hexagon rather than a dot.
    :return: Nothing
    """
    tq=crit_values.time_quant # reduce typing!
    # compute the return periods for the regional critical values
    rp = []
    for quant in tq:
        gev = scipy.stats.genextreme(*dist_now.sel(time_quant=quant).values.tolist())
        rp.append(1.0 / gev.sf(crit_values.sel(time_quant=quant)))
    rp = np.array(rp)
    rp = xarray.DataArray(data=rp, coords=dict(time_quant=tq))

    for k,plot_args in plot_styles.items(): # plot each time period
        col = plot_args['color']
        # plot intensity ratios -- selecting the time quant and return period for each time period
        ir = (all_ir[k].sel(time_quant=tq).sel({'Return Period': rp},method='nearest') * 100) - 100
        ir.plot(ax=ax, x='Return Period', **plot_args, linewidth=2, marker='o', ms=6)
        # plot qt value with a large hexagon.
        irr = ir.sel(time_quant=qt)
        ax.plot(float(irr['Return Period']), float(irr), color=col, marker='h', ms=12)
        # if k is "PI+2K" annotate  dots with qt values *100 and rounded to 1 SF
        if k == "PI+2K":
            for time_quant in tq:
                iir = ir.sel(time_quant=float(time_quant))
                x = float(iir['Return Period'])
                y = float(iir)
                ax.annotate(f'{float(time_quant * 100):3.1f}'.replace('.0', ''), (x, y),
                            textcoords='offset points', xytext=(3.5, -7.5), fontsize='small')

    ax.set_xlim(2, 30) #set limits

def plot_pr_sens(ax,all_pr,plot_styles,crit_values,qt=0.95):
    """
    Plot probability ratios as function of regional quantile. Done for each time period.
    :param ax: axis on which to plot
    :param all_pr: all probability ratios as a dataset. Each dataArray contains prob ratios as function of rainfall for a time period.

    :param plot_styles: dict of styles for plotting. Indexed by keys for time periods
    :param crit_values: Regional extreme values we want probability for.
    :param qt: Plot this regional extreme quantile with a large hexagon rather than a dot.
    :return: None
    """
    tq = crit_values.time_quant  # reduce typing!
    for k,plot_args in plot_styles.items(): #iterate over time periods.
        pr = all_pr[k].sel(Rx15min=crit_values.sel(time_quant=tq), method='nearest').sel(time_quant=tq)
        pr.plot(ax=ax, x='Rx15min', **plot_args, linewidth=2, marker='o', ms=6)
        prr = pr.sel(time_quant=qt) #plot the specified regional extreme with a large hexagon rather than a dot
        ax.plot(float(prr.Rx15min), float(prr), color=plot_args['color'], marker='h', ms=12)
        if k == 'PI+2K': # annotate the quantile values
            for time_quant in tq:
                ppr = pr.sel(time_quant=time_quant)
                x = float(ppr.Rx15min)
                y = float(ppr)
                ax.annotate(f'{float(time_quant * 100):3.1f}'.replace('.0', ''), (x, y),
                            textcoords='offset points', xytext=(7.5, 0), fontsize='small')





def plot_result(all_ir,all_pr,mc_all_ir,mc_all_pr,temperatures,cc_scale,cc_rng,mn_ratio_roll,dist_now,radar,plot_styles,num=None,qt=0.95):

    """
    Plot figure showing intensity and probability ratios and some sensitivity cases.
    :param all_ir: all intensity ratios (dataset)
    :param all_pr: all probability ratios (dataset)
    :param mc_all_ir: monte carlo intensity ratios -- index for mc is sample.
    :param mc_all_pr:  monte carlo probability ratios
    :param temperatures: temperatures used in fits. Used for CC lines
    :param cc_scale: Scaling (%/K) for cc ratios.
    :param cc_rng: Lower and upper bounds of cc scaling
    :param mn_ratio_roll: mean parameters as a function of rolling scale  (time-scales to define max rainfall)
    :param dist_now: "current" distribution for radar data.
    :param radar: radar data
    :param plot_styles:plotting styles for different time periods
    :param num: number of figure -- passed directly to plt.subplots
    :param qt: value of quant_time used to define regional extremes.
    :return: The figure created
    """


    fig, (ax, ax_sens) = plt.subplots(nrows=2, ncols=2, num=num, clear=True, figsize=[9, 7],sharey='col')
    cv=float(radar.critical2021.sel(time_quant=qt))
    plot_ir(ax[0],all_ir,mc_all_ir,plot_styles,dist_now,crit_value=cv)
    plot_pr(ax[1],all_pr,mc_all_pr,plot_styles,dist_now,crit_value=cv)
    plot_ir_sens(ax_sens[0],all_ir,dist_now,plot_styles,radar.critical2021.sel(time_quant=slice(0.05,0.95)))
    plot_pr_sens(ax_sens[1],all_pr,plot_styles,radar.critical2021.sel(time_quant=slice(0.05,0.95)))  # plot sens case for probability ratios
    # add on mean values from the different rolling cases
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for rolling, linestyle in zip(mn_ratio_roll['rolling'].values, linestyles):
        all_ir_roll, all_pr_roll, dist_now_roll, scales_roll = comp_intense_prob_ratios(radar_fit.Parameters, mn_ratio_roll.sel(rolling=rolling),
                                                                                        rain, rtn_period,
                                                                                        temperatures, scale_ref='2005-2020')

        for k, v in plot_styles.items():
            if k == '2012-2021': # label the line
                label=f"{rolling}hr"
            else:
                label = None
            plot_Args = v.copy()
            plot_Args.update(linewidth=2,linestyle=linestyle)
            ir = ((all_ir_roll[k].sel(time_quant=qt) * 100) - 100)
            ir.plot(ax=ax[0], label=label,**plot_Args)
            all_pr_roll[k].sel(time_quant=qt).plot(ax=ax[1], **plot_Args)
    # add on the CC for the intensity ratios.
    for k,plot_Args in plot_styles.items():
        # add on CC line for intensity.
        dT = temperatures[k] - temperatures['PI']
        for a in [ax[0],ax_sens[0]]:
            plot_cc(dT, a, plot_Args['color'], cc_scale, cc_rng)

    # plot the regional rainfalls.
    rp_to_rain, rain_to_rp = gen_inv_fns(dist_now) # need the inverse functions to convert from ranfall to return period
    for cv,lines in zip([radar.critical2020,radar.critical2021],['dashed','solid']):
        value =cv.sel(time_quant=0.95)
        #for Intensity Ratio need to know return prd
        for a,v in zip(ax,[rain_to_rp(value),value]):
            a.axvline(v,color='purple',linestyle=lines)


    for v, ls in zip([radar.castle2021, radar.castle2020], ['solid', 'dashed']):
        ax_sens[1].axvline(v, color='purple', linestyle=ls)
    for v in [1, 1.2, 1.4, 1.6, 1.8,2.0]:
        for a in [ax[1], ax_sens[1]]:
            a.axhline(v, color='black', linestyle='dotted')

    # labels and titles.
    label = commonLib.plotLabel()
    axis = ax.tolist()
    axis.extend(ax_sens.tolist())
    for a, title, xtitle, ytitle in zip(axis, ['Intensity Ratio Increase (%) ', 'Probability Ratio'] * 2,
                                        ['Return Period (Years)', f' {radar_time} (mm/hr)'] * 2, ['% increase', 'PR'] * 2):
        a.set_title(title)
        a.set_xlabel(xtitle)
        a.set_ylabel(ytitle)
        label.plot(a)

    fig.subplots_adjust(bottom=0.15)  # make room for the legend
    a = fig.legend(loc='lower center',  ncol=10,fontsize='small')
    a.set_in_layout(False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.125)  # make room for the legend
    fig.show()
    commonLib.saveFig(fig)
    return fig


# now to do the work!

file = edinburghRainLib.dataDir / 'radar_precip/summary_1km_15min.nc'
radar_time = 'Rx15min'
out_dir = edinburghRainLib.dataDir / 'ratios_roll' / (
    "".join(file.stem.split('_')[1:3]))  # where derived results get written
out_dir.mkdir(parents=True, exist_ok=True)

# read in the bootstrapped radar data fits
mc_radar_fit = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'bootstrap_reg_radar_params.nc')
# and the best estimate fit
radar_fit = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'reg_radar_params.nc')
# and the actual radar data -- as quantile in each assumed independant extreme region
radar = xarray.load_dataset(edinburghRainLib.dataDir / 'radar_precip' / 'reg_radar_rain.nc')
##radar_data = radar.radar
# read in monthlyMax radar and find values for Edinburgh castle for 2020 and 2021. Add to radar dataset.
ts = xarray.open_dataset(file).monthlyMax.sel(**edinburghRainLib.edinburgh_castle, method='nearest')
castle2021 = float(ts.sel(time='2021-07'))
castle2020 = float(ts.sel(time='2020-08'))
radar['castle2020']=castle2020
radar['castle2021']=castle2021
rgn = 'CET' # want CET covariate.
## get in the data -- only want the rolling =1 cases = no rolling!

outdir_gev = edinburghRainLib.dataDir / 'gev_fits_roll'
bs_file= outdir_gev/f'{rgn}_bootstrap.nc'
mn_file= outdir_gev/f'{rgn}_mean.nc'
mn_ratio_roll = xarray.load_dataarray(mn_file, decode_times=False)#

bootstrap_ratio = xarray.load_dataarray(bs_file, decode_times=False)#.sel(rolling=1)
#get cases with covariate shape.
bs_file_shape= outdir_gev/f'{rgn}_shape_bootstrap.nc'
mn_file_shape= outdir_gev/f'{rgn}_shape_mean.nc'
mn_ratio_shape_roll = xarray.load_dataarray(mn_file_shape, decode_times=False)
bootstrap_ratio_shape = xarray.load_dataarray(bs_file_shape, decode_times=False)#.sel(rolling=1)

# get in observed CET and compute means for time periods.
obs_cet = commonLib.read_cet()
t_ref = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2005','2020')).mean()) # mn temp for radar data
t_today= float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('2012','2021')).mean())
t_1980s= float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1980','1989')).mean())
t_1961_1990 = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1961', '1990')).mean())
t_pi = float(obs_cet.sel(time=(obs_cet.time.dt.month == 7)).sel(time=slice('1850', '1899')).mean())
temp_p2k = scipy.stats.norm(loc=2 * 0.94 + t_pi, scale=2 * 0.03)
# how much more summer-time CET is at +2K warming Values provided by Prof. Ed Hawkins (Reading)

temperatures = {'PI': t_pi, '2012-2021': t_today, '2005-2020':t_ref,'1961-1990': t_1961_1990, '1980s':t_1980s, 'PI+2K': temp_p2k.mean()}
npt=200
rain = np.linspace(5, 200, npt)
rtn_period = np.geomspace(5, 200, npt)
all_ir, all_pr, dist_now, scales = comp_intense_prob_ratios(radar_fit.Parameters, mn_ratio_roll.sel(rolling=1), rain, rtn_period,
                                                            temperatures, scale_ref='2005-2020')
dist_now = xarray.Dataset(dict(Parameters=dist_now))
all_ir_shape, all_pr_shape, dist_now_shape, scales_shape = comp_intense_prob_ratios(radar_fit.Parameters, mn_ratio_shape_roll.sel(rolling=1), rain, rtn_period,
                                                                                    temperatures, scale_ref='2005-2020')
dist_now_shape= xarray.Dataset(dict(Parameters=dist_now_shape))



## do the bootstrap calculations
refresh=False # set True to  refresh the bootstrap calculations. If files do not exist then bootstrap will be done.
mc_all_ir,mc_all_pr = pr_ir_monte_carlo(mc_radar_fit,bootstrap_ratio.sel(rolling=1),temperatures,temp_p2k,rain, rtn_period,
                      file_pattern = str(outdir_gev)+"/",refresh=refresh)
mc_shape_all_ir,mc_shape_all_pr = pr_ir_monte_carlo(mc_radar_fit,bootstrap_ratio_shape.sel(rolling=1),temperatures,temp_p2k,rain, rtn_period,
                      file_pattern = str(outdir_gev)+"/shape_",refresh=refresh)
# get the expected CC rate which arises because Edinburgh Temperatures increase at about 90% of CET rates.
# might need to re-compute here rather than in analyse_cpm code.
qsat_fit = sm.load(outdir_gev/'cet_qsat_fit.sav')
f=qsat_fit.get_prediction([1,t_today])
cc_scale = float(100*(qsat_fit.params[1]/f.predicted_mean))
cc_rng = 100*qsat_fit.conf_int()[1,:]/f.predicted_mean

# make plots
qt = 0.95 # regional extreme quantile

plot_styles = { # styles for plotting
    '1980s':dict(color='royalblue',linestyle='solid'),
    '2012-2021': dict(color='blue', linestyle='solid'),
    'PI+2K': dict(color='red', linestyle='solid')}

# plot results for std case
figResult=plot_result(all_ir,all_pr,mc_all_ir,mc_all_pr,temperatures,cc_scale,cc_rng,mn_ratio_roll,
                      dist_now['Parameters'],radar,plot_styles,num='prob_radar_cpm',qt=qt)
# and sens cases -- shape changing with temperature.
figShape=plot_result(all_ir_shape,all_pr_shape,mc_shape_all_ir,mc_shape_all_pr,temperatures,cc_scale,cc_rng,mn_ratio_shape_roll,
                     dist_now_shape['Parameters'],radar,
                     plot_styles,num='prob_radar_cpm_shape',qt=qt)
