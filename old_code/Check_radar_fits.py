"""
Plot Q-Q plots for radar data.
No longer used
"""
import xarray
import scipy.stats
import matplotlib.pyplot as plt
import commonLib
import edinburghRainLib
import statsmodels.graphics.gofplots
import numpy as np
import pandas as pd
ed_rgn = edinburghRainLib.edinburgh_region
#import gev_r
dist=scipy.stats.genextreme
rgn= []
for k,v in ed_rgn.items():
    rgn.extend([v.start,v.stop])
radar = edinburghRainLib.gen_radar_data()
radar_data = radar.radar
#radar_data, edRain, rainCount,rain_indx =



## plot
plt.close('all')
fig,axes= plt.subplots(nrows=2,ncols=3,num='dists',clear=True)
dists=[scipy.stats.genextreme,scipy.stats.weibull_min,scipy.stats.gumbel_r,scipy.stats.powerlognorm,scipy.stats.genpareto]
dd=dict()
ks=dict()
rp=dict()
for dist,ax in zip(dists,axes.flatten()):
    name = dist.name
    dd[name]=[]
    ks[name]=[]
    rp[name]=[]
    for q in radar_data.time_quant:
        rr=radar_data.sel(time_quant=q)
        if q == 0.95:
            statsmodels.graphics.gofplots.qqplot(rr,dist=dist,fit=True,ax=ax,line='45',ms=3)
        fit = dist.fit(rr)
        d=dist(*fit)
        dd[name].append(d)
        ks[name].append(scipy.stats.kstest(rr,d.cdf))
        rn=float(edRain.sel(time_quant=q))
        rp[name].append(1.0/d.sf(rn))
        #print(name,float(q),ks[name],rn,1.0/d.sf(rn))
    ax.set_title(name)
    dd[name]=pd.Series(dd[name],index=radar_data.time_quant)
    ks[name]=pd.DataFrame(ks[name],index=radar_data.time_quant)
    rp[name]=pd.Series(rp[name],index=radar_data.time_quant)

    #ax.set_xlim(-2,25)
    #ax.set_ylim(-2,25)

fig.tight_layout()
fig.show()
commonLib.saveFig(fig)

## probably should do something for the cpm data...
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri as rpandas2ri
import pandas as pd

def qsat(temperature):
    """
    Saturated humidity from temperature.
    :param temperature: temperature (in degrees c)
    :return: saturated humidity
    """
    es = 6.112 * np.exp(17.6 * temperature / (temperature + 243.5))
    return es


## read in the data we need

cet = xarray.load_dataset(edinburghRainLib.dataDir / 'cet_cpm.nc').tas
ed_extreme_precip = xarray.load_dataset(
    edinburghRainLib.dataDir / 'ed_reg_max_precip.nc').pr  ##.isel(grid_longitude=slice(10,20),grid_latitude=slice(10,20))

ed_ext=ed_extreme_precip.sel(grid_latitude=3.45,grid_longitude=359.62,method='nearest').load()
df_data = [ed_ext.values.flatten()]
ts_summer = cet.resample(time='QS-DEC').mean().dropna('time')
ts_summer = ts_summer.sel(time=(ts_summer.time.dt.month == 6))
ts_summer['time'] = ed_extreme_precip.time  # make sure times are the same
df_data.append(ts_summer.values.flatten())# remove places where x was nan from cov.
cols=['x','cov']
df=pd.DataFrame(np.array(df_data).T,columns=cols)

print("Created df")


with rpy2.robjects.conversion.localconverter(robjects.default_converter + rpandas2ri.converter):
    robjects.globalenv['df'] = df  # push the dataframe with info into R
print("Pushed df to R and ready to fit")
r_code = 'result<-fevd(x=x,data=df,location.fun=~cov,scale.fun=~cov)'
fit = robjects.r(r_code)  # do the fit
print("Fit data")
# and plot it.
robjects.r('plot(result,type="qq")')
# and save it
path=edinburghRainLib.fig_dir/'rplot_qq.png'
path = r'../figures/rplot_qq.png'
robjects.r(f'savePlot(filename="{str(path)}",type="png")')