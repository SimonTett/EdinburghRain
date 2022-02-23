#!/bin/env python 
"""Compute the daily hourly max rainfall from the radar data Seem to have
dodgy seconds data in 2004 & 2005 data, ValueError: invalid second
provided in cftime.datetime(2005, 5, 1, 8, 30, -32767, 0,
calendar='gregorian', has_year_zero=False) Looks like missing data so
might need to fix the nimrod code..

For 1km data need to run job on slurm as looks to be too heavy to run
on interactive node. (Could run a year in parallel...)  Also need to
redefine tmpdir to be homespace and possibly need a workspace...


QC: set missing any time segement that has more than 400 mm/hr
equivalent.  ~ 33 mm/5mins or 100 mm/15 mins. There appears to be
occasional radar artifacts of almost 1024 mm/hr -- which I suspect is
a conversion error.

"""
import xarray

import pathlib 
import itertools
import pandas as pd
import numpy as np
import commonLib

    
def annual_process(DS_daily,time_to_hours=False):
    """
    Process the daily max dataset.
    :param DS_daily -- datset if daily hourly max and time of hourly max.
    """
    # TOD change variable names time_max and max_time rather confusing
    time = DS_daily.time.min().values
    time_max = DS_daily.time.max().values # TODO -- keep it as an xarray coord so meta data can be brought in.
    max = DS_daily.dailyMaxRain.max('time',keep_attrs=True).rename('maxV')  # max of maxes
    max_idx =DS_daily.dailyMaxRain.fillna(0.0).argmax('time',skipna=True) # index  of max
    max_time = DS_daily.dailyMaxTime.isel(time=max_idx).drop_vars('time')
    # actual time. -- dropping time as has nothing and will fix later
    max_time = max_time.rename('maxVTime')
    if time_to_hours:
        max_time  = commonLib.time_convert(max_time)
        time_max  = commonLib.time_convert(time_max,set_attrs=False)
        
    ds = xarray.merge([max,max_time]).expand_dims(time=[time])
    ds.attrs['max_time']=time_max
    return ds

test = True
rgn = dict(projection_x_coordinate=slice(5e4,5e5),
           projection_y_coordinate=slice(5e5,1.1e6)) 
# set rgn to None if don't want slice anything out
resoln= '1km'
daily1hr_max_dir = pathlib.Path("./daily_max")
daily_mean_dir = pathlib.Path("./daily_mean")
annual1hr_max_dir = pathlib.Path(f"./annual_max_{resoln}")
# create the dirs
for direc in [daily1hr_max_dir,annual1hr_max_dir,daily_mean_dir]:
    direc.mkdir(exist_ok=True) # don't die if dir already exisits. 

nimrod_dir=pathlib.Path(f"/badc/ukmo-nimrod/data/composite/uk-{resoln}")
years = range(2006,2022) # from 2004 to 2021.
# full data only exisits since jan 2004 so will work with that. 
# there is image data prior to 2004
# hard bit is extracting the data. processing is easy

file_pattern="[0-9][0-9][0-9][0-9]_*.dat.gz.tar" # all data
expect_files=365
file_pattern="0[6-8][0-9][0-9]_*.dat.gz.tar" # all summer data
expect_files=92

if test:
    years =[2020,2021]#  testing
    file_pattern="*0[7-8][0-9][0-9]_*.dat.gz.tar" # July only -- testing
    expect_files=62

count_data=[] # number of samples in a daily data tar file
count_time=[] # times of the data.
for year in years:
    directory = nimrod_dir/f"{year:04d}"
    if not directory.exists():
        print(f"{directory} does not exist")
        continue
    # pattern to look for
    glob_pattern = f"*{year}{file_pattern}"
    files = directory.glob(glob_pattern)
    no_files=sum(1 for f in files)
    if no_files != expect_files:
        print(f"WARNING {year}: expected {expect_files} but got {no_files}")
    else:
        print(f"{year} OK")

    # iterate over files.
    expect_members = 24*4 # every 15 minutes.
    ds_annual=[] # where we will store the daily datasets -- only for a year.
    last_month = 0
    for file in directory.glob(glob_pattern):
        rain = handyFns.extract_nimrod_day(file)
        if rgn is not None:
            rain = rain.sel(**rgn) # extract the sub-region out.
        # QC - set any values > 400 to nan. (400 mm/hr = 100 mm in 15 minutes)
        rain = rain.where(rain <= 400) 
        time=rain.time.values[0]
        if last_month != rain.time.dt.month.values[0]:
            last_month = rain.time.dt.month.values[0]
            print(f"\n{last_month}: ",end='')
        print(f'{rain.time.dt.day.values[0]:02d}',end=' ',flush=True)
        count_time.append(time)
        count_data.append(len(rain.time))
        rain = rain.resample(time='1h').mean(keep_attrs=True)# mean rain/hour units mm/hr
        # check all on the hour -- and print warning if not..
        if np.any(rain.time.dt.minute !=0):
            print("WARNING -- some data not cannonical times",\
                  rain.time.values[0])

        rain_daily=rain.mean('time',skipna=False,keep_attrs=True).rename('dailyMeanRain')*24 # mean rain -- assuming missing data same as rest
        rain_daily.attrs['units'] = 'mm/day'
        rain_max = rain.max('time',skipna=False,keep_attrs=True).rename('dailyMaxRain') # max rain.
        rain_maxHr = rain.idxmax('time',skipna=False).rename('dailyMaxTime') # time of max rain
        # generate DataSet from max and maxHr then set the time
        rainDS=xarray.merge([rain_max,rain_maxHr]).expand_dims(time=[time])
        rainDSout=xarray.merge([rain_max,rain_maxHr.dt.hour.rename('dailyMaxHr')]).expand_dims(time=[time])
        ds_annual.append(rainDS)
        outfile=daily1hr_max_dir/(file.name.replace('.dat.gz.tar','_max.nc'))
        rainDSout.to_netcdf(outfile)
        outfile=daily_mean_dir/(file.name.replace('.dat.gz.tar','_total.nc'))
        rain_daily.to_netcdf(outfile)
        
    # done with loop over days
    ds_annual=xarray.combine_nested(ds_annual,'time',combine_attrs='drop_conflicts') # merge list of datasets
    ds = annual_process(ds_annual,time_to_hours=True)
    outfile="_".join(file.name.split('_')[0:2])+f"_{year}_{resoln}_max.nc"
    outfile = annual1hr_max_dir/outfile
    ds.to_netcdf(outfile)
    print(f"\n Done with {year} output: {str(outfile)}")
    # end loop over years.
    
data_info = pd.Series(count_data,index=pd.to_datetime(count_time))

"""
##uncompress gzip file
tar.extract(tar.getmembers()[1])
with gzip.open('metoffice-c-band-rain-radar_uk_202012312330_5km-composite.dat.gz') as f_in, open('fred.dat','wb') as f_out:
    shutil.copyfileobj(f_in,f_out)
# size 300678 Feb 21 13:15 fred.dat



fp=tar.extractfile(tar.getmembers()[1])
with gzip.GzipFile("fred2.dat",fileobj=fp) as f_in:
    print(iris.load_cube(f_in)
# fails 



"""
    
            
