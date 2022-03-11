#!/bin/env python

"""
Process radar data. 
"""
import pathlib
import commonLib
import xarray
import numpy as np
import pandas as pd
import argparse






def maxValue(da, maxVar, dim='time'):
    """
    Return value from da where maxVar is maximum wrt dim.
    :param da: dataArray where values come from
    :param maxVar: dataArray where index max is found
    :param dim: dim over which to work. Default is 'time'
    :return: dataArray
    """

    bad = da.isnull().all('time', keep_attrs=True)  # find out where *all* data null
    indx = maxVar.argmax(dim='time', skipna=False, keep_attrs=True)  # index of maxes
    result = da.isel(time=indx)
    result = result.where(~bad)  # mask data where ALL is missing
    return result


def time_of_max(da, dim='time'):
    """

    Work out time of maxes using argmax to get index and then select the times.

    """
    bad = da.isnull().all(dim, keep_attrs=True)  # find out where *all* data null

    indx = da.argmax(dim=dim, skipna=False, keep_attrs=True)  # index of maxes
    result = da.time.isel({dim: indx})

    result = result.where(~bad)  # mask data where ALL is missing
    return result


def process(dataArray, resample='1D', total=True):
    """
    process a chunk of data from the hourly processed data
    Compute  dailyMax, dailyMaxTime and dailyMean and return them in a dataset
    :param dataArray -- input data array to be resampled and processes to daily
    :param total (default True) If True  not compute the daily total.

    """
    name_keys = {'1D': 'daily', '1M': 'monthly'}
    name = name_keys.get(resample, resample)

    resamp = dataArray.resample(time=resample, label='left')
    # set up the resample
    mx = resamp.max(keep_attrs=True).rename(name + 'Max')
    if total:
        tot = resamp.sum(keep_attrs=True).rename(name + 'Total')
        tot.attrs['units'] = 'mm/day'
    mxTime = resamp.map(time_of_max).rename(name + 'MaxTime')
    ds = xarray.merge([tot, mx, mxTime])

    return ds


def time_process(DS, varPrefix='daily', summary_prefix=''):
    """
    Process a dataset of (daily) data
    :param DS -- Dataset to process
    :param outFile (default None). Name of file for summary data to output. If None  nothing will be written.
            All times in attributes or "payload" will be converted used commonLib.convert_time
    :param varPrefix (default 'daily') -- variable prefix on DataArrays in datasets)
    :param summary_prefix (default '') -- prefix to be added to output maxes etc
    """
    mx = DS[varPrefix + 'Max'].max('time', keep_attrs=True).rename(f'{summary_prefix}Max')  # max of maxes
    mx_idx = DS[varPrefix + 'Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = DS[varPrefix + 'MaxTime'].isel(time=mx_idx).drop_vars('time').rename(f'{summary_prefix}MaxTime')
    time_max = DS[varPrefix + 'Max'].time.max().values
    mn = DS[varPrefix + 'Total'].mean('time', keep_attrs=True).rename(f'{summary_prefix}Mean')
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn, mx, mx_time])

    ds.attrs['max_time'] = time_max

    return ds


def write_data(ds, outFile, summary_prefix=''):
    """
    Write data to netcdf file -- need to fix times!
      So only converts maxTime
    :param ds -- dataset to be written
    :param outFile: file to write to.
    :param summary_prefix. summary prefix text (usually daily or monthly)
    :return: converted dataset.
    """
    ds2 = ds.copy()  # as modifying things...
    try:
        ds2.attrs['max_time'] = commonLib.time_convert(ds2.attrs['max_time'])
    except KeyError: # no maxTime.,
        pass
    var = summary_prefix + "MaxTime"
    ds2[var] = commonLib.time_convert(ds2[var])
    # compress the ouput... (useful because most rain is zero...and quite a lot of the data is missing)
    encoding=dict()
    comp = dict(zlib=True)
    for v in ds2.data_vars:
        encoding[v]=comp
    ds2.to_netcdf(outFile,encoding=encoding)
    return ds2


def end_period_process(dailyData, outDir, period='1M',extra_name=''):
    if len(dailyData) == 0:
        return [] # nothing to do so return empty list
    name_keys = {'1D': 'daily', '1M': 'monthly'}

    time_write = pd.to_datetime(dailyData[-1].time.values[0])
    time_str = f'{time_write.year:04d}-{time_write.month:02d}'
    print(f"Writing summary and daily {extra_name} data for {len(dailyData)} days for {time_str}")
    summary_prefix = name_keys.get(period, period)

    split_name = f.name.split(".")[0].split("_")
    outFile = outDir / "_".join(
        [split_name[0], split_name[1], time_str, split_name[-1], f'{summary_prefix}{extra_name}.nc'])
    outFileDaily = outDir / "_".join(
        [split_name[0], split_name[1], time_str, split_name[-1], f'daily{extra_name}.nc'])
    dsDaily = xarray.concat(dailyData, 'time')
    resampDS = dsDaily.resample(time=period).map(time_process, summary_prefix=summary_prefix)
    write_data(dsDaily,outFile=outFileDaily,summary_prefix='daily')
    write_data(resampDS, outFile=outFile, summary_prefix=summary_prefix)
    return resampDS # return summary dataset

def two_hr_mean(ds):
    """
    Compute two hour rolling mean.
    """
    return ds.rolling(time=2, min_periods=2).mean()

# read cmd line args.
parser=argparse.ArgumentParser(description="Process UK Nimrod Radar Data to compute hourly maxes on monthly timescales")

parser.add_argument('year',type=int,nargs='+',help='years to process')
parser.add_argument('--resolution','-r',type=str,
                    help='Resolution wanted',choices=['1km','5km'],default='5km')
parser.add_argument('--glob',type=str,help='Pattern for glob month mathcing -- i.e. 0[6-8]',
                    default='[0-1][0-9]')
parser.add_argument('--test','-t',action='store_true',
                    help='If set run in test mode -- no data read or generated')
parser.add_argument('--verbose','-v',action='store_true',
                    help='If set be verbose')
parser.add_argument('--outdir','-o',type=str,help='Name of output directory')
args=parser.parse_args()
glob_patt=args.glob
test = args.test
resoln=args.resolution
outdir=args.outdir
if outdir is None: # default
    outdir=commonLib.outdir/f'summary_{resoln}'
else:
    outdir = pathlib.Path(outdir)
verbose = args.verbose
if verbose:
    print("Command line args",args)

if test:
    print(f"Would create {outdir}")
else:
    outdir.mkdir(parents=True, exist_ok=True) #create directory if needed

# initialise -- note this means results will be a bit different on the 1st of Jan if running years indep..
last_month = None
last = None

dailyData = []
dailyData2hr = []
for year in args.year:
    dataYr = commonLib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
    print(dataYr,dataYr.exists())
    # initialise the list...
    files = sorted(list(dataYr.glob(f'*{year:02d}{glob_patt}[0-3][0-9]*-composite.dat.gz.tar')))
    for f in files:
        if test:
            print(f"Would read {f} but in test mode")
            continue
        rain = commonLib.extract_nimrod_day(f, QCmax=400).resample(time='1h').mean(
            keep_attrs=True)  # mean rain/hour units mm/hr

        time = pd.to_datetime(rain.time.values[0])
        print(f, time)
        # this block will only be run if last_month is not None and month has changed. Taking advantage of lazy evaluation.
        if (last_month is not None) and \
           (time.month != last_month): # change of month -- could be quarter etc
            # starting a new month so save data and start again.
            print("Starting a new period. Summarizing and writing data out")
            summaryDS = end_period_process(dailyData, outdir, period='1M')  # process and write out data
            summaryDS2hr = end_period_process(dailyData2hr, outdir, period='1M', extra_name='2hr')  # process and write out data
            # new month so clean up dailyData & dailyData2hr
            dailyData=[]
            dailyData2hr=[]

        # start again stuff
        dailyData.append(process(rain)) # append rain to the daily list.
        # deal with 2hour data.
        if (last is not None) and  ((time-pd.to_datetime(last.time)).seconds== 3600):
            rain2hr = two_hr_mean(last.combine_first(rain)).isel(time=slice(1, None))
        else:  # nothing to merge so just use the 24 hours we have
            rain2hr = two_hr_mean(rain)
            rain2hr.attrs['two_hr_notes']='Missing the previous day'

        dailyData2hr.append(process(rain2hr))
        #  figure our last period and month.
        last = rain.isel(time=[-1])  # get last hour.
        last_month = pd.to_datetime(last.time).month
        # done loop over files for this year
    # end loop over years.