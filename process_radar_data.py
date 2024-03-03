#!/bin/env python

"""
Process radar data on JASMIN. Generate monthly-means of rain, monthly extremes and time of monthly extreme.


"""
import pathlib
import edinburghRainLib
import xarray
import numpy as np
import pandas as pd
import argparse
import resource # so we can see memory usage every month.
import warnings

warnings.filterwarnings('ignore',message='.*Vertical coord 5 not yet handled.*')
# warning from iris filter out!





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
    :param total (default True) If True   compute the daily total.

    """
    name_keys = {'1D': 'daily', '1M': 'monthly'}
    name = name_keys.get(resample, resample)
    resamp = dataArray.resample(time=resample, label='left')
    # set up the resample
    mx = resamp.max(keep_attrs=True).rename(name + 'Max')
    nsamps = mx.attrs.pop('Number_samples')
    if total: # this is problematic if process 15min data. In which case total will be 4 times too large.
        #TODO fix processing.
        tot = resamp.sum(keep_attrs=True).rename(name + 'Total')
        tot.attrs['units'] = 'mm/day'
        nsamps = tot.attrs.pop('Number_samples')

    mxTime = resamp.map(time_of_max).rename(name + 'MaxTime')
    ds = xarray.merge([tot, mx, mxTime])
    # add in the number of samples (and their time)
    try:
        v = xarray.DataArray([nsamps],dims=dict(time=mx.time))
        ds = ds.assign(No_samples=v)
    except KeyError: # not there
        pass
        
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
    #TODO modify to make a total.
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
        ds2.attrs['max_time'] = edinburghRainLib.time_convert(ds2.attrs['max_time'])
    except KeyError: # no maxTime.,
        pass
    var = summary_prefix + "MaxTime"
    ds2[var] = edinburghRainLib.time_convert(ds2[var])
    # compress the ouput... (useful because most rain is zero...and quite a lot of the data is missing)
    encoding=dict()
    comp = dict(zlib=True)
    for v in ds2.data_vars:
        encoding[v]=comp
    ds2.to_netcdf(outFile,encoding=encoding)
    return ds2


def end_period_process(dailyData, outDir, period='1M',extra_name='',writeDaily=True):

    """
    Deal with data processing at end of a period -- normally a calendar month.
    :param dailyData -- input list of daily datasets
    :param outDir -- output directory where data will be written to
    :param period (default '1M'). Time period over which data to be resampled to
    :param extra_name (defult '')
    :param writeDaily (default True). If True write out daily data
    """
    if len(dailyData) == 0:
        return [] # nothing to do so return empty list
    name_keys = {'1D': 'daily', '1M': 'monthly'}
    no_days = len(dailyData)
    time_write = pd.to_datetime(dailyData[-1].time.values[0])
    time_str= f'{time_write.year:04d}-{time_write.month:02d}'
    summary_prefix = name_keys.get(period, period)

    split_name = f.name.split(".")[0].split("_")
    outFile = outDir / "_".join(
        [split_name[0], split_name[1], time_str, split_name[-1], f'{summary_prefix}{extra_name}.nc'])
    outFileDaily = outDir / "_".join(
        [split_name[0], split_name[1], time_str, split_name[-1], f'daily{extra_name}.nc'])
    dsDaily = xarray.concat(dailyData, 'time')
    resampDS = dsDaily.resample(time=period).map(time_process, summary_prefix=summary_prefix)
    resampDS = resampDS.assign(No_samples=dsDaily.No_samples.rename(time='time_sample')) # add in the no of samples and  time of the sample
    if writeDaily:
        write_data(dsDaily,outFile=outFileDaily,summary_prefix='daily')
        print(f"Wrote daily data for {no_days} days for {time_str} to {outFileDaily}")

    write_data(resampDS, outFile=outFile, summary_prefix=summary_prefix)
    print(f"Wrote summary {extra_name} data for {no_days} days for {time_str} to {outFile}")

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
parser.add_argument('--monitor','-m',action='store_true',
                    help='Monitor memory')
parser.add_argument('--region',nargs=4,type=float,help='Region to extract (x0, x1,y0,y1)')

parser.add_argument('--minhours',type=int,help='Minium number of unique hours in data to process daily data (default 6 hours)',default=6)
parser.add_argument("--nodaily",action='store_false',
                    help='Do not write out daily data')
parser.add_argument("--resample",type=str,help='Time to resample input radar data to (default = 1h)',default='1h')
args=parser.parse_args()
glob_patt=args.glob
test = args.test
resoln=args.resolution
outdir=args.outdir
if outdir is None: # default
    outdir=edinburghRainLib.outdir/f'summary_{resoln}'
else:
    outdir = pathlib.Path(outdir)
writeDaily=args.nodaily    
verbose = args.verbose
region=None
if args.region is not None:
    region=dict(
        projection_x_coordinate = slice(args.region[0],args.region[1]),
        projection_y_coordinate = slice(args.region[2],args.region[3])
    )

resample_prd = args.resample
if verbose:
    print("Command line args",args)

monitor = args.monitor
minhours = args.minhours
max_mem= 0
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
    dataYr = edinburghRainLib.nimrodRootDir / f'uk-{resoln}/{year:04d}'
    # initialise the list...
    files = sorted(list(dataYr.glob(f'*{year:02d}{glob_patt}[0-3][0-9]*-composite.dat.gz.tar')))
    for f in files:
        if test: # test mode
            print(f"Would read {f} but in test mode")
            continue
        rain = edinburghRainLib.extract_nimrod_day(f, QCmax=400,check_date=True,region=region)
        if (rain is None) or (len(np.unique(rain.time.dt.hour)) <= minhours): # want min hours hours of data
            print(f"Not enough data for {f} ")
            if rain is None:
                print("No data at all")
            else:
                print(f"Only {len(np.unique(rain.time.dt.hour))} unique hours")

            last = None # no last data for the next day
            continue # no or not enough data so onto the next day
        # now we can process a days worth of data.    
        no_samples = len(rain.time)
        rain=rain.resample(time=resample_prd).mean(keep_attrs=True)  # mean rain/hour units mm/hr
        rain.attrs["Number_samples"]=no_samples # how many samples went in
        time = pd.to_datetime(rain.time.values[0])
# from https://medium.com/survata-engineering-blog/monitoring-memory-usage-of-a-running-python-program-49f027e3d1ba
        if monitor: # report on memory
            mem =  
            max_mem = max(max_mem,mem) 
            print(f"Memory: {mem}, Max Mem {max_mem}")
        print(f, time)
        # this block will only be run if last_month is not None and month has changed. Taking advantage of lazy evaluation.
        if (last_month is not None) and \
           (time.month != last_month): # change of month -- could be quarter etc
            # starting a new month so save data and start again.
            print("Starting a new period. Summarizing and writing data out")
            summaryDS = end_period_process(dailyData, outdir, period='1M',writeDaily=writeDaily)  # process and write out data
            summaryDS2hr = end_period_process(dailyData2hr, outdir, period='1M', extra_name='2hr',writeDaily=writeDaily)  # process and write out data
            if monitor: # report on memory
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
                max_mem = max(max_mem,mem) 
                print(f"Memory: {mem}, Max Mem {max_mem}")

            # new month so clean up dailyData & dailyData2hr
                

            dailyData=[]
            dailyData2hr=[]

        # end of start again stuff
        # deal with 2hour data.
        if (last is not None) and  ((time-pd.to_datetime(last.time)).seconds== 3600):
            rain2hr = two_hr_mean(last.combine_first(rain)).isel(time=slice(1, None))
        else:  # nothing to merge so just use the 24 hours we have
            rain2hr = two_hr_mean(rain)
            rain2hr.attrs['notes']='Missing the previous day'
        dailyData.append(process(rain)) # append rain to the daily list.
        dailyData2hr.append(process(rain2hr))
        #  figure our last period and month.
        last = rain.isel(time=[-1])  # get last hour.
        last_month = pd.to_datetime(last.time).month
        #print("End of loop",last_month)
        # done loop over files for this year
    # end loop over years.
# and we might have data to write out!

summaryDS = end_period_process(dailyData, outdir, period='1M',writeDaily=writeDaily)  # process and write out data
summaryDS2hr = end_period_process(dailyData2hr, outdir, period='1M', extra_name='2hr',writeDaily=writeDaily)  # process and write out data

if monitor: # report on memory
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss 
    max_mem = max(max_mem,mem) 
    print(f"Memory: {mem}, Max Mem {max_mem}")
