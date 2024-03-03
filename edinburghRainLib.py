"""
Provides classes, variables and functions used across the project
"""
import pathlib
import fiona
import cartopy
import iris
import tarfile
import gzip
import zlib # so we can trap an error
import tempfile
import shutil
import xarray
import pathlib
import cartopy
import cartopy.feature 
import pandas as pd
import numpy as np
import platform
import cftime
import cf_units
import matplotlib.pyplot as plt
bad_data_err = (zlib.error,ValueError,TypeError,iris.exceptions.ConstraintMismatchError,gzip.BadGzipFile) # possible bad data
#bad_data_err = (zlib.error,iris.exceptions.ConstraintMismatchError,gzip.BadGzipFile) # possible bad data
machine = platform.node()
if ('jasmin.ac.uk' in machine) or ('jc.rl.ac.uk' in machine):
    # specials for Jasmin cluster or LOTUS cluster
    dataDir = pathlib.Path.cwd()
    nimrodRootDir = pathlib.Path("/badc/ukmo-nimrod/data/composite") # where the nimrod data lives
    cpmDir = pathlib.Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85") # where the CPM data lives
    outdir = pathlib.Path(".")  #writing locally. Really need a workspace..
elif 'GEOS-' in machine.upper() :
    dataDir = pathlib.Path(r'C:\Users\stett2\OneDrive - University of Edinburgh\data\EdinburghRainfall')
    nimrodRootDir = dataDir/'nimrod_data'
    outdir = dataDir/'output_data'
else: # don't know what to do so raise an error.
    raise Exception(f"On platform {machine} no idea where data lives")
# create the outdir
outdir.mkdir(parents=True,exist_ok=True)
horizontal_coords = ['projection_x_coordinate','projection_y_coordinate'] # for radar data.
cpm_horizontal_coords= ['grid_latitude', 'grid_longitude'] # horizontal coords for CPM data.
edinburgh_castle = dict(projection_x_coordinate=325166,
           projection_y_coordinate=673477+50) # adjust the castle 50M north!


edinburgh_botanics = dict(projection_x_coordinate=324770,
                          projection_y_coordinate=675587)

edinburgh_KB = dict(projection_x_coordinate=326495,
                          projection_y_coordinate=670628)

# co-ords of relevant places on rotated grid. See comp_rotated_coords to actually generate them
# note only to 2 sf (~ 1km) which is enough precision... Not computing these here
# as needs cordex module which not available on jasmin. Coords around 360 as that is what grid_longitude is
rotated_coords = dict(Edinburgh=(359.62, 3.45),
    Rothampsted=(361.33, -0.68),
    Malvern= (360.11, -0.38),
    Squires_Gate=(359.68, 1.27),
    Ringway=(360.14, 0.85))

# sites we want to plot and the colors they are.
sites=dict(castle=edinburgh_castle,botanics=edinburgh_botanics)#,KB=edinburgh_KB)
edinburgh_region = dict()
for k,v in edinburgh_castle.items(): # 50km around edinburgh
    edinburgh_region[k]=slice(v-50e3,v+50e3)

colors = dict(castle='purple',botanics='green',KB='orange')
try:
    import GSHHS_WDBII
    gshhs = GSHHS_WDBII.GSHHS_WDBII()
    coastline = gshhs.coastlines(scale='full') # so we can have high resoln coastlines.
except ModuleNotFoundError:
    coastline = cartopy.feature.NaturalEarthFeature('physical','coastline','10m',edgecolor='black',facecolor='none')

def get_radar_data(file=dataDir / 'radar_precip/summary_1km_15min.nc', region=None,
                   height_range=slice(0,200), mxMeanRain=1000.):
    """
    read in radar data and mask it by heights and mean rain being reasonable.
    :param file: file to be read in
    :param region: region to extract. If None is Edinburgh region
    :param height_range: Height range to use (all data strictly *between* these values will be used)
    :param mxMeanRain: the maximum mean rain allowed (QC)
    :return: Data masked for region requested & mxTime
    """
    if region is None:
        region = edinburgh_region

    radar_precip = xarray.open_dataset(file).sel(**region)
    rseas = radar_precip.drop_vars('No_samples').resample(time='QS-Dec')
    rseas = rseas.map(time_process, varPrefix='monthly', summary_prefix='seasonal')
    rseas = rseas.sel(time=(rseas.time.dt.season == 'JJA'))  # Summer max rain
    # should compute resample from rseas.
    if '5km' in file.name:
        topog_grid = 55
    else:
        topog_grid = 11
    topog990m = read_90m_topog(region=region, resample=topog_grid)
    top_fit_grid = topog990m.interp_like(rseas.isel(time=0).squeeze())

    htMsk = (top_fit_grid > height_range.start) & (top_fit_grid < height_range.stop)  # in ht range
    # mask by seasonal sum < 1000.
    mskRain = ((rseas.seasonalMean * (30 + 31 + 31) / 4.) < mxMeanRain) & htMsk
    rseasMskmax = xarray.where(mskRain, rseas.seasonalMax, np.nan)
    mxTime = rseas.seasonalMaxTime

    return rseasMskmax,mxTime

def gen_radar_data(file=dataDir / 'radar_precip/summary_1km_15min.nc', quantiles=None,
                   region=None, height_range=slice(0,200), discrete_hr=12,min_count=25):
    """
    generated flattened radar data.
    Read in data, keep data between height_range, mask by seasonal total rainfall < 1000 mm, group by time (discretized to 12 hours by default), requiring at least 25 values.  and then compute quantiles for each grouping

    :param file: file where data lives.
    :param quantiles: quantiles wanted. Defaults (if None set) are [0.05,0.1,0.2,0.5, 0.8, 0.9, 0.95]
    :param region. Region wanted to work with. Specified as dict with each element being co-ord name, slice of min & max values wanted. Used in sel
    :param height_range. Heights which are wanted. Provide as a slice. Heights > start and < stop will be used,
    :param discrete_hr: discretisation time.
    :return: radar data and quantiles of rainfall for Edinburgh castle grouping.
    """


    if quantiles is None:
        quantiles = [0.05,0.1,0.2,0.5, 0.8, 0.9, 0.95]

    rseasMskmax, mxTime = get_radar_data(file=file,region=region,height_range=height_range,mxMeanRain=1000.)
    indx = (mxTime.dt.year - 2000) * 366 + mxTime.dt.dayofyear + (mxTime.dt.hour // discrete_hr) * discrete_hr/24.
    indx = xarray.where(rseasMskmax, indx, np.nan)
    rain_count = rseasMskmax.groupby(indx).count()
    radar_data = rseasMskmax.groupby(indx).quantile(quantiles).rename(group='time_index', quantile='time_quant')  # .values

    ok_count=(~rseasMskmax.isnull()).groupby(indx).sum().rename(group='time_index')    # count non nan
    radar_data=radar_data[(ok_count > min_count)] # want at least 25 values
    # To get the coords -- from chatGPT
    # Assuming you have already performed the groupby and calculated quantiles
    # index = funct(data.time)
    # quants = data.groupby(index).quantiles([0.25, 0.75])
    #
    # Get the quantile values you are interested in (e.g., 25th and 75th percentiles)
    #quantiles = [0.25, 0.75]

    # # Find the spatial coordinates corresponding to the quantile values
    # spatial_coordinates = {}
    # for quantile in quantiles:
    #     # Loop through each group (index_value) in the GroupBy object
    #     for index_value, indices in quants.groups.items():
    #         # Select data for the current group using the indices
    #         group_data = data.isel(time=indices)
    #         # Calculate the quantile value for the current group
    #         quantile_value = quants.sel(quantile=quantile).isel(group=index_value).values
    #
    #         # Find the spatial coordinates where the data is closest to the quantile value
    #         closest_index = np.abs(group_data.values - quantile_value).argmin()
    #         spatial_coords = {dim: coord.values[closest_index] for dim, coord in group_data.coords.items()}
    #
    #         # Store the spatial coordinates in the result dictionary
    #         spatial_coordinates[(index_value, quantile)] = spatial_coords
    #
    # # Print the result
    # for (index_value, quantile), coords in spatial_coordinates.items():
    #     print(f"Index Value: {index_value}, Quantile: {quantile}")
    #     print(coords)

    ed_indx = indx.sel(edinburgh_castle, method='nearest').sel(time='2021')
    rainC2021 = radar_data.sel(time_index=ed_indx).squeeze()
    ed_indx = indx.sel(edinburgh_castle, method='nearest').sel(time='2020')
    rainC2020 = radar_data.sel(time_index=ed_indx).squeeze()
    ds=xarray.Dataset(dict(radar=radar_data,critical2021=rainC2021,critical2020=rainC2020,rain_count=rain_count,indx=indx,mask=rseasMskmax))
    return ds

try:
    import rioxarray # not available on jasmin
    def read_90m_topog(region=edinburgh_region,resample=None):
        """
        Read 90m DEM data from UoE regridded OSGB data.
        Fix various problems
        :param region: region to select to.
        :param resample: If not None then the amount to coarsen by.
        :return: topography dataset
        """
        topog=rioxarray.open_rasterio(dataDir/'uk_srtm')
        topog = topog.reindex(y=topog.y[::-1]).rename(x='projection_x_coordinate',y='projection_y_coordinate')
        if region is not None:
            topog = topog.sel(**region)
        topog = topog.load().squeeze()
        L=(topog > -10000) & (topog < 10000) # fix bad data. L is where data is good!
        topog=topog.where(L)
        if resample is not None:
            topog= topog.coarsen(projection_x_coordinate=resample,
                                 projection_y_coordinate=resample,boundary='pad').mean()
        return topog

except ModuleNotFoundError:
    pass

fig_dir = pathlib.Path("figures")

## hack iris so bad times work...

def hack_nimrod_time(cube, field):
    """Add a time coord to the cube based on validity time and time-window. HACKED to ignore seconds"""
    NIMROD_DEFAULT = -32767.0

    TIME_UNIT = cf_units.Unit(
        "seconds since 1970-01-01 00:00:00", calendar=cf_units.CALENDAR_GREGORIAN
    )

    if field.vt_year <= 0:
        # Some ancillary files, eg land sea mask do not
        # have a validity time.
        return
    else:
        missing = field.data_header_int16_25 # missing data indicator 
        dt =[ 
            field.vt_year,
            field.vt_month,
            field.vt_day,
            field.vt_hour,
            field.vt_minute,
            field.vt_second] # set up a list with the dt cpts
        for indx in range(3,len(dt)): # check out hours/mins/secs and set to 0 if missing
            if dt[indx] == missing:
                dt[indx] = 0

        valid_date = cftime.datetime(*dt) # make a cftime datetime.

    point = np.around(iris.fileformats.nimrod_load_rules.TIME_UNIT.date2num(valid_date)).astype(np.int64)

    period_seconds = None
    if field.period_minutes == 32767:
        period_seconds = field.period_seconds
    elif (
        not iris.fileformats.nimrod_load_rules.is_missing(field, field.period_minutes)
        and field.period_minutes != 0
    ):
        period_seconds = field.period_minutes * 60
    if period_seconds:
        bounds = np.array([point - period_seconds, point], dtype=np.int64)
    else:
        bounds = None

    time_coord = iris.coords.DimCoord(
        points=point, bounds=bounds, standard_name="time", units=TIME_UNIT
    )

    cube.add_aux_coord(time_coord)
import iris.fileformats
iris.fileformats.nimrod_load_rules.time=hack_nimrod_time
print("WARNING MONKEY PATCHING iris.fileformats.nimrod_load_rules.time")


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
    mn = DS[varPrefix + 'Mean'].mean('time', keep_attrs=True).rename(f'{summary_prefix}Mean')
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn, mx, mx_time])

    ds.attrs['max_time'] = time_max

    return ds

def time_convert(DataArray,ref='1970-01-01',unit='h',set_attrs=True):
    """
    convert times to hours (etc) since reference time.
    :param DataAray -- dataArray values to be converted
    :param ref -- reference time as ISO string. Default is 1970-01-01
    :param unit -- unit default is h for hours
    :return -- returns dataarray with units reset and values converted
    """
    name_conversion=dict(m='minutes',h='hours',d='days')
    
    with xarray.set_options(keep_attrs=True):
        hour = (DataArray - np.datetime64(ref))/np.timedelta64(1,unit)
    u = name_conversion.get(unit,unit)
    try:
        hour.attrs['units']=f'{u} since {ref}'
    except AttributeError: # hour does not have attrs.
        pass
    return hour


def extract_nimrod_day(file,region=None,QCmax=None,gzip_min=85,check_date=False):
    """
    extract rainfall data from nimrod badc archive. 
    Archive is stored as a compressed tarfil of gzipped files. 
    Algorithm opens the tarfile. Iterates through files in tarfile 
    uncompresses each file to tempfile. 
    Reads tempfile then deletes it when done.
    returns a dataset of rainfall for the whole day. Note BADC archive
    seems to be missing data so some days will not be complete. 

    :param file -- pathlib path to file for data to be extracted
    :param region (default None) -- if not None then shoul be a dict of co-ords to be extracted.
    :param QCmax (default None) -- if not None then values > QCmax are set missing as crude QC.
    :param check_date (default False) -- if True check the dates are as expected. Complain if not but keep going
    :param gzip_min (default 85) -- if the gziped file (individual field) is less than this size ignore it -- likely becuase it is zero size.

    :example rain=extract_nimrod_day(path_to_file,
                region = dict(projection_x_coordinate=slice(5e4,5e5),
                projection_y_coordinate=slice(5e5,1.1e6)),QCmax=400.)
    """
    rain=[]
    with tarfile.open(file) as tar:
        # iterate over members uncompressing them
        for tmember in tar.getmembers():
            if tmember.size < gzip_min: # likely some problem with the gzip.
                print(f"{tmember} has size {tmember.size} so skipping")
                continue
            with tar.extractfile(tmember) as fp:
                f_out=tempfile.NamedTemporaryFile(delete=False)
                fname = f_out.name
                try: # handle bad gzip files etc...
                    with gzip.GzipFile("somefilename",fileobj=fp) as f_in:
                        # uncompress the data writing to the tempfile
                        shutil.copyfileobj(f_in,f_out) # 
                        f_out.close()
                        # doing various transforms to the cube here rather than all at once. 
                        # cubes are quite large so worth doing. 
                        cube = iris.load_cube(fname)
                        da=xarray.DataArray.from_iris(cube) # read data
                        if region is not None:
                            da = da.sel(**region) # extract if requested
                        if QCmax is not None:
                            da = da.where(da <= QCmax) # set missing all values > QCmax
                        # sort out the attributes
                        da = da.assign_attrs(units=cube.units,**cube.attributes,BADCsource='BADC nimrod data')
                        # drop forecast vars (if we have it) -- not sure why they are there!
                        da = da.drop_vars(['forecast_period','forecast_reference_time'],errors='ignore')
                        rain.append(da) # add to the list
                except bad_data_err:
                    print(f"bad data in {tmember}")
                pathlib.Path(fname).unlink() # remove the temp file.
            # end loop over members          (every 15 or 5  mins)
    # end dealing with tarfile -- which will close the tar file.
    if len(rain) == 0: # no data
        print(f"No data for {file} ")
        return None
    rain = xarray.concat(rain,dim='time') # merge list of datasets
    rain=rain.sortby('time')
    # make units a string so it can be saved.
    rain.attrs['units']=str(rain.attrs['units'])
    if check_date: # check the time...
        date  = pathlib.Path(file).name.split("_")[2]
        wanted_date = date[0:4]+"-"+date[4:6]+"-"+date[6:8]
        udate = np.unique(rain.time.dt.date)
        if len(udate) != 1: # got more than two dates in... 
            print(f"Have more than one date -- ",udate)
        lren=len(rain)
        rain = rain.sel(time=wanted_date) # extract the date
        if len(rain) != lren:
            print("Cleaned data for ",file)
        if len(rain) == 0:
            print(f"Not enough data for {check_date} in {file}")
        
    return rain

def time_process(DS, varPrefix='daily', summary_prefix=''):
    f"""
    Process a dataset of (daily) data
    :param DS -- Dataset to process. Should contain variables called:
      f"{varPrefix}Max", f"{varPrefix}MaxTime", f"{varPrefix}Mean" 
    :param varPrefix (default 'daily') -- variable prefix on DataArrays in datasets)
    :param summary_prefix (default '') -- prefix to be added to output maxes etc
    """
    mx = DS[varPrefix + 'Max'].max('time', keep_attrs=True).rename(f'{summary_prefix}Max')  # max of maxes
    mx_idx = DS[varPrefix + 'Max'].fillna(0.0).argmax('time', skipna=True)  # index  of max
    mx_time = DS[varPrefix + 'MaxTime'].isel(time=mx_idx).drop_vars('time').rename(f'{summary_prefix}MaxTime')
    time_max = DS[varPrefix + 'Max'].time.max().values
    mn = DS[varPrefix + 'Mean'].mean('time', keep_attrs=True).rename(f'{summary_prefix}Mean')
    # actual time. -- dropping time as has nothing and will fix later

    ds = xarray.merge([mn, mx, mx_time])

    ds.attrs['max_time'] = time_max

    return ds




## standard stuff for plots.


# UK local authorities.

# UK nations (and other national sub-units)
nations = cartopy.feature.NaturalEarthFeature(
    category='cultural',name='admin_0_map_subunits',
    scale='10m',facecolor='none',edgecolor='black')

# using OS data for regions --generated by create_counties.py
try:
    regions = cartopy.io.shapereader.Reader(dataDir/'GB_OS_boundaries'/'counties_0010.shp')
    regions = cartopy.feature.ShapelyFeature(regions.geometries(), crs=cartopy.crs.OSGB(), edgecolor='red',facecolor='none')
except fiona.errors.DriverError:
    regions = cartopy.feature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    edgecolor='red',
    facecolor='none')


# radar stations
metadata = pd.read_excel('radar_station_metadata.xlsx',index_col=[0],na_values=['-']).T
L=metadata.Working.str.upper() == 'Y'
metadata = metadata[L]

def std_decorators(ax,showregions=True,radarNames=False):
    """
    Add a bunch of stuff to an axis
    :param ax: axis
    :param showregions: If True show the county borders
    :param radarNames: If True label the radar stations
    :return: Nada
    """


    ax.plot(metadata.Easting,metadata.Northing,marker='h',color='orange',ms=10,linestyle='none',
            transform=cartopy.crs.OSGB(approx=True),clip_on=True) #  radar stations location.
    #ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    if showregions:
        ax.add_feature(regions, edgecolor='red')
    if radarNames:
        for name,row in metadata.iterrows():
            ax.annotate(name,(row.Easting+500,row.Northing+500),transform=cartopy.crs.OSGB(approx=True),
                        annotation_clip=True)

    ax.add_feature(coastline)
    #ax.add_feature(nations, edgecolor='black')
    

