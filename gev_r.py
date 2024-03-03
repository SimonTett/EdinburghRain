"""
SW to gev fit with R in python.
"""

import xarray
import scipy.stats
import pandas as pd
import os
import numpy as np
import collections

os.environ[
    'R_HOME'] = r'C:\Users\stett2\AppData\Local\Microsoft\AppV\Client\Integration\FC689017-A9BB-4A9B-B971-6AC52117BA03\Root'  # where R is...
os.environ['R_HOME']=\
r"C:\Users\stett2\AppData\Local\Microsoft\AppV\Client\Integration\0E3F48F7-DE80-473E-ABFA-A6BC515ABDCA\Root"
# will need adjusting depending on where R got installed
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri as rpandas2ri

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
packnames = (['extRemes'])  # list of packages to install.
# From example rpy2 install what needs to be installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(robjects.vectors.StrVector(names_to_install))
for package in packnames:
    rpackages.importr(package)  # so available
#gevFit = collections.namedtuple('gevFit', ['params', 'std_err', 'AIC', 'negloglike'])


def gev_fit(x, cov=None,  shapeCov=False, **kwargs):
    """
    Do GEV fit using R and return named tuple of relevant values.
    :param x: Data to be fit
    :param cov: covariate value (if None then not used)
    :param returnType. Type to return, allowed is:
        named -- return a named tuple (default)
        tuple -- return a tuple -- useful for apply_ufunc
        DataSet -- return a DataSet
    :param shapeCov -- If True allow the shape to vary with the covariate.
    :return: A dataset of the parameters of the fit.
    """

    L = ~np.isnan(x)
    df_data = [x[L]]  # remove nan from data]
    cols = ['x']
    npts=3
    if cov is not None:
        npts+=2
        if shapeCov:
            npts+=1
    r_code = 'fevd(x=x,data=df'
    if cov is not None:
        df_data.append(cov[L])  # remove places where x was nan from cov.
        cols.append('cov')
        r_code = r_code + ',location.fun=~cov,scale.fun=~cov'
        if shapeCov:
            r_code += ',shape.fun=~cov'
    r_code += ')'  # add on the trailing bracket.
    r_code_summ = 'summary(' + r_code + ',silent=TRUE)'
    df = pd.DataFrame(np.array(df_data).T, columns=cols)
    with rpy2.robjects.conversion.localconverter(robjects.default_converter + rpandas2ri.converter):
        robjects.globalenv['df'] = df  # push the dataframe with info into R
    try:
        fit = robjects.r(r_code_summ)  # do the fit
        # extract the data
        params = fit.rx2('par')
        se = fit.rx2('se.theta')

        if isinstance(se,rpy2.rinterface_lib.sexp.NULLType):
            # std err not present (for some reason) set everything to nan
            se = np.repeat(np.nan,npts)
        else:
            se = np.array(se)
        if isinstance(params,rpy2.rinterface_lib.sexp.NULLType):
            # std err not present (for some reason) set everything to nan
            params = np.repeat(np.nan,npts)
        else:
            params = np.array(params)

        # add on zeros if cov is None
        if cov is None:
            se = np.append(se,np.zeros(3))
            params = np.append(params,np.zeros(3))

        else: # need to shuffle data so dlocation, dscale move...
        # add on zero for covariance with shape when shapeCov is False
            if not shapeCov:
                params = np.append(params, 0.0)  # Dshape is 0.
                se = np.append(se, 0.0)
            shuffle = [0,2,4,1,3,5]
            params = params[shuffle]
            se=se[shuffle]


        # now shuffle the params around. We expect to have 6 at this point.
        params[[2,5]] *= -1  # negate two shape parameters as python convention differs from R
        nllh = np.array(fit.rx2('nllh'))
        aic = np.array(fit.rx2('AIC'))
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        # some problem in R with the fit. Set everything to nan.
        params = np.repeat(np.nan,6)
        se=params
        nllh=np.array([np.nan])
        aic = nllh

    return (params,se,nllh,aic)




import scipy.stats
def gev_fit_python(data,**kwargs):
    fit = scipy.stats.genextreme.fit(data,**kwargs)
    fit = np.array([fit[1],0.0,fit[2],0.0,fit[0],0.0]) # return in order location, scale, shape
    return fit
def xarray_gev_python(ds,dim='time_ensemble', file=None, recreate_fit=False, verbose=False, **kwargs):
    """
    Fit a GEV to xarray data using scipy.stats. Less powerful than R
    :param ds: dataset for which GEV is to be fit
    :param dim: The dimension over which to collapse.
    :param file -- if defined save fit to this file. If file exists then read data from it and so not actually do fit.
    :param recreate_fit -- if True even if file exists compute fit.
    :param verbose -- be verbose if True
    :param kwargs: any kwargs passed through to the fitting function
    :return: a dataset containing:
        Parameters -- the parameters of the fit; location, scale, shape
    """
    if (file is not None) and file.exists() and (not recreate_fit): # got a file specified, it exists and we are not recreating fit
        ds=xarray.load_dataset(file) # just load the dataset and return it
        if verbose:
            print(f"Loaded existing data from {file}")
        return ds

    params = xarray.apply_ufunc(gev_fit_python, ds, input_core_dims=[[dim]],
                                                   output_core_dims=[['parameter'] ],
                                                   vectorize=True, kwargs=kwargs)
    pnames = ['location', 'scale', 'shape','Dlocation', 'Dscale',  'Dshape']


    params = params.rename("Parameters")
    ds = xarray.Dataset(dict(Parameters=params)).assign_coords(parameter=pnames)
    if file is not None:
        ds.to_netcdf(file)  # save the dataset.
        if verbose:
            print(f"Wrote fit information to {file}")
    return ds
def xarray_gev(dataArray, cov=None, shapeCov=False, dim='time_ensemble', file=None, recreate_fit=False, verbose=False, **kwargs):
    """
    Fit a GEV to xarray data using R.
    :param dataArray: dataArray for which GEV is to be fit
    :param cov: covariate (If None not used) -- another dataArray. TODO generalize to multuple dataarrays
    :param shapeCov: If True allow the shape to vary with the covariate.
    :param dim: The dimension over which to collapse.
    :param file -- if defined save fit to this file. If file exists then read data from it and so not actually do fit.
    :param recreate_fit -- if True even if file exists compute fit.
    :param verbose -- be verbose if True
    :param kwargs: any kwargs passed through to the fitting function
    :return: a dataset containing:
        Parameters -- the parameters of the fit; location, location wrt cov, scale, scale wrt cov, shape, shape wrt cov
        StdErr -- the standard error of the fit -- same parameters as Parameters
        nll -- negative log likelihood of the fit -- measure of the quality of the fit
        AIC -- aitkin information criteria.
    """
    if (file is not None) and file.exists() and (not recreate_fit): # got a file specified, it exists and we are not recreating fit
        dataArray=xarray.load_dataset(file) # just load the dataset and return it
        if verbose:
            print(f"Loaded existing data from {file}")
        return dataArray

    #kwargs['returnType'] = 'tuple'
    kwargs['shapeCov'] = shapeCov
    if cov is None:
        params, std_err, nll, AIC = xarray.apply_ufunc(gev_fit, dataArray, input_core_dims=[[dim]],
                                                       output_core_dims=[['parameter'], ['parameter'], ['NegLog'], ['AIC']],
                                                       vectorize=True, kwargs=kwargs)
    else:
        params, std_err, nll, AIC = xarray.apply_ufunc(gev_fit, dataArray, cov, input_core_dims=[[dim], [dim]],
                                                       output_core_dims=[['parameter'], ['parameter'], ['NegLog'], ['AIC']],
                                                       vectorize=True, kwargs=kwargs)

    pnames = ['location', 'scale','shape',  'Dlocation', 'Dscale', 'Dshape']
    # name variables and then combine into one dataset.

    params = params.rename("Parameters")
    std_err = std_err.rename("StdErr")
    nll = nll.rename('nll').squeeze()
    AIC = AIC.rename('AIC').squeeze()
    dataArray = xarray.Dataset(dict(Parameters=params, StdErr=std_err, nll=nll, AIC=AIC)).assign_coords(parameter=pnames)
    if file is not None:
        dataArray.to_netcdf(file) # save the dataset.
        if verbose:
            print(f"Wrote fit information to {file}")
    return dataArray


## use apply ufunc to generate distributions...

def fn_isf(c, loc, scale, p=None, dist=scipy.stats.genextreme):
    x = dist.isf(p,c, loc=loc, scale=scale)
    #x = fdist.isf(p)  # values for 1-cdf.
    return x


def fn_sf(c, loc, scale, x=None, dist=scipy.stats.genextreme):
    p = dist.sf(x,c, loc=loc, scale=scale)
    #p = fdist.sf(x)  # 1-cdf for given x
    return p

def fn_interval(c, loc, scale, alpha=None, dist=scipy.stats.genextreme):
    #fdist = dist(c, loc=loc, scale=scale)
    range = dist.interval(alpha,c,loc=loc,scale=scale)  # range for dist
    return np.array([range[0],range[1]])


def xarray_sf(x,params, output_dim_name='value'):
    """
    Compute the survival value for different values based on dataframe of fit parameters.
    :param params: xarray dataarray of shape, location and scale values
    :param output_dim_name: name of output dimension. Default is "value" but set it to what ever you are using. E.g "Rx1hr"
    :param kwargs: passed to fn_sf which does the computation. Must contain x which is used for the computation.
    :return:dataset of survival function values (1-cdf for values specified)

    """
    sf = xarray.apply_ufunc(fn_sf, params.sel(parameter='shape'), params.sel(parameter='location'),
                            params.sel(parameter='scale'),
                            output_core_dims=[[output_dim_name]],
                            vectorize=True, kwargs=dict(x=x))
    sf = sf.assign_coords({output_dim_name: x}).rename('sf')

    return sf

def xarray_interval(alpha,params):
    """
    Compute the interval for different values based on dataframe of fit parameters.
    :param params: xarray dataarray of shape, location and scale values
    :param alpha -- alpha value for interval fn.
    :return:dataset of intervals
    """
    interval = xarray.apply_ufunc(fn_interval, params.sel(parameter='shape'), params.sel(parameter='location'),
                            params.sel(parameter='scale'),
                            output_core_dims=[['interval']],
                            vectorize=True, kwargs=dict(alpha=alpha))
    offset = (1-alpha)/2
    interval = interval.Parameters.assign_coords(interval=[offset,1-offset]).rename('interval')

    return interval


def xarray_isf(p,params):
    """
    Compute the inverse survival function for specified probability values
    :param output_dim_name: name of output_dim -- default is probability
    :param params: dataset of parameter values.
    :param p: sf values at which values to be computed
    :param kwargs:Additional keyword arguments passes to fn_isf. Make sure p is set.
    :return:
    """
    output_dim_name = 'probability'
    x = xarray.apply_ufunc(fn_isf, params.sel(parameter='shape'), params.sel(parameter='location'),
                           params.sel(parameter='scale'),
                           output_core_dims=[[output_dim_name]],
                           vectorize=True, kwargs=dict(p=p))
    x = x.assign_coords({output_dim_name: p}).rename('isf')
    return x

def param_cov(params,cov):
    raise Warning("Use param_at_cov")
    p = ['location', 'scale', 'shape']
    p2 = ["D" + a.lower() for a in p]
    params_c=params.Parameters.sel(parameter=p2).assign_coords(parameter=p) * cov + params.Parameters.sel(parameter=p)
    return params_c

def param_at_cov(params,cov):
    p = ['location', 'scale', 'shape']
    p2 = ["D" + a.lower() for a in p]
    params_c=params.sel(parameter=p2).assign_coords(parameter=p) * cov + params.sel(parameter=p)
    return params_c