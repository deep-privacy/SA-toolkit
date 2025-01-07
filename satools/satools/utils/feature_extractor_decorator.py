import inspect
import torch
import os

import functools

# REQUIRED For torchscript type anontation of functools.wrap to be found
import satools
from .. import hifigan
from .. import chain
import torch.nn as nn
import torch.nn.functional as F


from . import fs
from . import torch_utils
from .. import script_utils

def _register_feature_extractor(compute_device="cpu", scp_cache=False, sequence_feat=True):
    cache = fs.SCPCache(enabled=scp_cache,
                              specifier="scp:{dir}{func}{name}{worker}.scp",
                              )
    def wrapper(func):
        func = cache.decorate()(func)
        #  cache.update_formatter({"dir": "./"})
        def model_feat_wrapper(hisself, fname, egs, specifier_format, key,  exec_in_decorator=False, ask_compute="cpu"):
            with torch.no_grad():

                if not exec_in_decorator:
                    result = func(hisself, egs)

                if cache.enabled:
                    cache.key = lambda hisself, egs:key(egs)

                specifier_format["func"] = fname
                specifier_format_global = {k:v for k,v in specifier_format.items() if k in ["dir", "func"]}
                specifier_format_global["worker"] = ""
                specifier_format_global["name"] = ""
                if cache.enabled:
                    cache.update_formatter(specifier_format_global)
                result = func(hisself, egs, cache_formatter=specifier_format_global, from_cache_or_none=True)

                if result == None and str(ask_compute).startswith(compute_device):
                    cache.update_formatter(specifier_format)
                    result = func(hisself, egs.to(ask_compute), cache_formatter=specifier_format)

                if compute_device == "cuda" and not exec_in_decorator:
                    egs = egs.to("cpu")
                    if result != None:
                        result = result.detach().cpu()
                    torch.cuda.empty_cache()

                return result
        model_feat_wrapper.cache = cache
        model_feat_wrapper.sequence_feat = sequence_feat
        return model_feat_wrapper

    return wrapper

if os.getenv('SA_JIT_TWEAK', 'False').lower() == "false":
    def register_feature_extractor(compute_device="cpu", scp_cache=False, sequence_feat=True):
        return _register_feature_extractor(compute_device, scp_cache, sequence_feat)
else:
    def register_feature_extractor(compute_device="cpu", scp_cache=False, sequence_feat=True):
        def decorator(func):
            @functools.wraps(func)
            def wrapped_func(hisself, *args, **kwargs):
                result = func(hisself, *args, **kwargs)
                return result
            return wrapped_func
        return decorator



def extract_features_from_decorator(instance:torch.nn.Module, egs, ask_compute="cpu", specifier_format={}, cache_funcs=[], key=lambda x:x):
    decorated_functions = extract_features_fnames(instance)
    # cache on for everyone by default if cache_funcs not provided
    if len(cache_funcs) == 0:
        cache_funcs = [f[0] for f in decorated_functions]
    ret = {}
    for fname, fdef in decorated_functions:
        if fdef.cache.enabled and fname not in cache_funcs: # disable if not in cache_funcs
            fdef.cache.enabled = False
        feat = getattr(instance, fname)(fname, egs, exec_in_decorator=True, ask_compute=ask_compute, specifier_format=specifier_format, key=key)
        if feat != None:
            if not fdef.sequence_feat:
                fname = fname+"_no_sample"
            ret[fname] = feat

    return ret


def extract_features_fnames(instance:torch.nn.Module):
    # Get all functions from the class
    functions = inspect.getmembers(instance.__class__, predicate=inspect.isfunction)

    if isinstance(instance, torch_utils.WrappedTorchDDP):
        functions = inspect.getmembers(instance.module.module.__class__, predicate=inspect.isfunction)

    # Filter functions that have the decorator
    decorated_functions = [(f[0], f[1]) for f in functions if f[1].__name__.startswith("model_feat_wrapper")]
    return decorated_functions


def merge_cache(instance:torch.nn.Module, cache_path):
    for fname, fdef in extract_features_fnames(instance):
        if fdef.cache.enabled:
            merged_scp = "{cache_path}{func}.scp".format(cache_path=cache_path, func=fname)
            script_utils.run("cat " + "{cache_path}{func}*.scp".format(cache_path=cache_path, func=fname) + " > "+merged_scp,
                quit_on_error=False, shell=True)
