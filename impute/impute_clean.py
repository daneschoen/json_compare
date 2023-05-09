import os, sys, argparse
import datetime
from re import sub
from decimal import Decimal
import locale
import json
from itertools import chain
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd

from .validate import *
from apps.errors.errors import *

from . import df_map_propertytype, df_map_propertytype_singleheader, df_map_us_states


'''
`impute` and `clean` rules go here

The function names follow the naming convention in the qa_rules.json
For example, the clean rule 'percentage' is defined by the function 'clean_percentage'
'''

#
# clean rules
#
@except_key
def clean_impute_map(row, col_rule, rule_type, df_map, domain=None, target=None):
    datum = row[col_rule]
    if datum is not None and len(datum.strip()) > 0:
        if domain is None:
            df_res = df_map[df_map.iloc[:,0].str.lower() == datum.strip().lower()]
        else:
            df_res = df_map[df_map[domain].str.lower() == datum.strip().lower()]
        if len(df_res) > 0:
            if target is None:
                return (df_res.iloc[:,0].iloc[0], "")
            return (df_res[target].iloc[0], "")
        else:
            if target is None:
                df_res = df_map[df_map.iloc[:,1].str.lower() == datum.strip().lower()]
            else:
                df_res = df_map[df_map[target].str.lower() == datum.strip().lower()]
            if len(df_res) > 0:
                return (datum, "")
            return (datum, f"Error {rule_type.capitalize()}: Target value")
    return (datum, f"Error {rule_type.capitalize()}: Target value - empty value")


# @except_key
# def clean_us_states(row, col_rule, fn_name):
#     return clean_impute_map(row, col_rule, 'clean', globals()['df_map_'+fn_name])

@except_key
def clean_us_states_error(row, col_rule):
    datum = row[col_rule]
    if datum is not None and len(datum.strip()) > 0:
        res = df_map_us_states[df_map_us_states['abbreviation'].str.lower() == datum.strip().lower()]
        if len(res) > 0:
            return ""
    return "Error Clean: Target value"


@except_key
def clean_us_zipcode2(row, col_rule):
        zip = row[col_rule].strip()
        if len(zip) > 0:
            if zip.lower() == "various" or validate_zipcode_us(zip) == "":
                return (zip, "")
        else:
                return (zip, "Error Clean: Invalid zipcode")


def clean_us_zipcode(row, col_rule):
    try:
        zip = row[col_rule]
        if zip.strip().lower() == "various" or validate_zipcode_us(zip) == "":
            return zip
        else:
            raise ErrorValue("Invalid zipcode")
    except KeyError as e:
        raise KeyError('Missing field ' + str(e))

    # try:
    #     return validate_zipcode_us(datum)
    # except ValueError:
    #     return "Error: Invalid percentage"


@except_value
def clean_percentage(row, col_rule):
    return float(row[col_rule].strip('%'))/100


def clean_currency(row, col_rule):
    return clean_currency_usd(row, col_rule)

def clean_currency_usd(row, col_rule):
    try:
        return str(Decimal(sub(r'[^\d\-.]', '', row[col_rule])))
    except ValueError:
        raise ErrorValue("Invalid currency")

def clean_currency_eur(datum):
    #u"\N{euro sign}"
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR.UTF8')
        conv = locale.localeconv()
        raw_numbers = datum.strip(conv['currency_symbol'].decode('utf-8'))
        return locale.atof(raw_numbers)
    except ValueError:
        raise ErrorValue("Invalid currency")



#
# impute rules
#
@except_key
def impute_propertytype(row, col_rule, kwargs=None):
    res = df_map_propertytype.loc[(df_map_propertytype['PrimaryPropertyType'] == row['PrimaryPropertyType']) & (df_map_propertytype['DetailedPropertyType'] == row['DetailedPropertyType'] ), 'ModelPropertyType']

    return res.iloc[0] if len(res) > 0 else ""


@except_key
def impute_propertytype_singleheader(row, col_rule, kwargs=None):
    if 'DetailedPropertyType' in row:
        res = df_map_propertytype.loc[(df_map_propertytype['PrimaryPropertyType'] == row['PrimaryPropertyType']) & (df_map_propertytype['DetailedPropertyType'] == row['DetailedPropertyType'] ), 'ModelPropertyType']
    else:
        res = df_map_propertytype_singleheader.loc[df_map_propertytype_singleheader['PrimaryPropertyType'] == row['PrimaryPropertyType'], 'CSSAPropertyType']

    return res.iloc[0] if len(res) > 0 else ""

