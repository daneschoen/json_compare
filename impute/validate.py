import os, sys, argparse
import datetime
import re
from decimal import Decimal
import locale
import json
from itertools import chain
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd

'''
All validate functions must start with `valididate_` and must return a Truthy or Falsey value:
"" empty string means valid, and non-empty string contains error message:
"Error xyz" means invalid
'''

re_zipcode_us = re.compile(r"^[0-9]{5}(?:-[0-9]{4})?$")
re_zipcode_ca = re.compile(r"^[ABCEGHJ-NPRSTVXY][0-9][ABCEGHJ-NPRSTV-Z] [0-9][ABCEGHJ-NPRSTV-Z][0-9]$")  # [ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s\d[ABCEGHJ-NPRSTV-Z]\d


def message_schema_validate(jsn):

    '''
    v1 data schema : `split`:

    data': {
      "columns": [
          "col 1",
          "col 2"
      ],
      "index": [
          "row 1",
          "row 2"
      ],
      "data": [
          [
              "a",
              "b"
          ],
          [
              "c",
              "d"
          ]
      ]
    }


    v0:

    { 'data':
      [{"col 1":
           ["a", "c"]
       },
       {"col 2":
           ["b", "d"]
       }
      ]
    }

    schema: `row`:

    [
      {__EMPTY: "row 1", __rowNum__: 1, "col 1": "a", "col 2": "b"}
      {__EMPTY: "row 2", __rowNum__: 2, "col 1": "c", "col 2": "d"}
    ]

    '''

    valid_msg = {
      'status_schema': 'ERROR',
      'status_code': 400
    }

    #if 'schema' not in jsn or (jsn['schema'] != 'col' and jsn['schema'] != 'row'):
    if 'schema' not in jsn or (jsn['schema'] != 'split' and jsn['schema'] != 'array'):
        valid_msg['status_schema'] = "ERROR: 'schema' is required as 'split'"
        return valid_msg

    if type(jsn) is not dict:
        valid_msg['status_schema'] = 'ERROR: message needs to be json'
        return valid_msg

    if 'data' not in jsn:
        valid_msg['status_schema'] = "ERROR: payload 'data' is missing"
        return valid_msg

    # v0:
    # if type(jsn['data']) is not list:
    #     valid_msg['status_schema'] = 'ERROR: data payload is not a list'
    #     return valid_msg

    # get_json_file

    # for col in jsn['data']:
    #     if type(col) is not dict:
    #         valid_msg['status_schema'] = 'ERROR: data payload list does not consist of dictionaries (hashmap objects)'
    #         return valid_msg
    #     for k,v in col.items():
    #         if type(v) is not list:
    #             valid_msg['status_schema'] = 'ERROR: column values is not a list'
    #             return valid_msg

    try:
        pd.read_json(json.dumps(jsn['data']), orient='split')
    except ValueError:
        return "Error: Not valid split schema"

    valid_msg = {
      'status_schema': 'ok',
      'status_code': 200
    }
    return valid_msg



def validate_empty_is_error(datum):
    if datum is not None and len(datum.strip()) > 0:
        return ""
    else:
        return "Error Validation: Empty value"


def validate_data_type(datum, type):
    if type == "number":
        try:
            float(datum)
            return ""
        except ValueError:
            return "Error Validation: Not a number"

    if type == "string":
        if isinstance(datum, str):
            return ""
        else:
            return "Error Validation: Not a string"

    if type == "date":
        try:
            pd.to_datetime(datum)
            return ""
        except ValueError:
            return "Error Validation: Not a date"

    return ""


def validate_max_val_error(datum, term):
    try:
        val = float(datum)
        limit = float(term)
        if val >= limit:
            return "Error Validation: Max value " + str(term)
        return ""
    except ValueError:
        return "Error Validation: Not a number"

def validate_max_val_warning(datum, term):
    res = validate_max_val_error(datum, term)
    if res == "":
        return ""
    else:
        if "Error Validation: Max" in res:
            return res.replace("Error", "Warning")
        else:
            return res


def validate_min_val_error(datum, term):
    try:
        val = float(datum)
        limit = float(term)
        if val <= limit:
            return "Error Validation: Min value " + str(term)
        return ""
    except ValueError:
        return "Error Validation: Not a number"

def validate_min_val_warning(datum, term):
    res = validate_min_val_error(datum, term)
    if res == "":
        return ""
    else:
        if "Error Validation: Min" in res:
            return res.replace("Error", "Warning")
        else:
            return res


def validate_discrete_val(datum, lst):
    if datum in lst:
        return ""
    else:
        return "Error Validation: Not in discrete list"


def validate_zipcode_us(datum):
    if re_zipcode_us.match(datum.strip()) is None:
        return "Error Validation: Invalid zipcode"
    return ""

def validate_zipcode_ca(datum):
    if re_zipcode_ca.match(datum.strip()) is None:
        return "Error Validation: Invalid zipcode"
    return ""

