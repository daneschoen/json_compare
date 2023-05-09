import os, sys, argparse
import datetime
import json, string
from itertools import chain
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd

from . import qa_rules
from apps.settings.settings import QA_RULES_PATH, QA_RULES_FILENAME
from apps.json_utilities.utilities_json import get_json_file, get_df_excel, convert_json_row_col, jprint
from apps.errors.errors import *
from .validate import *
from .impute_clean import *
from apps.parser.parser_math import lambda_eval, lambda_function


def run_rules(df_data, reload_rules_file=True, rules_filename=QA_RULES_FILENAME, rules_include=None):
    ''' returns:
    { "error_warning_columns":
      [{"col 1":
           ["", "Error: ...",""]
       },
       {"col 2":
           ["Error: missing"; "Error: beyond max","",""]
       },
       {"col 3":
           [""; "Warning: over",""]
       }
      ]
    }
    '''

    error_warning_columns={}
    # error_warning_summary={}

    valid_msg = {
      #'errors_exist': False,
      #'warnings_exist': False,
      'error_warning_columns': None,
      'status_code': 400
    }

    rules_dct = qa_rules
    if rules_include:
        rules_dct = rules_include

    elif reload_rules_file:
        rules_path_filename = os.path.abspath(os.path.join(QA_RULES_PATH, rules_filename))
        try:
            rules_dct = get_json_file(rules_path_filename)
        except FileNotFoundError:
            valid_msg['status'] = f"File not found: {rules_filename}"
            return valid_msg

    schema_ret = rules_dct.get('schema_ret', 'records')

    col_map = {k: "" for k in list(df_data.columns)}
    col_names = list(df_data.columns)
    col_names_proto = list(df_data.columns)

    for col_rule, v in rules_dct.items():
        rule_clean = v.get("rule_clean", None)
        rule_impute = v.get("rule_impute", None)
        rule_validate = v.get("rule_validate", None)

        if rule_validate:
            if col_rule not in df_data:
                if 'aliases' in rule_validate:
                    for c in df_data.columns:
                        if c in rule_validate['aliases']:
                            df_data = df_data.rename(columns={c: col_rule})
                            col_map[c] = col_rule
                            break
            else:
                col_map[col_rule] = col_rule

        # Clean Rules
        if rule_clean:
            #error_warning_summary['cols_clean'].append(rule_clean)
            err_warn_col = []
            if col_rule in df_data:
                if rule_clean.get("all_or_none", True):
                    try:
                        if 'function' in rule_clean:
                            #cols_clean_fn.append(col_rule)
                            fn_name = rule_clean['function']
                            if "clean_" + rule_clean['function'] in globals():
                                se_res = df_data.apply(lambda row: globals()["clean_" + fn_name](row, col_rule, fn_name), axis=1)
                            else:
                                se_res = df_data.apply(lambda row: clean_impute_map(row, col_rule, 'clean', globals()['df_map_'+fn_name]), axis=1)

                            if type(se_res.iloc[0]) is tuple:
                                df_data[col_rule] = se_res.apply(lambda x: x[0])
                                if rule_clean.get("error_target", True):
                                    error_warning_columns = append_error_warning(error_warning_columns, col_rule, list(se_res.apply(lambda x: x[1])))
                            else:
                                df_data[col_rule] = se_res
                        if 'lambda' in rule_clean:
                            #cols_clean_lambda.append(col_rule)
                            df_data[col_rule] = lambda_function(df_data, rule_clean['lambda'])
                        # if rule_clean.get("error_target", True):
                        #     err_warn_col = df_data.apply(lambda row: globals()["clean_" + rule_clean['function'] + "_error"](row, col_rule), axis=1)
                        #     error_warning_columns = append_error_warning(error_warning_columns, col_rule, list(err_warn_col))
                    except Exception as e:
                        err_warn_col = ['Error Clean: ' + get_err_msg(e)]*df_data.shape[0]
                        error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)
                        #continue
                else:
                    for index, row in df_data.iterrows():
                        try:
                            df_data[col_rule] = df_data.apply(globals()["clean_" + rule_clean['function']](col_rule), axis=1)
                            error_warn_col.append("")
                        except Exception as e:
                            error_warn_col.append('Error Clean: ' + get_err_msg(e))
                            continue
            else:
                err_warn_col = ['Error Clean: Missing field']*df_data.shape[0]
                error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)
                # continue

        # Impute Rules
        if rule_impute:
            err_warn_col = []
            if col_rule not in df_data:
                if rule_impute.get("all_or_none", True):
                    try:
                        if 'function' in rule_impute:
                            #cols_impute_fn.append(col_rule)
                            df_data.insert(0, col_rule, df_data.apply(lambda row: globals()["impute_" + rule_impute['function']](row, col_rule), axis=1))
                            col_map[col_rule] = col_rule
                            col_names.insert(0, col_rule)
                        if 'lambda' in rule_impute:
                            #cols_impute_fn.append(col_rule)
                            df_data.insert(0, col_rule, lambda_function(df_data, rule_impute['lambda']))
                            col_map[col_rule] = col_rule
                            col_names.insert(0, col_rule)
                        if rule_clean.get("error_target", False):
                            err_warn_col = df_data.apply(lambda row: globals()["impute_" + rule_clean['function'] + "_error"](row, col_rule), axis=1)
                            error_warning_columns = append_error_warning(error_warning_columns, col_rule, list(err_warn_col))
                    except Exception as e:
                        err_warn_col = ['Error Impute: ' + get_err_msg(e)]*df_data.shape[0]
                        error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)
                        # continue
                else:
                    for index, row in df_data.iterrows():
                        try:
                            df_data[col_rule] = df_data[col_rule].apply(globals()["impute_" + rule_impute['function']])
                            error_warn_col.append("")
                        except Exception as e:
                            error_warn_col.append('Error Impute: '+ get_err_msg(e))
                            # continue
            else:
                err_warn_col = ['Error Impute: Column already exists']*df_data.shape[0]
                error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)
                # continue

        # Validation Rules:
        if rule_validate:
            if rule_validate.get("optional_field", True) and col_rule not in df_data:
                continue
            if not rule_validate.get("optional_field", True) and col_rule not in df_data:
                err_warn_col = ["Error Validation: Missing field"]*df_data.shape[0]
                error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)
                continue

            err_warn_col = []
            for index, row in df_data.iterrows():

                err_warn_col_row = ""
                if rule_validate.get("empty_is_error", False):
                    # if k_col not in qa_rules:   # #if k_col.lower() not in [k.lower() for k in qa_rules.keys()]
                    # rule_col = qa_rules[k_col]
                    res = validate_empty_is_error(row[col_rule])
                    if any(res):
                        err_warn_col_row = concat_err_warn_col_row(err_warn_col_row, res)
                        err_warn_col.append(err_warn_col_row)
                        continue

                if rule_validate.get("data_type", None):
                    res = validate_data_type(row[col_rule], rule_validate["data_type"].strip().lower())
                    if any(res):
                        err_warn_col_row = concat_err_warn_col_row(err_warn_col_row, res)
                        err_warn_col.append(err_warn_col_row)
                        continue

                # Max, min values
                for validate_rule in ['max_val_error', 'max_val_warning', 'min_val_error', 'min_val_warning']:
                    if rule_validate.get(validate_rule, None):
                        res = globals()["validate_" + validate_rule](row[col_rule], rule_validate[validate_rule])
                        if any(res):
                            err_warn_col_row = concat_err_warn_col_row(err_warn_col_row, res)
                            break

                if "discrete_val" in rule_validate and type(rule_validate['discrete_val']) is list and len(rrule_validate['discrete_val']) > 0:
                    res = validate_discrete_val(row[col_rule].lower(), [item.lower() for item in rule_validate["discrete_val"]])
                    if any(res):
                        err_warn_col_row = concat_err_warn_col_row(err_warn_col_row, res)

                err_warn_col.append(err_warn_col_row)

            error_warning_columns = append_error_warning(error_warning_columns, col_rule, err_warn_col)

    valid_msg['error_warning_columns'] = error_warning_columns
    valid_msg['error_warning_summary'] = summary_reduce_error_warning(error_warning_columns, qa_rules, df_data)
    valid_msg['col_map'] = col_map
    valid_msg['col_names'] = col_names
    valid_msg['col_names_proto'] = col_names_proto
    if schema_ret == 'split':
        valid_msg['data'] = json.loads(df_data.to_json(orient='split'))
    elif schema_ret == 'records':
        valid_msg['data'] = {
            'columns': list(df_data.columns),
            'data': json.loads(df_data.to_json(orient='records'))
        }
    valid_msg['status_code'] = 200

    return valid_msg


def append_error_warning(error_warning_columns, col_rule, err_warn_col):
    if not any(err_warn_col):
        return error_warning_columns

    if col_rule not in error_warning_columns:
        error_warning_columns[col_rule] = err_warn_col
    else:
        error_warning_columns[col_rule] = [x+"; "+y if len(x) > 0 else y for x,y in zip(error_warning_columns[col_rule], err_warn_col)]

    return error_warning_columns


def concat_err_warn_col_row(err_warn_col_row, s):
    if err_warn_col_row:
        if s:
            err_warn_col_row += "; " + s
    else:
        err_warn_col_row = s
    return err_warn_col_row


def summary_reduce_error_warning(error_warning_columns, rules_dct, df_data):
    summary = dict(
        shape = list(df_data.shape),
        cols_valid = [],
        cols_valid_rule = [],
        cols_clean = [],
        cols_clean_rule = [],
        cols_clean_fn = [],
        cols_clean_fn_rule = [],
        cols_clean_lambda = [],
        cols_clean_lambda_rule = [],
        cols_impute = [],
        cols_impute_rule = [],
        cols_impute_fn = [],
        cols_impute_fn_rule = [],
        cols_impute_lambda = [],
        cols_impute_lambda_rule = [],

        cnt_col_valid_err = 0,
        cnt_col_valid_warn = 0,
        cnt_col_clean_err = 0,
        cnt_col_clean_fn_err = 0,
        cnt_col_clean_lambda_err = 0,
        cnt_col_impute_err = 0,
        cnt_col_impute_fn_err = 0,
        cnt_col_impute_lambda_err = 0,

        #tot_clean_columns = len(clean_fn_columns) + len(clean_lambda_columns),
        err_warn_columns = {}
    )

    for col_rule, v in rules_dct.items():
        if "rule_validate" in v:
            summary['cols_valid_rule'].append(col_rule)
            if col_rule in df_data:
                summary['cols_valid'].append(col_rule)
        if "rule_clean" in v:
            summary['cols_clean_rule'].append(col_rule)
            if col_rule in df_data:
                summary['cols_clean'].append(col_rule)
            if "function" in v['rule_clean']:
                summary['cols_clean_fn_rule'].append(col_rule)
                if col_rule in df_data:
                    summary['cols_clean_fn'].append(col_rule)
            if "lambda" in v['rule_clean']:
                summary['cols_clean_lambda_rule'].append(col_rule)
                if col_rule in df_data:
                    summary['cols_clean_lambda'].append(col_rule)
        if "rule_impute" in v:
            summary['cols_impute_rule'].append(col_rule)
            if col_rule in df_data:
                summary['cols_impute'].append(col_rule)
                if "function" in v['rule_impute']:
                    summary['cols_impute_fn_rule'].append(col_rule)
                if "lambda" in v['rule_impute']:
                    summary['cols_impute_lambda_rule'].append(col_rule)

    for col_err_warn, err_warn_col in error_warning_columns.items():
        summary_err_warn_col = {}
        #summary_err_warn_col['cnt_rec_validated'] = sum("Error Validation" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_valid_err'] = sum("Error Validation" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_valid_warn'] = sum("Warning Validation" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_clean_err'] = sum("Error Clean" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_clean_fn_err'] = sum("Error Clean: Function" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_clean_lambda_err'] = sum("Error Clean: Lambda" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_impute_err'] = sum("Error Impute" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_impute_fn_err'] = sum("Error Impute: Function" in s for s in err_warn_col)
        summary_err_warn_col['cnt_rec_impute_lambda_err'] = sum("Error Impute: Lambda" in s for s in err_warn_col)

        summary['err_warn_columns'][col_err_warn] = summary_err_warn_col

        if summary_err_warn_col['cnt_rec_valid_err'] > 0:
            summary['cnt_col_valid_err'] += 1
        if summary_err_warn_col['cnt_rec_valid_warn'] > 0:
            summary['cnt_col_valid_warn'] += 1
        if summary_err_warn_col['cnt_rec_clean_err'] > 0:
            summary['cnt_col_clean_err'] += 1
        if summary_err_warn_col['cnt_rec_clean_fn_err'] > 0:
            summary['cnt_col_clean_fn_err'] += 1
        if summary_err_warn_col['cnt_rec_clean_lambda_err'] > 0:
            summary['cnt_col_clean_lambda_err'] += 1
        if summary_err_warn_col['cnt_rec_impute_err'] > 0:
            summary['cnt_col_impute_err'] += 1
        if summary_err_warn_col['cnt_rec_impute_fn_err'] > 0:
            summary['cnt_col_impute_fn_err'] += 1
        if summary_err_warn_col['cnt_rec_impute_lambda_err'] > 0:
            summary['cnt_col_impute_lambda_err'] += 1

    return summary


def validate_rules_v0(lst_data):
    ''' returns:
    { "error_warning":
      [{"col 1":
           ["", "Error: ..."]
       },
       {"col 2":
           ["Error: missing", "Error: beyond max"]
       },
       {"col 2":
           ["Warning: over", "Error: beyond max"]
       }
      ]
    }

    ? include if no errors or warnings?
    ? alt format
    '''

    error_warning = {}
    valid_msg = {
      #'errors_exist': False,
      #'warnings_exist': False,
      'error_warning_columns': None,
      'status_code': 400
    }

    for col in lst_data:
        for k_col, v_data in col.items():
            # Rules must be checked in this order:
            if k_col not in rules_dct:   # #if k_col.lower() not in [k.lower() for k in rules_dct.keys()]
                continue
            rule_col = rules_dct[k_col]

            err_warn_col = []
            for datum in v_data:
                err, warn = False, False
                if rule_col.get("empty_is_error", False):
                    res = validate_empty_is_error(datum)
                    if any(res):
                        err_warn_col.append(res)
                        if "Error:" in res:
                            err = True
                        elif "Warning:" in res:
                            warn = True
                        continue

                if rule_col.get("data_type", None):
                    res = validate_data_type(datum, rule_col["data_type"].strip().lower())
                    if any(res):
                        err_warn_col.append(res)
                        if "Error:" in res:
                            err = True
                        elif "Warning:" in res:
                            warn = True
                        continue

                # Max, min values
                for validate_rule in ['max_val_error', 'max_val_warning', 'min_val_error', 'min_val_warning']:
                    if rule_col.get(validate_rule, None):
                        res = globals()["validate_" + validate_rule](datum, rule_col[validate_rule])
                        if any(res):
                            err_warn_col.append(res)
                            if "Error:" in res:
                                err = True
                            elif "Warning:" in res:
                                warn = True
                            break

                if "discrete_val" in rule_col and type(rule_col['discrete_val']) is list and len(rule_col['discrete_val']) > 0:
                    res = validate_discrete_val(datum.lower(), [item.lower() for item in rule_col["discrete_val"]])
                    if any(res):
                        err_warn_col.append(res)
                        if "Error:" in res:
                            err = True
                        elif "Warning:" in res:
                            warn = True

                if not err and not warn:
                    err_warn_col.append("")

            if any(err_warn_col):
                error_warning[k_col] = err_warn_col

    # Lastly check missing columns that are required
    lst_fields = [list(_.keys())[0] for _ in lst_data]
    for k,v in rules_dct.items():
        if not v.get("optional_field", True):
            if k not in lst_fields:
                error_warning[k] = ["Error: Missing field "]


    valid_msg['error_warning_columns'] = error_warning
    valid_msg['status_code'] = 200

    return valid_msg
