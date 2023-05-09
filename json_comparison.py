""" Utilities to be used for pre-testing and cleansing of json data
    --isvalid_json - returns if json file is valid
    --clean_json   - outputs a valid json file
    --diff_json    - returns a diff of two json files
    --merge_json   - merges two json files
"""

import os, sys, argparse
import json
import logging
from difflib import SequenceMatcher, unified_diff
from itertools import chain
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd


try:
    from numbers import Number
except ImportError:
    Number = complex, int, eval('long'), float

log = logging.getLogger('datadiff')

string_types = str

class NotHashable(TypeError): pass
class NotSequence(TypeError): pass
class DiffTypeError(TypeError): pass
class DiffNotImplementedForType(DiffTypeError):
    def __init__(self, attempted_type):
        self.attempted_type = attempted_type

    def __str__(self):
        return "diff() not implemented for %s" % self.attempted_type

def unified_diff_strings(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', context=3):
    """
    Wrapper around difflib.unified_diff that accepts 'a' and 'b' as multi-line strings
    and returns a multi-line string, instead of lists of strings.
    """
    return '\n'.join(unified_diff(a.split('\n'), b.split('\n'),
                                  fromfile, tofile, fromfiledate, tofiledate, context,
                                  lineterm=''))

def diff(a, b, context=3, depth=0, fromfile='a', tofile='b'):
    if isinstance(a, string_types) and isinstance(b, string_types):
        # special cases
        if '\n' in a or '\n' in b:
            return unified_diff_strings(a, b, fromfile=fromfile, tofile=tofile, context=context)
        else:
            # even though technically it is a sequence,
            # we don't want to diff char-by-char
            raise DiffNotImplementedForType(str)
    if type(a) != type(b):
        raise DiffTypeError('Types differ: %s=%s %s=%s  Values of a and b are: %r, %r' % (fromfile, tofile, type(a), type(b), a, b))
    if type(a) == dict:
        return diff_dict(a, b, context, depth, fromfile=fromfile, tofile=tofile)
    if hasattr(a, 'intersection') and hasattr(a, 'difference'):
        return diff_set(a, b, context, depth, fromfile=fromfile, tofile=tofile)
    try:
        return try_diff_seq(a, b, context, depth, fromfile=fromfile, tofile=tofile)
    except NotSequence:
        raise DiffNotImplementedForType(type(a))

class DataDiff(object):

    def __init__(self, datatype, type_start_str=None, type_end_str=None, fromfile='a', tofile='b'):
        self.diffs = []
        self.datatype = datatype
        self.fromfile = fromfile
        self.tofile = tofile
        if type_end_str is None:
            if type_start_str is not None:
                raise Exception("Must specify both type_start_str and type_end_str, or neither")
            self.type_start_str = datatype.__name__ + '(['
            self.type_end_str = '])'
        else:
            self.type_start_str = type_start_str
            self.type_end_str = type_end_str

    def context(self, a_start, a_end, b_start, b_end):
        self.diffs.append(('context', [a_start, a_end, b_start, b_end]))

    def context_end_container(self):
        self.diffs.append(('context_end_container', []))

    def nested(self, datadiff):
        self.diffs.append(('datadiff', datadiff))

    def multi(self, change, items):
        self.diffs.append((change, items))

    def delete(self, item):
        return self.multi('delete', [item])

    def insert(self, item):
        return self.multi('insert', [item])

    def equal(self, item):
        return self.multi('equal', [item])

    def insert_multi(self, items):
        return self.multi('insert', items)

    def delete_multi(self, items):
        return self.multi('delete', items)

    def equal_multi(self, items):
        return self.multi('equal', items)

    def __str__(self):
        return self.stringify()

    def stringify(self, depth=0, include_preamble=True):
        if not self.diffs:
            return ''
        output = []
        if depth == 0 and include_preamble:
            output.append('--- %s' % self.fromfile)
            output.append('+++ %s' % self.tofile)
        output.append(' '*depth + self.type_start_str)
        for change, items in self.diffs:
            if change == 'context':
                context_a = str(items[0])
                if items[0] != items[1]:
                    context_a += ',' + str(items[1])
                context_b = str(items[2])
                if items[2] != items[3]:
                    context_b += ',' + str(items[3])
                output.append(' '*depth + '@@ -%s +%s @@' % (context_a, context_b))
                continue
            if change == 'context_end_container':
                output.append(' '*depth + '@@  @@')
                continue
            elif change == 'datadiff':
                output.append(' '*depth + items.stringify(depth+1) + ',')
                continue
            if change == 'delete':
                ch = '-'
            elif change == 'insert':
                ch = '+'
            elif change == 'equal':
                ch = ' '
            else:
                raise Exception('Unknown change type %r' % change)
            for item in items:
                output.append(' '*depth + "%s%r," % (ch, item))
        output.append(' '*depth + self.type_end_str)
        return '\n'.join(output)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        return bool([d for d in self.diffs if d[0] != 'equal'])

def hashable(s):
    try:
        # convert top-level container
        if type(s) == list:
            ret = tuple(s)
        elif type(s) == dict:
            ret = frozenset(hashable(_) for _ in s.items())
        elif type(s) == set:
            ret = frozenset(s)
        else:
            ret = s

        # make it recursive
        if type(ret) == tuple:
            ret = tuple(hashable(_) for _ in ret)

        # validate
        hash(ret)
    except TypeError:
        log.debug('hashable error', exc_info=True)
        raise NotHashable("Hashable type required (for parent diff) but got %s with value %r" % (type(s), s))
    else:
        return ret

def try_diff_seq(a, b, context=3, depth=0, fromfile='a', tofile='b'):
    """
    Safe to try any containers with this function, to see if it might be a sequence
    Raises TypeError if its not a sequence
    """
    try:
        return diff_seq(a, b, context, depth, fromfile=fromfile, tofile=tofile)
    except NotHashable:
        raise
    except:
        log.debug('tried SequenceMatcher but got error', exc_info=True)
        raise NotSequence("Cannot use SequenceMatcher on %s" % type(a))

def diff_seq(a, b, context=3, depth=0, fromfile='a', tofile='b'):
    if not hasattr(a, '__iter__') and not hasattr(a, '__getitem__'):
        raise NotSequence("Not a sequence %s" % type(a))
    hashable_a = [hashable(_) for _ in a]
    hashable_b = [hashable(_) for _ in b]
    sm = SequenceMatcher(a = hashable_a, b = hashable_b)
    if type(a) == tuple:
        ddiff = DataDiff(tuple, '(', ')', fromfile=fromfile, tofile=tofile)
    elif type(b) == list:
        ddiff = DataDiff(list, '[', ']', fromfile=fromfile, tofile=tofile)
    else:
        ddiff = DataDiff(type(a), fromfile=fromfile, tofile=tofile)
    for chunk in sm.get_grouped_opcodes(context):
        ddiff.context(max(chunk[0][1]-1,0), max(chunk[-1][2]-1, 0),
                     max(chunk[0][3]-1,0), max(chunk[-1][4]-1, 0))
        for change, i1, i2, j1, j2 in chunk:
            if change == 'replace':
                consecutive_deletes = []
                consecutive_inserts = []
                for a2, b2 in zip(a[i1:i2], b[j1:j2]):
                    try:
                        nested_diff = diff(a2, b2, context, depth+1)
                        ddiff.delete_multi(consecutive_deletes)
                        ddiff.insert_multi(consecutive_inserts)
                        consecutive_deletes = []
                        consecutive_inserts = []
                        ddiff.nested(nested_diff)
                    except DiffTypeError:
                        consecutive_deletes.append(a2)
                        consecutive_inserts.append(b2)

                # differing lengths get truncated by zip()
                # here we handle the truncated items
                ddiff.delete_multi(consecutive_deletes)
                if i2-i1 > j2-j1:
                    common_length = j2-j1 # covered by zip
                    ddiff.delete_multi(a[i1+common_length:i2])
                ddiff.insert_multi(consecutive_inserts)
                if i2-i1 < j2-j1:
                    common_length = i2-i1 # covered by zip
                    ddiff.insert_multi(b[j1+common_length:j2])
            else:
                if change == 'insert':
                    items = b[j1:j2]
                else:
                    items = a[i1:i2]
                ddiff.multi(change, items)
        if i2 < len(a):
            ddiff.context_end_container()
    return ddiff


class dictitem(tuple):
    def __repr__(self):
        key, val = self
        if type(val) == DataDiff:
            diff_val = val.stringify(depth=self.depth, include_preamble=False)
            return "%r: %s" % (key, diff_val.strip())
        return "%r: %r" % (key, val)

def diff_dict(a, b, context=3, depth=0, fromfile='a', tofile='b'):
    ddiff = DataDiff(dict, '{', '}', fromfile=fromfile, tofile=tofile)
    for key in a.keys():
        if key not in b:
            ddiff.delete(dictitem((key, a[key])))
        elif a[key] != b[key]:
            try:
                nested_diff = diff(a[key], b[key], context, depth+1)
                nested_item = dictitem((key, nested_diff))
                nested_item.depth = depth+1
                ddiff.equal(nested_item) # not really equal
            except DiffTypeError:
                ddiff.delete(dictitem((key, a[key])))
                ddiff.insert(dictitem((key, b[key])))
        else:
            if context:
                ddiff.equal(dictitem((key, a[key])))
            context -= 1
    for key in b:
        if key not in a:
            ddiff.insert(dictitem((key, b[key])))

    def diffitem_dictitem_sort_key(diffitem):
        change, dictitem = diffitem
        if type(dictitem) == DataDiff:
            return 0
        key = dictitem[0][0]
        # use hash, to make sure its always orderable against other potential key types
        if isinstance(key, string_types) or isinstance(key, Number):
            return key
        else:
            return abs(hash(key)) # abs for consistency between py2/3, at least for datetime
    ddiff.diffs.sort(key=diffitem_dictitem_sort_key)

    if context < 0:
        ddiff.context_end_container()

    return ddiff

def diff_set(a, b, context=3, depth=0, fromfile='b', tofile='a'):
    ddiff = DataDiff(type(a), fromfile=fromfile, tofile=tofile)
    ddiff.delete_multi(a - b)
    ddiff.insert_multi(b - a)
    equal = list(a.intersection(b))
    ddiff.equal_multi(equal[:context])
    if len(equal) > context:
        ddiff.context_end_container()
    return ddiff


def get_json_file(pth_fname):
    if not os.path.isfile(pth_fname):
        return f"ERROR: {pth_fname} not found"

    with open(pth_fname) as fin:
        try:
            dct = json.load(fin)
            return dct
        except ValueError as e:
            return f"Error: {e}"


def get_df_file(pth, fname):
    pth_fname = os.path.abspath(os.path.join(pth, fname))
    if fname[-4:] == '.csv':
        return get_df_csv(pth_fname)
    elif fname[-5:] == '.xlsx':
        return get_df_excel(pth_fname)

def get_df_excel(pth_fname):
    if not os.path.isfile(pth_fname):
        return f"ERROR: {pth_fname} not found"
    return pd.read_excel(pth_fname)

def get_df_csv(pth_fname):
    if not os.path.isfile(pth_fname):
        return f"ERROR: {pth_fname} not found"
    return pd.read_csv(pth_fname)

def get_df_json(jsn, orient='split'):
    if type(jsn) is dict:
        jsn = json.dumps(jsn)
    return pd.read_json(jsn, orient=orient)


def clean_json(jsn_filename_in):
    """
    Clean non-valid json file to valid json file

    Notes:
        Writes a validated json file with '_valid' suffix added to filename
        Currently designed to run outside of runtime of main application

    Args:
        json filename (str)

    """
    name, ext = jsn_filename_in.split('.')
    jsn_filename_out = name + '_valid.' + ext

    try:
        with open(jsn_filename_in, 'r') as fin, open(jsn_filename_out, "w") as fout:
            jsn_txt = fin.read()
            jsn_txt = jsn_txt.replace('\n','')
            dct = json.loads(jsn_txt)
            json.dump(dct , fout, indent=4)

    except IOError as e:
        return f"Error: {e}"


def diff_json(jsn0, jsn1):
    print(diff(jsn0, jsn1))


def merge_json(jsn_orig, jsn_new, merged_filename="variable_specs.json"):
    """
    Merges two json files

    Notes:
        Not a simple merge, rule logic is:
        - if new file has key or values that exist, this overrides original
        - for aliases list, does a unique union

    Args:
        Two json filenames (str)

    """

    try:
        with open(jsn_orig, 'r') as fin0, open(jsn_new, 'r') as fin1, open(merged_filename, "w") as fout:
            jsn_txt0 = fin0.read()
            jsn_txt1 = fin1.read()
            dct0 = json.loads(jsn_txt0)
            dct1 = json.loads(jsn_txt1)

            # Can't do simple merge - neither of these will work:
            # dct_merged = {key: value for (key, value) in (list(jsn_orig.items()) + list(jsn_new.items()))}
            #for k in dict0:
            #    dict0[k].update(dict1.get(k, {})) # dict1.get(k).update(dict2.get(k, {}))

            merged_dct = OrderedDict({})
            for k, v in OrderedDict(dct0).items():
                if k in dct1:
                    # Treat aliases - keep order but remove duplicates
                    # v['aliases'] = list(np.unique(v['aliases'] + dct1[k]['aliases']))
                    aliases = list(v['aliases'])
                    aliases.extend(x for x in dct1[k]['aliases'] if x not in aliases)
                    v['aliases'] = aliases
                    for k_, v_ in OrderedDict(v).items():
                        if k_ != 'aliases' and k_ in dct1[k]:
                            v[k_] = dct1[k][k_]

                merged_dct[k] = v

            for k, v in OrderedDict(dct1).items():
                if k not in dct0:
                    merged_dct[k] = v
                else:
                    for k_, v_ in OrderedDict(v).items():
                        if k_ != 'aliases':
                            merged_dct[k][k_] = v_

            json.dump(merged_dct, fout, indent=4)

    except IOError as e:
        return f"Error: {e}"


def convert_json_row_col(jsn_row):
    jsn_col = {

    }

    return jsn_col


def jprint(d):
    print(json.dumps(d, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--isvalid_json', nargs='+')
    parser.add_argument('--clean_json', nargs='+')
    parser.add_argument('--diff_json', nargs=2)
    parser.add_argument('--merge_json', nargs=2)

    args = parser.parse_args()

    if args.clean_json:
        for jsn in args.clean_json:
            clean_json(jsn)


    elif args.isvalid_json:
        for jsn in args.isvalid_json:
            res = get_json_file(jsn)
            if type(res) != dict:
                print(jsn + " : " + res)
            else:
                print(f"{jsn} is valid json")


    elif args.merge_json:
        bl_valid = True
        for jsn in args.merge_json:
            dct = get_json_file(jsn)
            if type(dct) != dict:
                print(jsn + " : " + dct)
                bl_valid = False

        if bl_valid:
            merge_json(args.merge_json[0], args.merge_json[1])


    elif args.diff_json:
        bl_allvalid = True
        dct_lst = []
        for jsn in args.diff_json:
            dct = get_json_file(jsn)
            if type(dct) != dict:
                bl_allvalid = False
                print(dct)
            else:
                dct_lst.append(dct)

        if bl_allvalid and len(args.diff_json) == 2:
            diff_json(dct_lst[0], dct_lst[1])
