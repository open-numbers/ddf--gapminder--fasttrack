# -*- coding: utf-8 -*-

import os
import os.path as osp
import pandas as pd
import numpy as np
import parse

from decimal import Decimal
from ddf_utils.factory.common import retry

from gspread.exceptions import APIError
from gspread_pandas import Spread
from gspread_pandas.conf import get_config_dir


# fasttrack doc id
# for the democracy branch we used another sheet: "1qIWmEYd58lndW-KLk8ouDakgyYGSp4nEn2QQaLPXmhI"
DOCID = "1P1KQ8JHxjy8wnV02Hwb1TnUEJ3BejMbMKbQ0i_VAjyo"
SOURCE_DIR = '../source/'


# define 3 exceptions for error handling
class EmptySheet(Exception):
    pass


class EmptyColumn(Exception):
    pass


class EmptyCell(Exception):
    pass


def get_docid_sheet(link):
    p = parse.parse(
        "https://docs.google.com/spreadsheets/d/{docid}/gviz/tq?tqx=out:csv&sheet={sheet_name}",
        link)
    docid = p.named['docid']
    sheet_name = p.named['sheet_name']

    return docid, sheet_name


def get_csv_link_dict(ser):
    res = dict()
    for s in ser:
        docid, sheet_name = get_docid_sheet(s)
        if docid in res:
            if sheet_name in res[docid]:
                continue
            else:
                res[docid][sheet_name] = s
        else:
            res[docid] = dict()
            res[docid][sheet_name] = s
    return res


def find_column(df, dimension_pair):
    for d in dimension_pair[::-1]:  # reversed order: from specific to general
        if d in df.columns:
            return d


def parse_number(s, decimal=False):
    # TODO: maybe use locale module to handle different formats.
    # see https://stackoverflow.com/a/46411203
    if len(s) == 0:
        return np.nan
    tbl = str.maketrans('(', '-', '),%')
    if decimal:
        return Decimal(s.translate(tbl))
    return float(s.translate(tbl))


def parse_dimension_pairs(dimensions):
    return [p.fixed for p in parse.findall("{:w}:{:w}", dimensions)]


@retry(times=10, backoff=10, exceptions=(EmptyColumn, EmptySheet, EmptyCell, APIError))
def read_sheet(doc: Spread, sheet_name):
    df = doc.sheet_to_df(sheet=sheet_name, index=None)
    # detect error in sheet
    if df.empty:
        raise EmptySheet(f"{sheet_name} is empty")
    elif df.shape[0] == 1 and df.iloc[0, 0] in ['#N/A', '#VALUE!', 0]:
        msg = f"{sheet_name} contains all NA values"
        raise EmptyColumn(f"{sheet_name} contains all NA values")
    elif len(df['geo'].unique()) == 1 and 'world' not in df['geo'].values:
        msg = f"{sheet_name}, geo column contains NA values"
        raise EmptyColumn(msg)
    else:
        for c in df.columns:
            if df[c].hasnans or '[Invalid]' in df[c].values:
                msg = f'{sheet_name}, column {c} has NA values'
                raise EmptyCell(msg)
    print(df.head())
    return df


def get_docid_sheet_from_file(fname):
    res = fname.split('.')[0].split(' - ')
    return res[0], res[1]


def download(doc, sheet_name, outfile):
    try:
        # read_sheet will retry a few times on some errors.
        # check its definition for details.
        df = read_sheet(doc, sheet_name)
    except Exception as e:
        print(f"error: {e}")
    df.to_csv(outfile, encoding='utf8')


def process(row: pd.Series, env: dict):
    """process one row in the datapoints table"""
    flag = row['Status']
    concept_id = row['concept_id']
    dimensions = row['dimensions']
    docid, sheet_name = get_docid_sheet(row['csv_link'])
    filename = f'{docid}-{sheet_name}.csv'
    filename_full = osp.join(SOURCE_DIR, 'datapoints', filename)
    if flag in ['S', 's']:  # s for skip
        print(f'skipped: {concept_id}, {dimensions}')
        return
    if flag in ['U', 'u']:  # u for update
        if filename in env['downloaded']:  # already downloaded when processing other row
            return
        print(f"Downloading from sheet {docid} - {sheet_name}")
        doc = env['datapoint_docs'].setdefault(docid, Spread(spread=docid))
        download(doc, sheet_name, filename_full)
        env['downloaded'].add(filename)
    # if not flag and not osp.exists(filename_full):  # no flags but the file not exists
    #     print(f"Downloading from sheet {docid} - {sheet_name} (file missing in source dir)")
    #     doc = env['datapoint_docs'].setdefault(docid, Spread(spread=docid))
    #     download(doc, sheet_name, filename_full)
    #     env['downloaded'].add(filename)


def main():
    print('loading source files...')
    main_doc = Spread(spread=DOCID)
    datapoints = main_doc.sheet_to_df(sheet='datapoints', index=None)
    topics = main_doc.sheet_to_df(sheet='topics', index=None)
    concepts = main_doc.sheet_to_df(sheet='concepts', index=None)
    datapoint_docs = dict()   # we can reuse the spreadsheet object to download multiple sheets
    downloaded = set()
    env = {
        'downloaded': downloaded,
        'datapoint_docs': datapoint_docs
    }
    print('saving datapoints into etl/source/datapoints...')
    for _, row in datapoints.iterrows():
        process(row, env)

    datapoints.to_csv(osp.join(SOURCE_DIR, 'datapoints.csv'), index=False)
    topics.to_csv(osp.join(SOURCE_DIR, 'topics.csv'), index=False)
    concepts.to_csv(osp.join(SOURCE_DIR, 'concepts.csv'), index=False)


if __name__ == '__main__':
    print(get_config_dir())
    main()
    print('Done.')
