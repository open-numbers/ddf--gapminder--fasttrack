# -*- coding: utf-8 -*-

import os.path as osp
import requests as req
import pandas as pd
import numpy as np
import parse
import re
import time

from io import BytesIO
from decimal import Decimal
from ddf_utils.str import format_float_digits
from ddf_utils.factory.common import retry
from urllib.error import HTTPError

from gspread.exceptions import APIError
from gspread_pandas import Spread
from gspread_pandas.conf import get_config_dir


SOURCE_DIR = '../source'

# define 3 exceptions for error handling
class EmptySheet(Exception):
    pass


class EmptyColumn(Exception):
    pass


class EmptyCell(Exception):
    pass


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


def serve_datapoints(datapoints, concepts, csv_dict):

    # map concept_name -> concept_id
    concept_map = datapoints.set_index('concept_name')['concept_id'].to_dict()

    # dictionary for translating plural form to singal form
    translate_dict = {'countries': 'country', 'world_4regions': 'world_4region', 'regions': 'world_4region'}

    def get_dataframe(docid, sheet_name, dimension_pairs, concept_name, copy=True):
        df = csv_dict[docid][sheet_name]
        # do some cleanups
        df = df.dropna(axis=0, how='all')
        df.columns = df.columns.map(lambda x: x.replace('#N/A', '').strip())

        columns = [find_column(df, x) for x in dimension_pairs]
        columns.append(concept_name)
        try:
            if copy:
                return df[columns].copy()
            else:
                return df[columns]
        except KeyError:
            print("column not found!\n"
                  "expected columns: {}\n"
                  "available columns: {}".format(columns, list(df.columns)))
            raise KeyError("Key not found.")

    for _, row in datapoints.iterrows():
        dimension_pairs = parse_dimension_pairs(row['dimensions'])
        docid, sheet_name = get_docid_sheet(row['csv_link'])
        print("working on file {}, sheet {}".format(docid, sheet_name))
        df = get_dataframe(docid, sheet_name, dimension_pairs, row['concept_name'])
        by = [find_column(df, x) for x in dimension_pairs]

        df = df.set_index(by)
        # print(df.columns)
        df = df.rename(columns=concept_map)
        concept = df.columns[0]
        # test if the concept exists in concepts table
        if concept not in concepts['concept'].values:
            raise ValueError(f'the concept {concept} not found in concepts table! Please double check'
                             'both the main file and data file.')
        if df[concept].dtype == 'object':  # didn't reconized as number
            concept_type = concepts.loc[concepts['concept'] == concept, 'concept_type'].iloc[0]
            if concept_type == 'measure':  # it should be numbers
                try:
                    df[concept] = df[concept].map(parse_number).map(format_float_digits)
                except (AttributeError, ValueError):
                    print(f"can't convert the column {concept} to numbers. Maybe it contains non-numeric values?")
                    raise
            else:
                df[concept] = df[concept].map(lambda v: v.strip())
        else:
            df[concept] = df[concept].map(format_float_digits)
        by_fn = list()
        for k, v in dict(dimension_pairs).items():
            if k == 'time':
                by_fn.append('time')
            else:
                by_fn.append(v)
        by_fn = [translate_dict.get(x, x) for x in by_fn]
        df.index.names = by_fn
        df.dropna().sort_index().to_csv(
            '../../ddf--datapoints--{}--by--{}.csv'.format(row['concept_id'], '--'.join(by_fn)),
            encoding='utf8')


def serve_concepts(concepts, entities_columns):
    concepts_ontology = pd.read_csv('../source/ddf--open_numbers/ddf--concepts.csv')

    # first, concepts from google spreadsheet
    cdf1 = concepts.copy()
    cdf1 = cdf1.rename(columns={'concept_id': 'concept', 'topic': 'tags'})
    cdf1 = cdf1.set_index('concept')
    # trim descriptions
    # cdf1['description'] = cdf1['description'].map(lambda v: re.sub(r'\s+', ' ', v).strip())

    # second, entity concepts
    geo_predicate = (concepts_ontology.concept == 'geo') | (concepts_ontology.domain == 'geo')
    cdf2 = concepts_ontology[geo_predicate].copy()
    cdf2 = cdf2.set_index('concept')

    # third, concepts in entity columns
    cdf3 = concepts_ontology[concepts_ontology.concept.isin(entities_columns)].copy()
    cdf3 = cdf3.set_index('concept')

    # also check them in ontology
    cdf4 = concepts_ontology[concepts_ontology.concept.isin(entities_columns)].copy()
    cdf4 = cdf4.set_index('concept')

    # concepts that are no in the ontology
    cdf5 = pd.DataFrame([['time', 'Time', 'time'],
                         ['version', 'Version', 'string'],
                         ['updated', 'Updated', 'string'],
                         ['unit', 'Unit', 'string']], columns=['concept', 'name', 'concept_type'])
    cdf5 = cdf5.set_index('concept')

    # import ipdb; ipdb.set_trace()
    # combining above concepts
    cdf_full = pd.concat([cdf1, cdf2, cdf3, cdf4, cdf5], sort=False)

    # check all columns and see if it's in ontology. Use ontology if possible
    cdf6 = concepts_ontology[concepts_ontology.concept.isin(cdf_full.columns)]
    cdf6 = cdf6.set_index('concept')
    cdf_full = pd.concat([cdf_full, cdf6], sort=False)

    # removing duplications
    cdf_full = cdf_full.reset_index().dropna(how='all').drop_duplicates(subset=['concept'], keep='last')
    cdf_full.to_csv('../../ddf--concepts.csv', index=False, encoding='utf8')

    return cdf_full


def get_docid_sheet(link):
    p = parse.parse(
        "https://docs.google.com/spreadsheets/d/{docid}/gviz/tq?tqx=out:csv&sheet={sheet_name}",
        link)
    docid = p.named['docid']
    sheet_name = p.named['sheet_name']

    return docid, sheet_name


def main():
    print('loading source files...')
    concepts = pd.read_csv(osp.join(SOURCE_DIR, 'concepts.csv'), dtype=str)
    datapoints = pd.read_csv(osp.join(SOURCE_DIR, 'datapoints.csv'), dtype=str)
    tags = pd.read_csv(osp.join(SOURCE_DIR, 'topics.csv'), dtype=str)
    concepts_ontology = pd.read_csv(osp.join(SOURCE_DIR, 'ddf--open_numbers/ddf--concepts.csv', dtype=str))

    print('creating ddf datasets...')
    # entities
    entities_columns = set()  # mark down the columns, use to create concept table later
    geo_concepts = concepts_ontology[concepts_ontology.domain == 'geo'].concept.values
    for e in geo_concepts:
        file_path = f'../source/ddf--open_numbers/ddf--entities--geo--{e}.csv'
        if osp.exists(file_path):
            edf = pd.read_csv(f'../source/ddf--open_numbers/ddf--entities--geo--{e}.csv',
                              na_filter=False, dtype=str)
            edf.to_csv(f'../../ddf--entities--geo--{e}.csv', index=False, encoding='utf8')
            for c in edf.columns:
                entities_columns.add(c)
        else:
            print(f'WARNING: file not found: {file_path}, skipping')

    # tags entities
    tags = tags.rename(columns={'topic_id': 'tag', 'topic_name': 'name', 'parent_topic': 'parent'})
    tags.to_csv('../../ddf--entities--tag.csv', index=False, encoding='utf8')
    for c in tags.columns:
        entities_columns.add(c)

    # concepts
    cdf = serve_concepts(concepts, entities_columns)

    # datapoints
    datapoint_dfs = dict()
    for _, row in datapoints.iterrows():
        docid, sheet_name = get_docid_sheet(row['csv_link'])
        key = f'{docid}-{sheet_name}'
        filename_full = osp.join(SOURCE_DIR, 'datapoints', f'{key}.csv')
        df = datapoint_dfs.setdefault(key, pd.read_csv(filename_full, dtype=str))
        # TODO: continue working
    serve_datapoints(datapoints, cdf, csv_dict)


if __name__ == '__main__':
    print(get_config_dir())
    main()
    print('Done.')
