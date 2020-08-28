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
from urllib.error import HTTPError

from gspread_pandas import Spread
from gspread_pandas.conf import get_config_dir


# fasttrack doc id
DOCID =  "1P1KQ8JHxjy8wnV02Hwb1TnUEJ3BejMbMKbQ0i_VAjyo" # for the democracy branch we used another sheet: "1qIWmEYd58lndW-KLk8ouDakgyYGSp4nEn2QQaLPXmhI"


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


def serve_datapoints(datapoints, concepts, csv_dict):

    # map concept_name -> concept_id
    concept_map = datapoints.set_index('concept_name')['concept_id'].to_dict()

    # translate plural form to singal form
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
        df.dropna().to_csv('../../ddf--datapoints--{}--by--{}.csv'.format(row['concept_id'], '--'.join(by_fn)), encoding='utf8')


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


def main():
    print('loading source files...')
    main_doc = Spread(spread=DOCID)
    concepts = main_doc.sheet_to_df(sheet='concepts', index=None)
    datapoints = main_doc.sheet_to_df(sheet='datapoints', index=None)
    tags = main_doc.sheet_to_df(sheet='topics', index=None)

    # construct a dictionary, keys are docids, values are dictionaries which
    # keys are sheet names and values are the csv links for the docid/sheet name pair.
    csv_link_dict = get_csv_link_dict(datapoints.csv_link.values)

    # create a dictionary that has same layout of `csv_link_dict` but the values are
    # dataframes, instead of links
    csv_dict = csv_link_dict.copy()
    for docid, di in csv_link_dict.items():
        print(f"Downloading sheets from file: {docid}")
        csv_dict[docid] = dict()
        doc = Spread(spread=docid)
        for sheet_name, link in di.items():
            print(f"sheet: {sheet_name}")
            df = doc.sheet_to_df(sheet=sheet_name, index=None)
            if df.empty:
                print(f"WARNING: empty dataframe: doc: {docid}, sheet_name: {sheet_name}")
            csv_dict[docid][sheet_name] = df

    print('creating ddf datasets...')

    concepts_ontology = pd.read_csv('../source/ddf--open_numbers/ddf--concepts.csv')
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
    tags = tags.rename(columns={'topic_id': 'tag', 'topic_name': 'name', 'parent_topic': 'parent' })
    tags.to_csv('../../ddf--entities--tag.csv', index=False, encoding='utf8')
    for c in tags.columns:
        entities_columns.add(c)

    # concepts
    cdf = serve_concepts(concepts, entities_columns)

    # datapoints
    serve_datapoints(datapoints, cdf, csv_dict)


if __name__ == '__main__':
    print(get_config_dir())
    main()
    print('Done.')
