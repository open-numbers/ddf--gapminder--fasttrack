# -*- coding: utf-8 -*-

import os.path as osp
import requests as req
import pandas as pd
import numpy as np
import parse

from io import BytesIO
from decimal import Decimal
from ddf_utils.str import format_float_digits

# TODO: Some of below functions should be moved in ddf_utils.

# fasttrack doc id
DOCID = "1P1KQ8JHxjy8wnV02Hwb1TnUEJ3BejMbMKbQ0i_VAjyo"


def open_google_spreadsheet(docid):
    tmpl_xls = "https://docs.google.com/spreadsheets/d/{docid}/export?format=xlsx&id={docid}"
    url = tmpl_xls.format(docid=docid)
    res = req.get(url)
    if res.ok:
        return BytesIO(res.content)
    return None


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
    tbl = str.maketrans('(', '-', '),%')
    if decimal:
        return Decimal(s.translate(tbl))
    return float(s.translate(tbl))


def parse_dimension_pairs(dimensions):
    return [p.fixed for p in parse.findall("{:w}:{:w}", dimensions)]


def serve_datapoints(datapoints, concept_map, csv_dict):

    def get_dataframe(docid, sheet_name, dimension_pairs, concept_name, copy=True):
        df = csv_dict[docid][sheet_name]
        columns = [find_column(df, x) for x in dimension_pairs]
        columns.append(concept_name)
        if copy:
            return df[columns].copy()
        else:
            return df[columns]

    for _, row in datapoints.iterrows():
        dimension_pairs = parse_dimension_pairs(row['dimensions'])
        docid, sheet_name = get_docid_sheet(row['csv_link'])
        df = get_dataframe(docid, sheet_name, dimension_pairs, row['concept_name'])
        by = [find_column(df, x) for x in dimension_pairs]
        df = df.set_index(by)
        # print(df.columns)
        df = df.rename(columns=concept_map)
        concept = df.columns[0]
        if df[concept].dtype == 'object':
            df[concept] = df[concept].map(parse_number).map(format_float_digits)
        else:
            df[concept] = df[concept].map(format_float_digits)
        by_fn = list()
        for k, v in dict(dimension_pairs).items():
            if k == 'time':
                by_fn.append('time')
            else:
                by_fn.append(v)
        df.index.names = by_fn
        df.dropna().to_csv('../../ddf--datapoints--{}--by--{}.csv'.format(row['concept_id'], '--'.join(by_fn)))


def serve_concepts(concepts, entities_columns):
    concepts_geo = pd.read_csv('../source/ddf--gapminder--geo_entity_domain/ddf--concepts.csv')
    concepts_ontology = pd.read_csv('../source/ddf--gapminder--ontology/ddf--concepts--discrete.csv')

    # first, concepts from google spreadsheet
    cdf1 = concepts.copy()
    cdf1 = cdf1.rename(columns={'concept_id': 'concept'})
    cdf1 = cdf1.set_index('concept')

    # second, entity concepts
    geo_concepts = ['geo', 'country', 'world_4region', 'global', 'domain', 'drill_up']
    cdf2 = concepts_geo[concepts_geo.concept.isin(geo_concepts)].copy()
    cdf2 = cdf2.set_index('concept')

    # third, concepts in entity columns
    cdf3 = concepts_geo[concepts_geo.concept.isin(entities_columns)].copy()
    cdf3 = cdf3.set_index('concept')

    # concepts that are no in the ontology
    cdf4 = pd.DataFrame([['time', 'Time', 'time'],
                         ['version', 'Version', 'string'],
                         ['updated', 'Updated', 'string'],
                         ['topic', 'Topic', 'string'],
                         ['unit', 'Unit', 'string']], columns=['concept', 'name', 'concept_type'])
    cdf4 = cdf4.set_index('concept')

    # combining above concepts
    cdf_full = pd.concat([cdf1, cdf2, cdf3, cdf4], sort=False)

    # check all columns and see if it's in ontology. Use ontology if possible
    cdf5 = concepts_ontology[concepts_ontology.concept.isin(cdf_full.columns)]
    cdf5 = cdf5.set_index('concept')
    cdf_full = pd.concat([cdf_full, cdf5], sort=False)

    # removing duplications
    cdf_full = cdf_full.reset_index().drop_duplicates(subset=['concept'], keep='last')
    cdf_full.to_csv('../../ddf--concepts.csv', index=False)


def main():
    print('loading source files...')
    fo = open_google_spreadsheet(DOCID)
    concepts = pd.read_excel(fo, sheet_name='concepts')
    datapoints = pd.read_excel(fo, sheet_name='datapoints')

    # construct a dictionary, keys are docids, values are dictionaries which
    # keys are sheet names and values are the csv links for the docid/sheet name pair.
    csv_link_dict = get_csv_link_dict(datapoints.csv_link.values)

    # create a dictionary that has same layout of `csv_link_dict` but the values are
    # dataframes, instead of links
    csv_dict = csv_link_dict.copy()
    for docid, di in csv_link_dict.items():
        csv_dict[docid] = dict([sheet_name, pd.read_csv(link)]
                               for sheet_name, link in csv_dict[docid].items())

    print('creating ddf datasets...')

    # map concept_name -> concept_id
    concept_map = datapoints.set_index('concept_name')['concept_id'].to_dict()

    # datapoints
    serve_datapoints(datapoints, concept_map, csv_dict)

    # entities
    entities_columns = set()  # mark down the columns, use to create concept table later
    for e in ['country', 'global', 'world_4region']:
        edf = pd.read_csv(f'../source/ddf--gapminder--geo_entity_domain/ddf--entities--geo--{e}.csv',
                          na_filter=False, dtype=str)
        edf.to_csv(f'../../ddf--entities--geo--{e}.csv', index=False)
        for c in edf.columns:
            entities_columns.add(c)

    # concepts
    serve_concepts(concepts, entities_columns)


if __name__ == '__main__':
    main()
    print('Done.')
