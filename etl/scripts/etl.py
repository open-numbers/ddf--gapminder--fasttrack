# -*- coding: utf-8 -*-

import os
import os.path as osp
import pandas as pd
import numpy as np
import parse

from decimal import Decimal
from ddf_utils.str import format_float_digits
from gspread_pandas.conf import get_config_dir


SOURCE_DIR = '../source'


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


def serve_concepts(concepts, entities_columns):
    concepts_ontology = pd.read_csv('../source/ddf--open_numbers/ddf--concepts.csv')

    # first, concepts from google spreadsheet
    cdf1 = concepts.copy()
    # check duplicates. This spreadsheet should not have duplicated concepts.
    dups1 = cdf1[cdf1['concept'].duplicated()]
    if not dups1.empty:
        print("ERROR: the concept sheet has duplicated entries:")
        print(dups1['concept'].values)
        raise ValueError("duplicated concepts in concept sheet.")
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


def csv_link_to_filename(link):
    docid, sheet_name = get_docid_sheet(link)
    return f'{docid}-{sheet_name}'


def load_file_preprocess(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.map(lambda x: x.replace('#N/A', '').strip())
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    return df


def process_datapoints(row, env):
    concept_id = row['concept_id']
    concept_name = row['concept_name'].strip()
    dimension = row['dimensions'].strip()
    table_format = row['table_format'].strip()
    csv_link = row['csv_link'].strip()
    status = row['Status']
    dimension_pairs = parse_dimension_pairs(dimension)
    datapoints_prev = env['datapoints_prev'].set_index(['concept_id', 'dimensions'])
    if status in ['s', 'S']:
        print(f'skipped: {concept_name}, {dimension_pairs}')
        return
    if pd.isnull(status) and (concept_id, dimension) not in datapoints_prev.index:
        print(f'{concept_name} does not have old version and no `u` flag. skipping')
        return
    print(f'processing {concept_name}, {dimension_pairs}')
    datapoint_dfs = env['datapoint_dfs']
    concepts = env['concepts']
    translate_dict = env['translate_dict']
    concept_map = env['concept_map']
    if status in ['u', 'U']:
        fn = csv_link_to_filename(csv_link)
    else:
        fn = datapoints_prev.loc[(concept_id, dimension), 'filename']
    filename_full = osp.join(SOURCE_DIR, 'datapoints', f'{fn}.csv')
    if not osp.exists(filename_full):
        print(f'no source for {concept_name}, {dimension_pairs}, assumming it is still wip and skipping')
        return
    env['datapoint_and_doc_list'].append((concept_id, concept_name, dimension, table_format, fn))
    df = datapoint_dfs.setdefault(fn, load_file_preprocess(filename_full))
    by = [find_column(df, x) for x in dimension_pairs]
    if by == [None]:
        print(f"couldn't find column {dimension_pairs} in the spreadsheet")
        raise ValueError("couldn't find key columns")
    columns = by.copy()
    columns.append(concept_name)
    df = df[columns].dropna(how='any').set_index(by)
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
            except (AttributeError, ValueError, TypeError):
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
    if 'country' in by_fn:
        outdir = "countries_etc_datapoints"
    else:
        outdir = "global_regions_datapoints"
    df.dropna().sort_index().to_csv(
        osp.join('../../',
                 outdir,
                 'ddf--datapoints--{}--by--{}.csv'.format(
                     row['concept_id'].strip(), '--'.join(by_fn))),
        encoding='utf8')


def main():
    print('loading source files...')
    concepts = pd.read_csv(osp.join(SOURCE_DIR, 'concepts.csv'), dtype=str)
    datapoints = pd.read_csv(osp.join(SOURCE_DIR, 'datapoints.csv'), dtype=str)
    tags = pd.read_csv(osp.join(SOURCE_DIR, 'topics.csv'), dtype=str)
    concepts_ontology = pd.read_csv(osp.join(SOURCE_DIR, 'ddf--open_numbers/ddf--concepts.csv'), dtype=str)

    print('creating ddf datasets...')
    # entities
    entities_columns = set()  # mark down the columns, use to create concept table later
    geo_concepts = concepts_ontology[concepts_ontology.domain == 'geo'].concept.values.tolist()

    geo_concepts.append('')  # there are some geos not belonging all groups

    from_dir = '../source/ddf--open_numbers/'
    to_dir = '../../'
    for e in geo_concepts:
        if e == '':
            basename = 'ddf--entities--geo.csv'
        else:
            basename = f'ddf--entities--geo--{e}.csv'
        file_path = os.path.join(from_dir, basename)
        if osp.exists(file_path):
            edf = pd.read_csv(file_path, na_filter=False, dtype=str)
            edf.to_csv(os.path.join(to_dir, basename), index=False, encoding='utf8')
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
    # create datapoints folders if not exist
    os.makedirs(osp.join("../../", "countries_etc_datapoints"), exist_ok=True)
    os.makedirs(osp.join("../../", "global_regions_datapoints"), exist_ok=True)
    # map concept_name -> concept_id
    concept_map = datapoints.set_index('concept_name')['concept_id'].to_dict()
    # dictionary for translating plural form to singal form
    translate_dict = {'countries': 'country', 'world_4regions': 'world_4region', 'regions': 'world_4region'}
    datapoints_prev = pd.read_csv(osp.join(SOURCE_DIR, 'datapoints.cache.csv'), dtype=str)
    env = {
        'datapoints_prev': datapoints_prev,
        'datapoint_dfs': dict(),
        'concept_map': concept_map,
        'translate_dict': translate_dict,
        'concepts': cdf,
        'datapoint_and_doc_list': list()  # a list of datapoints and the linked google docs, for caching
    }
    for _, row in datapoints.iterrows():
        process_datapoints(row, env)

    datapoint_and_doc = pd.DataFrame.from_records(env['datapoint_and_doc_list'],
                                                  columns=['concept_id', 'concept_name', 'dimensions',
                                                           'table_format', 'filename'])
    datapoint_and_doc.to_csv(osp.join(SOURCE_DIR, 'datapoints.cache.csv'), index=False)
    # TODO: cleanup files which are not in the datapoints any more.


if __name__ == '__main__':
    print(get_config_dir())
    main()
    print('Done.')
