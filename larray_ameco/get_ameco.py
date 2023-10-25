import pandas as pd
import larray as la
import zipfile
import os

# aameco_get
ameco_input_path = 'http://ec.europa.eu/economy_finance/db_indicators/ameco/documents/'
ameco_output_path = ''
ameco_output_filename = 'ameco.csv'


def change_label(s):
    ls = s.split('.')
    nls = []
    nls.append(ls[0])
    nls.append('-'.join(ls[1:5]))
    nls.append(ls[5])

    return '.'.join(nls)


def read_ameco(path, ameco_file):
    df = pd.read_csv(path + ameco_file, sep=';', na_values=la.nan, skipinitialspace=True)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    # change last column to numeric
    df[df.columns[-1]] = df[df.columns[-1]].apply(pd.to_numeric, errors='coerce')

    df = df.drop(['COUNTRY', 'SUB-CHAPTER', 'TITLE', 'UNIT'], axis=1)
    idx_columns = [x for x in df.columns if not x.isdigit()]
    df_var = df.set_index(idx_columns)
    la_data = la.from_frame(df_var).rename({1: 'time'})
    la_data = la_data.set_labels('CODE', change_label)

    return la_data


def ameco_get(indicator, path='', drop_csv=True):
    '''

    Parameters
    ----------
    indicator
    path
    drop_csv

    Returns
    -------

    '''

    import urllib.request

    ameco_zip = indicator + '.zip'
    url = ameco_input_path + ameco_zip

    file_name, headers = urllib.request.urlretrieve(url, path + ameco_zip)
    print(file_name, 'Date', headers['Date'], 'Content-Length:', headers['Content-Length'])
    ziplist = unzip_ameco(path, ameco_zip)
    os.remove(path + ameco_zip)
    for idx, ameco_file in enumerate(ziplist):
        if idx == 0:
            la_data = read_ameco(path, ameco_file)
        else:
            la_data2 = read_ameco(path, ameco_file)
            la_data = la_data.append('CODE', la_data2)

        if drop_csv:
            os.remove(path + ameco_file)

    la_data = la_data.rename({'CODE': 'RCT'})
    return la_data


def unzip_ameco(path, ameco_zip):
    print('Unzipping', path + ameco_zip)
    namelist = []
    with zipfile.ZipFile(path + ameco_zip, 'r') as zip_ref:
        zip_ref.extractall(path)
        namelist = zip_ref.namelist()

    return namelist


def reshape_and_dump_ameco_dataset():
    import larray as la
    ameco_data = ameco_get('ameco0', path='__array_cache__/')
    ameco_data = ameco_data.split_axes(sep='.')

    # >>> The RCT Axis contains data of the form 'DEU.1-0-0-0.NPTD'.
    # >>> Objective is to split this into three distinct axes: Var, Country, and Numo.

    # Firs handle special case where '_' interferes with our splitting logic.
    # Replace 'D_W' with a placeholder 'D°°°W'
    ameco_data = ameco_data.set_labels('RCT', lambda label: label.replace('D_W', 'D°°°W'))

    # ROUND 1 
    # Split and reformat the string to prepare for the first split.
    # The format becomes "B_C|A" in preparation for a split on '_'.
    ameco_data = ameco_data.set_labels('RCT', lambda label: (lambda parts: f"{parts[1]}_{parts[2]}|{parts[0]}")(label.split('.')))
    axis_combined_NVC = ameco_data.axes[0].rename('numo_VarCountry')
    axis_time = ameco_data.axes[1]
    ameco_partially_cleaned = la.Array(ameco_data, axes=[axis_combined_NVC, axis_time]).split_axes(axis_combined_NVC)

    # ROUND 2
    # Modify labels from "C|A" to "C_A" to prepare for the second split on '_'.
    ameco_partially_cleaned = ameco_partially_cleaned.set_labels('VarCountry', lambda x: x.replace('|', '_'))
    axis_numo = ameco_partially_cleaned.axes[0]
    axis_combined_VC = ameco_partially_cleaned.axes[1].rename('Var_Country')
    axis_time = ameco_partially_cleaned.axes[2]
    ameco_final_cleaned = la.Array(ameco_partially_cleaned, axes=[axis_numo, axis_combined_VC, axis_time]).split_axes(axis_combined_VC)

    # Restore the special case 'D°°°W' back to 'D_W'.
    ameco_final_cleaned = ameco_final_cleaned.set_labels('Country', {'D°°°W': 'D_W'})

    ameco_final_cleaned.to_hdf('__array_cache__/data.h5', 'ameco0')
    print('file dumped to __array_cache__/data.h5')
