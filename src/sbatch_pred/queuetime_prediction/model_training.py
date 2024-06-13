import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

import traceback

import timeit

partition_params = {
    'short': {'objective': 'reg:absoluteerror',
        'lambda': 0.9113048169248219, 
        'alpha': 7.152516267517524e-08, 
        'subsample': 0.8653024056251306, 
        'colsample_bytree': 0.9180764517917145, 
        'max_depth': 5, 
        'min_child_weight': 9, 
        'eta': 0.0730979317347613, 
        'gamma': 0.9593886752442862},
    'standard': {'objective': 'reg:absoluteerror',
        'lambda': 0.023118568011426248, 
        'alpha': 1.2508534690123174e-08, 
        'subsample': 0.6868288476811126, 
        'colsample_bytree': 0.5050586814636849, 
        'max_depth': 5, 
        'min_child_weight': 5, 
        'eta': 0.09970611946172692, 
        'gamma': 0.010105687005242697},
    'standard-stdby': {'objective': 'reg:absoluteerror',
        'lambda': 0.3619307143153695, 
        'alpha': 1.4967673022735968, 
        'subsample': 0.8063761320047677, 
        'colsample_bytree': 0.8077542166004148, 
        'max_depth': 3, 
        'min_child_weight': 9, 
        'eta': 0.0562167622424796, 
        'gamma': 0.9691633377490854},
    'short-stdby': {'objective': 'reg:absoluteerror',
        'lambda': 1.1519411170043984e-05, 
        'alpha': 0.007902435933472556, 
        'subsample': 0.6788703083582164, 
        'colsample_bytree': 0.5506132171777032, 
        'max_depth': 3, 
        'min_child_weight': 2, 
        'eta': 0.1674490797764757, 
        'gamma': 0.7508939831948626},
    'long': {'objective': 'reg:absoluteerror',
        'lambda': 1.8496272880022395e-05, 
        'alpha': 0.011108781149569936, 
        'subsample': 0.8968275173116053, 
        'colsample_bytree': 0.6913966036460466, 
        'max_depth': 3, 
        'min_child_weight': 10, 
        'eta': 0.30693151898624954, 
        'gamma': 0.33459637244883855},
    'off1': {'objective': 'reg:absoluteerror',
        'lambda': 1.059697138977052e-08, 
        'alpha': 0.0030068172624127367, 
        'subsample': 0.7831377211840796, 
        'colsample_bytree': 0.8048731749522329, 
        'max_depth': 2, 
        'min_child_weight': 2, 
        'eta': 0.49654969388986453, 
        'gamma': 0.2642244822799702},
    'debug': {'objective': 'reg:absoluteerror',
        'lambda': 0.002488044384155592,
        'alpha': 0.6927419492831092,
        'subsample': 0.5167739897123247,
        'colsample_bytree': 0.9229657144636869,
        'max_depth': 1,
        'min_child_weight': 1,
        'eta': 0.14436186901220613,
        'gamma': 0.9251777030095379},
    'gpu': {'objective': 'reg:absoluteerror',
        'lambda': 0.40641464056286003, 
        'alpha': 0.0013054208083885565, 
        'subsample': 0.7335462657980485, 
        'colsample_bytree': 0.731377276707252, 
        'max_depth': 3, 
        'min_child_weight': 5, 
        'eta': 0.08918537156841723, 
        'gamma': 0.9659036090938147},
    'bigmem': {'objective': 'reg:absoluteerror',
        'lambda': 0.037894530718397997, 
        'alpha': 0.0011756706772269414, 
        'subsample': 0.5540569371597356, 
        'colsample_bytree': 0.6657357759239696, 
        'max_depth': 3, 
        'min_child_weight': 8, 
        'eta': 0.0778528831872922, 
        'gamma': 0.21442945387011558},
    'long-stdby': {'objective': 'reg:absoluteerror',
        'lambda': 0.6019363427817087, 
        'alpha': 6.040842985416869e-05, 
        'subsample': 0.6297134779220037, 
        'colsample_bytree': 0.52150235155023, 
        'max_depth': 10, 
        'min_child_weight': 8, 
        'eta': 0.033473908521051976, 
        'gamma': 0.4875894472980114},
    'off2': {'objective': 'reg:absoluteerror',
        'lambda': 0.000321892389565315, 
        'alpha': 0.0055600917380620235, 
        'subsample': 0.6931631402581113, 
        'colsample_bytree': 0.5233703835412075, 
        'max_depth': 1, 
        'min_child_weight': 9, 
        'eta': 0.43435153088628975, 
        'gamma': 0.13169883930169596},
    'debug-stdby': {'objective': 'reg:absoluteerror',
        'lambda': 0.034748149709146636, 
        'alpha': 5.976362637805212e-08, 
        'subsample': 0.5701524903110685, 
        'colsample_bytree': 0.7654383778161886, 
        'max_depth': 2, 
        'min_child_weight': 6, 
        'eta': 0.13213150298425969, 
        'gamma': 0.68615226349395},
    'gpu-stdby': {'objective': 'reg:absoluteerror',
        'lambda': 2.89192249852083e-05, 
        'alpha': 0.0034973672412518444, 
        'subsample': 0.9259468938279266, 
        'colsample_bytree': 0.656673703620848, 
        'max_depth': 10, 
        'min_child_weight': 8, 
        'eta': 0.28304592680658774, 
        'gamma': 0.24885627639456873},
    'gpul': {'objective': 'reg:absoluteerror',
        'lambda': 0.9932725103292945, 
        'alpha': 5.979530566192595, 
        'subsample': 0.8342626894245434, 
        'colsample_bytree': 0.6536916238936807, 
        'max_depth': 5, 
        'min_child_weight': 4, 
        'eta': 0.17512860886717052, 
        'gamma': 0.6389841283885577},
    'bigscratch': {'objective': 'reg:absoluteerror',
        'lambda': 1.6798649743173752e-08, 
        'alpha': 0.012665586560264923, 
        'subsample': 0.6262865101019934, 
        'colsample_bytree': 0.8754042368509773, 
        'max_depth': 1, 
        'min_child_weight': 6, 
        'eta': 0.35311631507273533, 
        'gamma': 0.5249233360322736}
}

partition_clusters = {'standard': {0: {'min': 0.0, 'max': 2.0, 'len': 10078},
                                  1: {'min': 3.0, 'max': 11.0, 'len': 17163},
                                  2: {'min': 12.0, 'max': 60.0, 'len': 13246},
                                  3: {'min': 61.0, 'max': 240.0, 'len': 13215},
                                  4: {'min': 241.0, 'max': 780.0, 'len': 23382},
                                  5: {'min': 781.0, 'max': 2100.0, 'len': 30309},
                                  6: {'min': 2101.0, 'max': 5400.0, 'len': 42801},
                                  7: {'min': 5401.0, 'max': 14400.0, 'len': 40890},
                                  8: {'min': 14401.0, 'max': 39600.0, 'len': 32786},
                                  9: {'min': 39601.0, 'max': 111600.0, 'len': 36353},
                                  10: {'min': 111601.0, 'max': 950400.0, 'len': 22954},
                                  11: {'min': 950401.0, 'max': 99999999.0, 'len': 0}},
                      'short-stdby': {0: {'min': 0.0, 'max': 12.0, 'len': 2231},
                                  1: {'min': 13.0, 'max': 180.0, 'len': 1738},
                                  2: {'min': 181.0, 'max': 1500.0, 'len': 3088},
                                  3: {'min': 1501.0, 'max': 6300.0, 'len': 3850},
                                  4: {'min': 6301.0, 'max': 21600.0, 'len': 11878},
                                  5: {'min': 21601.0, 'max': 46800.0, 'len': 23385},
                                  6: {'min': 46801.0, 'max': 82800.0, 'len': 22774},
                                  7: {'min': 82801.0, 'max': 259200.0, 'len': 17384},
                                  8: {'min': 259201.0, 'max': 604800.0, 'len': 10572},
                                  9: {'min': 604801.0, 'max': 2851200.0, 'len': 5839},
                                  10: {'min': 2851201.0, 'max': 99999999.0, 'len': 0}},
                      'gpu': {0: {'min': 0.0, 'max': 4.0, 'len': 1357},
                                  1: {'min': 5.0, 'max': 30.0, 'len': 1049},
                                  2: {'min': 31.0, 'max': 180.0, 'len': 799},
                                  3: {'min': 181.0, 'max': 720.0, 'len': 1060},
                                  4: {'min': 721.0, 'max': 3600.0, 'len': 1113},
                                  5: {'min': 3601.0, 'max': 18000.0, 'len': 1848},
                                  6: {'min': 18001.0, 'max': 86400.0, 'len': 1892},
                                  7: {'min': 86401.0, 'max': 864000.0, 'len': 1318},
                                  8: {'min': 864001.0, 'max': 99999999.0, 'len': 0}},
                      'debug': {0: {'min': 0.0, 'max': 0.0, 'len': 1994},
                                  1: {'min': 1.0, 'max': 2.0, 'len': 4893},
                                  2: {'min': 3.0, 'max': 5.0, 'len': 5397},
                                  3: {'min': 6.0, 'max': 12.0, 'len': 3836},
                                  4: {'min': 13.0, 'max': 45.0, 'len': 3141},
                                  5: {'min': 46.0, 'max': 120.0, 'len': 950},
                                  6: {'min': 121.0, 'max': 300.0, 'len': 987},
                                  7: {'min': 301.0, 'max': 1200.0, 'len': 1359},
                                  8: {'min': 1201.0, 'max': 4500.0, 'len': 1104},
                                  9: {'min': 4501.0, 'max': 18000.0, 'len': 672},
                                  10: {'min': 18001.0, 'max': 90000.0, 'len': 755},
                                  11: {'min': 90001.0, 'max': 99999999.0, 'len': 0}},
                      'short': {0: {'min': 0.0, 'max': 2.0, 'len': 15585},
                                  1: {'min': 3.0, 'max': 13.0, 'len': 23815},
                                  2: {'min': 14.0, 'max': 120.0, 'len': 17703},
                                  3: {'min': 121.0, 'max': 300.0, 'len': 21720},
                                  4: {'min': 301.0, 'max': 900.0, 'len': 30910},
                                  5: {'min': 901.0, 'max': 2400.0, 'len': 35544},
                                  6: {'min': 2401.0, 'max': 6300.0, 'len': 38302},
                                  7: {'min': 6301.0, 'max': 14400.0, 'len': 34440},
                                  8: {'min': 14401.0, 'max': 36000.0, 'len': 43751},
                                  9: {'min': 36001.0, 'max': 100800.0, 'len': 31039},
                                  10: {'min': 100801.0, 'max': 777600.0, 'len': 18433},
                                  11: {'min': 777601.0, 'max': 99999999.0, 'len': 0}},
                      'bigmem': {0: {'min': 0.0, 'max': 30.0, 'len': 278},
                                  1: {'min': 31.0, 'max': 1200.0, 'len': 352},
                                  2: {'min': 1201.0, 'max': 8100.0, 'len': 900},
                                  3: {'min': 8101.0, 'max': 46800.0, 'len': 1707},
                                  4: {'min': 46801.0, 'max': 259200.0, 'len': 2747},
                                  5: {'min': 259201.0, 'max': 1900800.0, 'len': 2413},
                                  6: {'min': 1900801.0, 'max': 99999999.0, 'len': 0}},
                      'off2': {0: {'min': 0.0, 'max': 6.0, 'len': 1258},
                                  1: {'min': 7.0, 'max': 120.0, 'len': 1471},
                                  2: {'min': 121.0, 'max': 660.0, 'len': 1262},
                                  3: {'min': 661.0, 'max': 3300.0, 'len': 1935},
                                  4: {'min': 3301.0, 'max': 50400.0, 'len': 1246},
                                  5: {'min': 50401.0, 'max': 1209600.0, 'len': 481},
                                  6: {'min': 1209601.0, 'max': 99999999.0, 'len': 0}},
                      'long-stdby': {0: {'min': 0.0, 'max': 7.0, 'len': 111},
                                  1: {'min': 8.0, 'max': 180.0, 'len': 154},
                                  2: {'min': 181.0, 'max': 6300.0, 'len': 169},
                                  3: {'min': 6301.0, 'max': 43200.0, 'len': 242},
                                  4: {'min': 43201.0, 'max': 144000.0, 'len': 525},
                                  5: {'min': 144001.0, 'max': 432000.0, 'len': 1180},
                                  6: {'min': 432001.0, 'max': 864000.0, 'len': 1598},
                                  7: {'min': 864001.0, 'max': 4060800.0, 'len': 2876},
                                  8: {'min': 4060801.0, 'max': 99999999.0, 'len': 0}},
                      'long': {0: {'min': 0.0, 'max': 4.0, 'len': 3452},
                                  1: {'min': 5.0, 'max': 30.0, 'len': 3921},
                                  2: {'min': 31.0, 'max': 120.0, 'len': 3926},
                                  3: {'min': 121.0, 'max': 420.0, 'len': 4882},
                                  4: {'min': 421.0, 'max': 1500.0, 'len': 6674},
                                  5: {'min': 1501.0, 'max': 3300.0, 'len': 15434},
                                  6: {'min': 3301.0, 'max': 6300.0, 'len': 10205},
                                  7: {'min': 6301.0, 'max': 14400.0, 'len': 11206},
                                  8: {'min': 14401.0, 'max': 39600.0, 'len': 6770},
                                  9: {'min': 39601.0, 'max': 129600.0, 'len': 6665},
                                  10: {'min': 129601.0, 'max': 1209600.0, 'len': 4156},
                                  11: {'min': 1209601.0, 'max': 99999999.0, 'len': 0}},
                      'standard-stdby': {0: {'min': 0.0, 'max': 4.0, 'len': 11155},
                                  1: {'min': 5.0, 'max': 120.0, 'len': 6209},
                                  2: {'min': 121.0, 'max': 1800.0, 'len': 3788},
                                  3: {'min': 1801.0, 'max': 11700.0, 'len': 7543},
                                  4: {'min': 11701.0, 'max': 39600.0, 'len': 21979},
                                  5: {'min': 39601.0, 'max': 108000.0, 'len': 19330},
                                  6: {'min': 108001.0, 'max': 345600.0, 'len': 37510},
                                  7: {'min': 345601.0, 'max': 691200.0, 'len': 30937},
                                  8: {'min': 691201.0, 'max': 3628800.0, 'len': 15125},
                                  9: {'min': 3628801.0, 'max': 99999999.0, 'len': 0}},
                      'off1': {0: {'min': 0.0, 'max': 1.0, 'len': 5548},
                                  1: {'min': 2.0, 'max': 6.0, 'len': 7130},
                                  2: {'min': 7.0, 'max': 45.0, 'len': 6005},
                                  3: {'min': 46.0, 'max': 360.0, 'len': 4140},
                                  4: {'min': 361.0, 'max': 1500.0, 'len': 5473},
                                  5: {'min': 1501.0, 'max': 5400.0, 'len': 13692},
                                  6: {'min': 5401.0, 'max': 21600.0, 'len': 4801},
                                  7: {'min': 21601.0, 'max': 100800.0, 'len': 6425},
                                  8: {'min': 100801.0, 'max': 691200.0, 'len': 5238},
                                  9: {'min': 691201.0, 'max': 99999999.0, 'len': 0}},
                      'debug-stdby': {0: {'min': 0.0, 'max': 4.0, 'len': 1218},
                                  1: {'min': 5.0, 'max': 45.0, 'len': 1040},
                                  2: {'min': 46.0, 'max': 420.0, 'len': 448},
                                  3: {'min': 421.0, 'max': 2700.0, 'len': 369},
                                  4: {'min': 2701.0, 'max': 14400.0, 'len': 746},
                                  5: {'min': 14401.0, 'max': 86400.0, 'len': 1036},
                                  6: {'min': 86401.0, 'max': 1468800.0, 'len': 1036},
                                  7: {'min': 1468801.0, 'max': 99999999.0, 'len': 0}},
                      'bigscratch': {0: {'min': 0.0, 'max': 3.0, 'len': 11},
                                  1: {'min': 4.0, 'max': 30.0, 'len': 2},
                                  2: {'min': 31.0, 'max': 720.0, 'len': 1},
                                  3: {'min': 721.0, 'max': 28800.0, 'len': 5},
                                  4: {'min': 28801.0, 'max': 144000.0, 'len': 11},
                                  5: {'min': 144001.0, 'max': 99999999.0, 'len': 0}},
                      'off1-stdby': {0: {'min': 0.0, 'max': 25.0, 'len': 4},
                                  1: {'min': 26.0, 'max': 30.0, 'len': 8},
                                  2: {'min': 31.0, 'max': 40.0, 'len': 7},
                                  3: {'min': 41.0, 'max': 45.0, 'len': 8},
                                  4: {'min': 46.0, 'max': 90.0, 'len': 8},
                                  5: {'min': 91.0, 'max': 99999999.0, 'len': 0}},
                      'gpul': {0: {'min': 0.0, 'max': 3.0, 'len': 53},
                                  1: {'min': 4.0, 'max': 45.0, 'len': 50},
                                  2: {'min': 46.0, 'max': 360.0, 'len': 29},
                                  3: {'min': 361.0, 'max': 4500.0, 'len': 39},
                                  4: {'min': 4501.0, 'max': 32400.0, 'len': 54},
                                  5: {'min': 32401.0, 'max': 169200.0, 'len': 48},
                                  6: {'min': 169201.0, 'max': 604800.0, 'len': 77},
                                  7: {'min': 604801.0, 'max': 99999999.0, 'len': 0}},
                      'gpu-stdby': {0: {'min': 0.0, 'max': 13.0, 'len': 153},
                                  1: {'min': 14.0, 'max': 600.0, 'len': 113},
                                  2: {'min': 601.0, 'max': 18000.0, 'len': 297},
                                  3: {'min': 18001.0, 'max': 68400.0, 'len': 414},
                                  4: {'min': 68401.0, 'max': 259200.0, 'len': 607},
                                  5: {'min': 259201.0, 'max': 518400.0, 'len': 969},
                                  6: {'min': 518401.0, 'max': 99999999.0, 'len': 0}},
                      'bigmem-stdby': {0: {'min': 0.0, 'max': 32400.0, 'len': 7},
                                  1: {'min': 32401.0, 'max': 158400.0, 'len': 98},
                                  2: {'min': 158401.0, 'max': 345600.0, 'len': 313},
                                  3: {'min': 345601.0, 'max': 518400.0, 'len': 447},
                                  4: {'min': 518401.0, 'max': 99999999.0, 'len': 0}}}

    

def get_feature(df1, df2):
    # Concatenates the dataframes and returns the value
    # in the column for the correct partition (i.e. the partition for that job)
    concatenated_df = pd.concat([df1, df2], axis=1)
    return concatenated_df.apply(lambda row: row[row.partition], axis=1)
    
def get_model_data(knowledge_depth='cluster', wallclock_knowledge='user', model_data_fp=None):
    # Retrieve the model data if it has already been built
    if model_data_fp is None:
        model_data_fp = '../../data/model_data/model_data_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    try:
        model_data_df = pd.read_parquet(model_data_fp)
        print(f'Loaded model data at path {model_data_fp}')
        return model_data_df
    except FileNotFoundError:
        if knowledge_depth == 'combined':
            knowledge_depth='partition'
            model_data_df = get_model_data(knowledge_depth, wallclock_knowledge)
            knowledge_depth='node'
            node_model_data_df = get_model_data(knowledge_depth, wallclock_knowledge)
            model_data_df['queue_depth_min_log_NODE'] = node_model_data_df['queue_depth_min_log']
            model_data_df['queue_size_min_log_NODE'] = node_model_data_df['queue_size_min_log']
            model_data_df['queue_mem_min_log_NODE'] = node_model_data_df['queue_mem_min_log']
            model_data_df['queue_depth_min_log_NODE'] = node_model_data_df['queue_depth_min_log']
            model_data_df['queue_avg_size_min_log_NODE'] = node_model_data_df['queue_avg_size_min_log']
            model_data_df['queue_avg_mem_min_log_NODE'] = node_model_data_df['queue_avg_mem_min_log']
            model_data_df['nodes_available_NODE'] = node_model_data_df['nodes_available']
            print('Finished combinining partition- and node-level data')
            return model_data_df
        else:
            print(f'Model data not found at {model_data_fp}')

def get_input_features(knowledge_depth='baseline'):
    baseline_features = ['priority', 'qos_num', 'mem_req', 'gpus_req', 'processors_req', 'nodes_req', 'wallclock_req_log', 'array_pos']
    if knowledge_depth == 'node':
        features = baseline_features + ['queue_depth_min_log', 'queue_size_min_log', 'queue_mem_min_log',
                    'queue_avg_size_min', 'queue_avg_mem_min', 
                    'nodes_available', 'runtime_remaining_min_log', 'mem_remaining_min_log']
    elif knowledge_depth in ['partition', 'cluster']:
        features = baseline_features + ['queue_depth_log', 'queue_size_log', 'queue_mem_log',
                    'queue_avg_size', 'queue_avg_mem', 
                    'nodes_available', 'runtime_remaining_min_log', 'mem_remaining_min_log']
    elif knowledge_depth == 'combined':
        # Combining partition and node-level features
        features = baseline_features + ['queue_depth_log', 'queue_size_log', 'queue_mem_log',
                    'queue_avg_size', 'queue_avg_mem', 
                    'nodes_available', 'runtime_remaining_min_log', 'mem_remaining_min_log',
                    'queue_depth_min_log_NODE', 'queue_size_min_log_NODE', 'queue_mem_min_log_NODE', 
                    'queue_avg_size_min_log_NODE', 'queue_avg_mem_min_log_NODE', 
                    'nodes_available_NODE']
    return features
        
def prepare_model_data(knowledge_depth='cluster', wallclock_knowledge='user', model_data_fp=None, save_model_data=True, verbose=True):
    # Retrieve the model data if it has already been built
    if model_data_fp is None:
        model_data_fp = '../../data/model_data/model_data_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    try:
        model_data_df = pd.read_parquet(model_data_fp)
        print(f'Loaded model data at path {model_data_fp}')
        return model_data_df
    except FileNotFoundError:
        print('Building model data from source files.')

    # Get Slurm & partition data
    slurm_fp = '../../data/slurm_data.parquet'
    partition_fp = '../../data/partitions.parquet'

    # Build file paths for required data
    directory = '../../data/system_state/'
    runtime_remaining_fp = directory + 'runtime_remaining_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    queue_avg_size_fp = directory +  'avg_queue_size_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    queue_avg_mem_fp = directory + 'avg_queue_mem_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    queue_depth_fp = directory + 'queue_depth_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    queue_size_fp = directory + 'queue_size_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'
    queue_mem_fp = directory + 'queue_mem_' + knowledge_depth + '_' + wallclock_knowledge + '.parquet'

    # Get required dataframes
    if verbose: print('Loading source files')
    slurm_df = pd.read_parquet(slurm_fp).reset_index(drop=True)
    partition_df = pd.read_parquet(partition_fp)
    
    runtime_remaining_df = pd.read_parquet(runtime_remaining_fp)
    if knowledge_depth == 'node':
        queue_avg_size_df = pd.read_parquet(queue_avg_size_fp)
        queue_avg_mem_df = pd.read_parquet(queue_avg_mem_fp)
    queue_depth_df = pd.read_parquet(queue_depth_fp)
    queue_size_df = pd.read_parquet(queue_size_fp)
    queue_mem_df = pd.read_parquet(queue_mem_fp)
    

    # Error checking
    assert len(slurm_df) == len(queue_depth_df)
    assert len(slurm_df) == len(queue_size_df)

    assert slurm_df.submit_time.is_monotonic_increasing
    assert queue_depth_df.submit_time.is_monotonic_increasing
    assert queue_size_df.submit_time.is_monotonic_increasing
    assert queue_mem_df.submit_time.is_monotonic_increasing

    assert (slurm_df.submit_time == queue_depth_df.submit_time).all()
    assert (slurm_df.submit_time == queue_size_df.submit_time).all()
    assert (slurm_df.submit_time == queue_mem_df.submit_time).all()

    if verbose: print('Filtering source files and building feature dataframes')
    if knowledge_depth == 'node':
        queue_depth_mean_df = queue_depth_df[['submit_time'] + [col for col in queue_depth_df.columns if 'mean' in col]].copy()
        queue_depth_min_df = queue_depth_df[['submit_time'] + [col for col in queue_depth_df.columns if 'min' in col]].copy()
        queue_size_mean_df = queue_size_df[['submit_time'] + [col for col in queue_size_df.columns if 'mean' in col]].copy()
        queue_size_min_df = queue_size_df[['submit_time'] + [col for col in queue_size_df.columns if 'min' in col]].copy()
        queue_mem_mean_df = queue_mem_df[['submit_time'] + [col for col in queue_mem_df.columns if 'mean' in col]].copy()
        queue_mem_min_df = queue_mem_df[['submit_time'] + [col for col in queue_mem_df.columns if 'min' in col]].copy()

        queue_avg_size_df['submit_time'] = queue_size_df['submit_time']
        queue_avg_mem_df['submit_time'] = queue_mem_df['submit_time']
        queue_avg_size_mean_df = queue_avg_size_df[['submit_time'] + [col for col in queue_avg_size_df.columns if 'mean' in col]].copy()
        queue_avg_size_min_df = queue_avg_size_df[['submit_time'] + [col for col in queue_avg_size_df.columns if 'min' in col]].copy()
        queue_avg_mem_mean_df = queue_avg_mem_df[['submit_time'] + [col for col in queue_avg_mem_df.columns if 'mean' in col]].copy()
        queue_avg_mem_min_df = queue_avg_mem_df[['submit_time'] + [col for col in queue_avg_mem_df.columns if 'min' in col]].copy()
        
        suffix = '_mean'
        queue_avg_size_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_avg_size_mean_df.columns]
        queue_avg_mem_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_avg_mem_mean_df.columns]
        queue_depth_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_depth_mean_df.columns]
        queue_size_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_size_mean_df.columns]
        queue_mem_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_mem_mean_df.columns]
        
        suffix = '_min'
        queue_avg_size_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_avg_size_min_df.columns]
        queue_avg_mem_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_avg_mem_min_df.columns]
        queue_depth_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_depth_min_df.columns]
        queue_size_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_size_min_df.columns]
        queue_mem_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_mem_min_df.columns]
        
    elif knowledge_depth == 'partition':
        suffix = '_partition'
        queue_depth_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_depth_df.columns]
        queue_size_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_size_df.columns]
        queue_mem_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in queue_mem_df.columns]

    runtime_remaining_df = runtime_remaining_df[runtime_remaining_df.event_type == 'submit'].copy()

    if wallclock_knowledge == 'user': runtime = 'wallclock_req'
    elif wallclock_knowledge == 'perfect': runtime = 'wallclock_used'
    elif wallclock_knowledge == 'pred': runtime = 'wallclock_pred'

    runtime_remaining_df.rename(columns={'event_time': 'submit_time'}, inplace=True)

    runtime_nnodes_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'nnodes' in col]
    runtime_median_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'rt_median' in col or 'runtime_median' in col]
    runtime_mean_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'rt_mean' in col or 'runtime_mean' in col]
    runtime_min_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'rt_min' in col or 'runtime_min' in col]
    mem_median_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'mem_median' in col]
    mem_mean_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'mem_mean' in col]
    mem_min_columns = ['submit_time'] + [col for col in runtime_remaining_df.columns if 'mem_min' in col]
    
    runtime_remaining_nnodes_df = runtime_remaining_df[runtime_nnodes_columns].copy()
    runtime_remaining_median_df = runtime_remaining_df[runtime_median_columns].copy()
    runtime_remaining_mean_df = runtime_remaining_df[runtime_mean_columns].copy()
    runtime_remaining_min_df = runtime_remaining_df[runtime_min_columns].copy()
    mem_remaining_median_df = runtime_remaining_df[mem_median_columns].copy()
    mem_remaining_mean_df = runtime_remaining_df[mem_mean_columns].copy()
    mem_remaining_min_df = runtime_remaining_df[mem_min_columns].copy()

    suffix = '_nnodes'
    runtime_remaining_nnodes_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in runtime_remaining_nnodes_df.columns]
    
    suffix = '_rt_median'
    runtime_remaining_median_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in runtime_remaining_median_df.columns]
    suffix = '_rt_mean'
    runtime_remaining_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in runtime_remaining_mean_df.columns]
    suffix = '_rt_min'
    runtime_remaining_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in runtime_remaining_min_df.columns]

    suffix = '_mem_median'
    mem_remaining_median_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in mem_remaining_median_df.columns]
    suffix = '_mem_mean'
    mem_remaining_mean_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in mem_remaining_mean_df.columns]
    suffix = '_mem_min'
    mem_remaining_min_df.columns = [col[:-len(suffix)] if col.endswith(suffix) else col for col in mem_remaining_min_df.columns]
    
    
    runtime_remaining_nnodes_df.reset_index(inplace=True, drop=True)
    
    runtime_remaining_median_df.reset_index(inplace=True, drop=True)
    runtime_remaining_mean_df.reset_index(inplace=True, drop=True)
    runtime_remaining_min_df.reset_index(inplace=True, drop=True)
    
    mem_remaining_median_df.reset_index(inplace=True, drop=True)
    mem_remaining_mean_df.reset_index(inplace=True, drop=True)
    mem_remaining_min_df.reset_index(inplace=True, drop=True)


    assert len(slurm_df) == len(runtime_remaining_median_df)
    assert len(slurm_df) == len(runtime_remaining_mean_df)
    assert len(slurm_df) == len(runtime_remaining_nnodes_df)
    
    assert runtime_remaining_nnodes_df.submit_time.is_monotonic_increasing
    
    assert runtime_remaining_median_df.submit_time.is_monotonic_increasing
    assert runtime_remaining_mean_df.submit_time.is_monotonic_increasing
    assert runtime_remaining_min_df.submit_time.is_monotonic_increasing
    
    assert mem_remaining_median_df.submit_time.is_monotonic_increasing
    assert mem_remaining_mean_df.submit_time.is_monotonic_increasing
    assert mem_remaining_min_df.submit_time.is_monotonic_increasing
    
    assert (slurm_df.submit_time == runtime_remaining_median_df.submit_time).all()
    assert (slurm_df.submit_time == runtime_remaining_mean_df.submit_time).all()
    assert (slurm_df.submit_time == runtime_remaining_min_df.submit_time).all()

    assert (slurm_df.submit_time == mem_remaining_median_df.submit_time).all()
    assert (slurm_df.submit_time == mem_remaining_mean_df.submit_time).all()
    assert (slurm_df.submit_time == mem_remaining_min_df.submit_time).all()
    
    assert (slurm_df.submit_time == runtime_remaining_nnodes_df.submit_time).all()

    if knowledge_depth == 'node':
        # Add off3 column for error handling due to off3 not being in sinfo data
        queue_depth_mean_df['off3'] = 0
        queue_depth_min_df['off3'] = 0
        queue_size_mean_df['off3'] = 0
        queue_size_min_df['off3'] = 0
        queue_mem_mean_df['off3'] = 0
        queue_mem_min_df['off3'] = 0
        
        queue_avg_size_mean_df['off3'] = 0
        queue_avg_size_min_df['off3'] = 0
        queue_avg_mem_mean_df['off3'] = 0
        queue_avg_mem_min_df['off3'] = 0
        
        runtime_remaining_median_df['off3'] = 0
        runtime_remaining_mean_df['off3'] = 0
        runtime_remaining_min_df['off3'] = 0
        mem_remaining_median_df['off3'] = 0
        mem_remaining_mean_df['off3'] = 0
        mem_remaining_min_df['off3'] = 0
        runtime_remaining_nnodes_df['off3'] = 0

    model_data_df = slurm_df.copy()

    if verbose: print('Getting features from feature dataframes')
    if knowledge_depth == 'node':
        model_data_df['queue_avg_size_mean'] = get_feature(model_data_df, queue_avg_size_mean_df)
        model_data_df['queue_avg_size_min'] = get_feature(model_data_df, queue_avg_size_min_df)
        model_data_df['queue_avg_mem_mean'] = get_feature(model_data_df, queue_avg_mem_mean_df)
        model_data_df['queue_avg_mem_min'] = get_feature(model_data_df, queue_avg_mem_min_df)
        model_data_df['queue_depth_mean'] = get_feature(model_data_df, queue_depth_mean_df)
        model_data_df['queue_depth_min'] = get_feature(model_data_df, queue_depth_min_df)
        model_data_df['queue_size_mean'] = get_feature(model_data_df, queue_size_mean_df)
        model_data_df['queue_size_min'] = get_feature(model_data_df, queue_size_min_df)
        model_data_df['queue_mem_mean'] = get_feature(model_data_df, queue_mem_mean_df)
        model_data_df['queue_mem_min'] = get_feature(model_data_df, queue_mem_min_df)
        
        model_data_df['queue_avg_size_mean_log'] = np.log(model_data_df['queue_avg_size_mean'] + .1)
        model_data_df['queue_avg_size_min_log'] = np.log(model_data_df['queue_avg_size_min'] + .1)
        model_data_df['queue_avg_mem_mean_log'] = np.log(model_data_df['queue_avg_mem_mean'] + .1)
        model_data_df['queue_avg_mem_min_log'] = np.log(model_data_df['queue_avg_mem_min'] + .1)
        model_data_df['queue_depth_mean_log'] = np.log(model_data_df['queue_depth_mean'] + .1)
        model_data_df['queue_depth_min_log'] = np.log(model_data_df['queue_depth_min'] + .1)
        model_data_df['queue_size_mean_log'] = np.log(model_data_df['queue_size_mean'] + .1)
        model_data_df['queue_size_min_log'] = np.log(model_data_df['queue_size_min'] + .1)
        model_data_df['queue_mem_mean_log'] = np.log(model_data_df['queue_mem_mean'] + .1)
        model_data_df['queue_mem_min_log'] = np.log(model_data_df['queue_mem_min'] + .1)


    elif knowledge_depth in ['partition', 'cluster']:
        if knowledge_depth == 'partition':
            model_data_df['queue_depth'] = get_feature(model_data_df, queue_depth_df)
            model_data_df['queue_size'] = get_feature(model_data_df, queue_size_df)
            model_data_df['queue_mem'] = get_feature(model_data_df, queue_mem_df)
        elif knowledge_depth == 'cluster':
            model_data_df['queue_depth'] = queue_depth_df['queue_depth']
            model_data_df['queue_size'] = queue_size_df['queue_size']
            model_data_df['queue_mem'] = queue_mem_df['queue_mem']
            
        model_data_df['queue_avg_size'] = model_data_df['queue_size'] / model_data_df['queue_depth']
        model_data_df['queue_avg_mem'] = model_data_df['queue_mem'] / model_data_df['queue_depth']

        model_data_df['queue_avg_size_log'] = np.log(model_data_df['queue_avg_size'] + .1)
        model_data_df['queue_avg_mem_log'] = np.log(model_data_df['queue_avg_mem'] + .1)
        model_data_df['queue_depth_log'] = np.log(model_data_df['queue_depth'] + .1)
        model_data_df['queue_size_log'] = np.log(model_data_df['queue_size'] + .1)
        model_data_df['queue_mem_log'] = np.log(model_data_df['queue_mem'] + .1)

    if knowledge_depth in ['node', 'partition']:
        model_data_df['runtime_remaining_nnodes'] = get_feature(model_data_df, runtime_remaining_nnodes_df)
        model_data_df['runtime_remaining_median'] = get_feature(model_data_df, runtime_remaining_median_df)
        model_data_df['runtime_remaining_mean'] = get_feature(model_data_df, runtime_remaining_mean_df)
        model_data_df['runtime_remaining_min'] = get_feature(model_data_df, runtime_remaining_min_df)
        model_data_df['mem_remaining_median'] = get_feature(model_data_df, mem_remaining_median_df)
        model_data_df['mem_remaining_mean'] = get_feature(model_data_df, mem_remaining_mean_df)
        model_data_df['mem_remaining_min'] = get_feature(model_data_df, mem_remaining_min_df)
    elif knowledge_depth == 'cluster':
        model_data_df['runtime_remaining_nnodes'] = runtime_remaining_nnodes_df['nnodes']
        model_data_df['runtime_remaining_median'] = runtime_remaining_median_df['runtime_median']
        model_data_df['runtime_remaining_mean'] = runtime_remaining_mean_df['runtime_mean']
        model_data_df['runtime_remaining_min'] = runtime_remaining_min_df['runtime_min']
        model_data_df['mem_remaining_median'] = mem_remaining_median_df['mem_median']
        model_data_df['mem_remaining_mean'] = mem_remaining_mean_df['mem_mean']
        model_data_df['mem_remaining_min'] = mem_remaining_min_df['mem_min']
    
    model_data_df['runtime_remaining_median_log'] = np.log(model_data_df['runtime_remaining_median'] + .1)
    model_data_df['runtime_remaining_mean_log'] = np.log(model_data_df['runtime_remaining_mean'] + .1)
    model_data_df['runtime_remaining_min_log'] = np.log(model_data_df['runtime_remaining_min'] + .1)
    model_data_df['mem_remaining_median_log'] = np.log(model_data_df['mem_remaining_median'] + .1)
    model_data_df['mem_remaining_mean_log'] = np.log(model_data_df['mem_remaining_mean'] + .1)
    model_data_df['mem_remaining_min_log'] = np.log(model_data_df['mem_remaining_min'] + .1)

    model_data_df['wallclock_req_log'] = np.log(model_data_df['wallclock_req'] + .1)
    model_data_df['node_secs_req_log'] = np.log(model_data_df['node_secs_req'] + .1)
    model_data_df['mem_req_log'] = np.log(model_data_df['mem_req'] + .1)

    if knowledge_depth in ['node', 'partition']:
        model_data_df['total_nodes'] = model_data_df.partition.map(dict(zip(partition_df.PARTITION, partition_df.num_nodes))).astype('float')
    elif knowledge_depth == 'cluster':
        nodes_df = pd.read_parquet('../../data/nodes.parquet')
        model_data_df['total_nodes'] = len(nodes_df.NODELIST.unique())
        
    model_data_df['nodes_available'] = (model_data_df['total_nodes'] - model_data_df['runtime_remaining_nnodes']).astype('float')

    model_data_df['queue_wait_log'] = np.log(model_data_df['queue_wait'] + .1)
    
    if save_model_data: 
        if verbose: print(f'Saving model data to {model_data_fp}')
        model_data_df.to_parquet(model_data_fp)
    else:
        return model_data_df


def get_feature_correlation(df, target):
    return df.select_dtypes(exclude=['object']).corr()[target].sort_values()
    

def xgb_regression(params, X_train, y_train, X_test, y_test, target, verbose):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if target == 'queue_wait_log':
        mae = mean_absolute_error(np.exp(y_test), np.exp(predictions)) / 3600
    else:
        mae = mean_absolute_error(y_test, predictions) / 3600
    if verbose: print(f'Train: {train_window}, Test: {test_window}, Split Time: {split_time}, MAE: {mae}')

    return model, predictions, mae

def assign_cluster(row):
    partition = row['partition']
    if partition == 'off3': return 0 # No node information for off3 cluster
        
    wait_time = row['queue_wait']
    
    for cluster, times in partition_clusters[partition].items():
        if times['min'] <= wait_time <= times['max']: return cluster
            
    return None


def xgb_classification(params, X_train, y_train, X_test, target, verbose=False):   
    try:
        le = LabelEncoder()
        model = xgb.XGBClassifier(**params)
        y_train = le.fit_transform(y_train)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = le.inverse_transform(predictions)
    except Exception as e:
        print(f'Exception type: {type(e).__name__}')
        print(f'Error message: {e}')
        traceback.print_exc()
        print('Something went wrong. Returning None')
        return None, None

    return model, predictions


def train_test_model(model_data_df, train_window, test_window, split_time, features, target, partition, model_type='xgb_reg', params=None, verbose=False):
    train_condition = model_data_df.start_time.between(split_time - pd.Timedelta(days=train_window), split_time, inclusive='left')
    test_condition = model_data_df.submit_time.between(split_time, split_time + pd.Timedelta(days=test_window))
    if partition != 'cluster':
        partition_condition = model_data_df.partition == partition
        train_condition &= partition_condition
        test_condition &= partition_condition # Commenting this out temporarily because I'm pre-splitting data into partitions
        
    train_df = model_data_df[train_condition].copy()
    test_df = model_data_df[test_condition].copy()

    if len(test_df) == 0:
        if model_type == 'xgb_reg':
            return np.nan, None, None
        elif model_type == 'xgb_cls':
            return None, None
    
    X_train = train_df[features].copy()
    y_train = train_df[target]
    X_test = test_df[features].copy()
    y_test = test_df[target]

    if model_type == 'xgb_reg':
        model, predictions, mae = xgb_regression(params, X_train, y_train, X_test, y_test, target, verbose)
    elif model_type == 'xgb_cls':
        model, predictions = xgb_classification(params, X_train, y_train, X_test, target, verbose)
        if model is not None and predictions is not None and verbose:
            conf_matrix = confusion_matrix(y_test, predictions)
            class_counts = conf_matrix.sum(axis=1)
            per_class_accuracy = np.zeros_like(class_counts, dtype=float)
            for i in range(len(class_counts)):
                if class_counts[i] > 0:
                    per_class_accuracy[i] = conf_matrix[i, i] / class_counts[i]
                else:
                    per_class_accuracy[i] = np.nan
            for i, accuracy in enumerate(per_class_accuracy):
                print(f'Accuracy for class {i}: {accuracy}')
                
    feature_importance_df = None
    if model is not None:
        importance = model.feature_importances_
    
        feature_importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        # if verbose: display(feature_importance_df)
        
    results_df = test_df.copy()
    results_df['split_time'] = split_time

    if model_type == 'xgb_reg':
        results_df['wait_time_act'] = test_df['queue_wait']
        results_df['wait_time_pred'] = predictions
        return mae, feature_importance_df, results_df
    elif model_type == 'xgb_cls':
        results_df['cluster_pred'] = predictions
        return feature_importance_df, results_df

    
def get_partition_results(model_data_df, train_window, test_window, start_date, features, target, partition, model_type='xgb_reg', params=None, n_days=120, verbose=False):
    results = {'train_window' : [],
               'test_window' : [],
               'split_time' : [],
               'feature_importance': [],
               'results_df': []
              }
    if model_type == 'xgb_reg':
        results['mae'] = []
        for days in range(n_days):
            split_time = pd.to_datetime(start_date) + pd.Timedelta(days=days)
            mae, feature_importance_df, results_df = train_test_model(model_data_df, train_window, test_window, split_time, features, target, partition, model_type=model_type, params=params, verbose=verbose)
            if results_df is not None:
                results['train_window'].append(train_window)
                results['test_window'].append(test_window)
                results['split_time'].append(split_time)
                results['mae'].append(mae)
                results['feature_importance'].append(feature_importance_df)
                results['results_df'].append(results_df)
    elif model_type == 'xgb_cls':
        for days in range(n_days):
            split_time = pd.to_datetime(start_date) + pd.Timedelta(days=days)
            feature_importance_df, results_df = train_test_model(model_data_df, train_window, test_window, split_time, features, target, partition, model_type, params, verbose)
            if results_df is not None:
                results['train_window'].append(train_window)
                results['test_window'].append(test_window)
                results['split_time'].append(split_time)
                results['feature_importance'].append(feature_importance_df)
                results['results_df'].append(results_df)
                
    return results

def get_feature_importance_stats(results):
    pivoted_importance_dfs = [df.pivot(index='Feature', columns=[], values='Importance') for df in results['feature_importance']]
    combined_importance_df = pd.concat(pivoted_importance_dfs, axis=1)
    combined_importance_df['average'] = combined_importance_df.mean(axis=1)
    combined_importance_df['median'] = combined_importance_df[[col for col in combined_importance_df if not col == 'average']].median(axis=1)
    return combined_importance_df

def combine_results(results):
    return pd.concat(results['results_df'], ignore_index=True).reset_index(drop=True)

def get_results_df(partition_results, model_type='xgb_reg', feature_set='all'):
    results_dfs = []
    for partition in partition_results:
        if partition == 'off3':
            continue # No node-level information for off3 partition
        if len(partition_results[partition]['results_df']) == 0:
            continue
        results_dfs.append(combine_results(partition_results[partition]))
    results_df = pd.concat(results_dfs, ignore_index=True).reset_index(drop=True)

    if model_type == 'xgb_reg':
        results_df['wait_time_pred_seconds'] = np.exp(results_df['wait_time_pred'])
        results_df['wait_time_pred_hours'] = results_df['wait_time_pred_seconds'] / 3600
        results_df['wait_time_act_hours'] = results_df['wait_time_act'] / 3600
        results_df['wait_time_err_hours'] = results_df.wait_time_pred_hours - results_df.wait_time_act_hours
        results_df['wait_time_abs_err_hours'] = np.abs(results_df['wait_time_err_hours'])

    results_df['feature_set'] = feature_set
    return results_df

def save_results(partition_results, knowledge_depth='node', wallclock_knowledge='user', model_type='xgb_reg'):
    feature_set = f'{knowledge_depth}_{wallclock_knowledge}'
    df = get_results_df(partition_results, model_type, feature_set)
    filepath = f'../../data/results/{model_type}_{feature_set}_results.parquet'
    df.to_parquet(filepath)

    # Test that file was saved correcly
    pd.read_parquet(filepath)
    return df
    