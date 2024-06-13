import pandas as pd
import numpy as np
from tqdm import tqdm

def map_node_group_to_name(ng):
    if ng == {'off1-stdby', 'off1', 'short', 'off2-stdby', 'short-stdby', 'long', 'standard-stdby', 'off2', 'standard', 'long-stdby'}: 
        return 'off2-long-short-standard-off1'
    elif ng == {'off1-stdby', 'short', 'off2-stdby', 'off1', 'short-stdby', 'standard-stdby', 'off2', 'standard'}: 
        return 'off2-short-standard-off1'
    elif ng == {'gpu-stdby', 'gpul-stdby', 'short', 'short-stdby', 'gpu', 'gpul'}: 
        return 'gpu-gpul-short'
    elif ng == {'short', 'long', 'short-stdby', 'standard-stdby', 'bigmem-stdby', 'bigmem', 'standard', 'long-stdby'}: 
        return 'bigmem-long-short-standby'
    elif ng == {'debug-stdby', 'debug'}: 
        return 'debug'
    elif ng == {'gpu-stdby', 'gpul-stdby', 'short', 'bigscratch-stdby', 'bigscratch', 'short-stdby', 'gpu', 'gpul'}: 
        return 'bigscratch-gpu-gpul-short'
    elif ng == {'bigscratch-stdby', 'bigscratch', 'long', 'standard-stdby', 'bigmem-stdby', 'bigmem', 'standard', 'long-stdby'}: 
        return 'bigmem-bigscratch-long-standard'
    elif ng == {'gpu', 'gpu-stdby'}: 
        return 'gpu'
    elif ng == {'long', 'standard-stdby', 'bigmem-stdby', 'bigmem', 'standard', 'long-stdby'}: 
        return 'bigmem-long-standard'
    else:
        return None

def get_partition_nodes(partition_df, slurm_df):
    partition_nodes = partition_df.set_index('PARTITION')['NODELIST'].to_dict()
    
    remove = [partition for partition in partition_nodes if partition not in slurm_df['partition'].unique()]
    
    for partition in remove: del partition_nodes[partition] # Remove partitions with no instances in dataset
        
    return partition_nodes

def get_partition_node_group_proportions(partition_nodes, nodes_df):
    nodes_df['node_group'] = nodes_df.PARTITION.apply(set)
    nodes_df['node_group_name'] = nodes_df['node_group'].apply(map_node_group_to_name)
    
    partition_node_group_proportions = {}
    for partition, p_nodes in partition_nodes.items():
        partition_node_group_proportions[partition] = {}
        for ng, ng_nodes in nodes_df.groupby('node_group_name')['NODELIST']:
            partition_node_group_proportions[partition][ng] = len(set(p_nodes).intersection(ng_nodes))/len(set(p_nodes))
            
    return partition_node_group_proportions
        

def get_node_groups(nodes_df, slurm_df):
    node_groups = nodes_df['node_group'].apply(frozenset).unique()
    
    def get_node_group_set(partition):
        node_group_set = set()
        for node_group in node_groups:
            if partition in node_group:
                node_group_set.add(map_node_group_to_name(node_group))
        return node_group_set

    return slurm_df.partition.apply(get_node_group_set)
    

def create_job_event_timeline(df, conditions, target='queue_depth'):
    filtered_df = df[conditions].copy()

    key_columns = ['submit_time', 'start_time']
    if target == 'queue_depth':
        value_columns = ['submit_flag', 'start_flag']
    elif target == 'queue_size':
        value_columns = ['submit_size', 'start_size']
    elif target == 'queue_mem':
        value_columns = ['submit_mem', 'start_mem']
    
    event_timeline_df = pd.concat([
        filtered_df[[key_columns[0], value_columns[0]]].rename(columns={key_columns[0]: 'time'}),
        filtered_df[[key_columns[1], value_columns[1]]].rename(columns={key_columns[1]: 'time'})
    ])
    event_timeline_df[value_columns[0]] = event_timeline_df[value_columns[0]].fillna(0)
    event_timeline_df[value_columns[1]] = event_timeline_df[value_columns[1]].fillna(0)

    event_timeline_df.sort_values(by='time', inplace=True)

    return event_timeline_df

def get_queue_state_features(df, wallclock_knowledge='user', verbose=False):
    if not df.submit_time.is_monotonic_increasing:
        df.sort_values(by='submit_time', inplace=True)
    df['submit_flag'] = 1
    df['start_flag'] = df['start_time'].notnull().astype(int)
    df['end_flag'] = df['end_time'].notnull().astype(int)
    
    if wallclock_knowledge == 'perfect':
        df['submit_size'] = df['node_secs_used']
        df['start_size'] = df['node_secs_used']
        df['submit_mem'] = df['mem_req'] * df['wallclock_used']
        df['start_mem'] = df['mem_req'] * df['wallclock_used']
    elif wallclock_knowledge == 'user':
        df['submit_size'] = df['node_secs_req']
        df['start_size'] = df['node_secs_req']
        df['submit_mem'] = df['mem_req'] * df['wallclock_req']
        df['start_mem'] = df['mem_req'] * df['wallclock_req']
    elif wallclock_knowledge == 'pred':
        df['submit_size'] = df['node_secs_pred']
        df['start_size'] = df['node_secs_pred']
        df['submit_mem'] = df['mem_req'] * df['wallclock_pred']
        df['start_mem'] = df['mem_req'] * df['wallclock_pred']

    if verbose: print('Calculating queue depth at each job event')
    queue_conditions = (~df.start_time.isna()) | (~df.end_time.isna())
    event_timeline_df = create_job_event_timeline(df, queue_conditions, target='queue_depth')
    event_timeline_df['cumulative_submits'] = event_timeline_df['submit_flag'].cumsum()
    event_timeline_df['cumulative_starts'] = event_timeline_df['start_flag'].cumsum()
    event_timeline_df['queue_depth'] = event_timeline_df['cumulative_submits'] - event_timeline_df['cumulative_starts']
    event_timeline_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    df = df.merge(event_timeline_df[['time', 'queue_depth']], left_on='submit_time', right_on='time', how='left')

    if verbose: print('Calculating queue size at each job event')
    queue_conditions = (~df.start_time.isna()) | (~df.end_time.isna())
    event_timeline_df = create_job_event_timeline(df, queue_conditions, target='queue_size')
    event_timeline_df['cumulative_submit_size'] = event_timeline_df['submit_size'].cumsum()
    event_timeline_df['cumulative_start_size'] = event_timeline_df['start_size'].cumsum()
    event_timeline_df['queue_size'] = event_timeline_df['cumulative_submit_size'] - event_timeline_df['cumulative_start_size']
    event_timeline_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    df = df.merge(event_timeline_df[['time', 'queue_size']], left_on='submit_time', right_on='time', how='left')

    if verbose: print('Calculating queue memory at each job event')
    queue_conditions = (~df.start_time.isna()) | (~df.end_time.isna())
    event_timeline_df = create_job_event_timeline(df, queue_conditions, target='queue_mem')
    event_timeline_df['cumulative_submit_mem'] = event_timeline_df['submit_mem'].cumsum()
    event_timeline_df['cumulative_start_mem'] = event_timeline_df['start_mem'].cumsum()
    event_timeline_df['queue_mem'] = event_timeline_df['cumulative_submit_mem'] - event_timeline_df['cumulative_start_mem']
    event_timeline_df.drop_duplicates(subset=['time'], keep='last', inplace=True)
    df = df.merge(event_timeline_df[['time', 'queue_mem']], left_on='submit_time', right_on='time', how='left')

    df.drop(['submit_flag', 'start_flag', 'end_flag', 'submit_size', 'start_size', 'submit_mem', 'start_mem'], axis=1, inplace=True)
    
    return df

def calculate_queue_state_features(slurm_df, nodes_df, partition_node_group_proportions, knowledge_depth='node', wallclock_knowledge='user', verbose=False):
    processed_dfs = []
    names = []
    if knowledge_depth == 'node':
        for ng in nodes_df['node_group_name'].unique():
            # Get all rows where the nodes available to the job are in this node group (i.e. this node group is in the 'node_groups' set)
            filtered_df = slurm_df[[ng in node_group for node_group in slurm_df['node_groups']]].copy()
            if verbose: print(f'Calculating features for {ng} node-group with {len(filtered_df)} jobs.')
            processed_df = get_queue_state_features(filtered_df, wallclock_knowledge)
            processed_dfs.append(processed_df)
            names.append(ng)
    elif knowledge_depth == 'partition':
        for partition in slurm_df['partition'].unique():
            filtered_df = slurm_df[slurm_df['partition'] == partition].copy()
            if verbose: print(f'Calculating features for {partition} partition with {len(filtered_df)} jobs.')
            processed_df = get_queue_state_features(filtered_df, wallclock_knowledge)
            processed_dfs.append(processed_df)
            names.append(partition)
    elif knowledge_depth == 'cluster':
        processed_df = get_queue_state_features(slurm_df.copy(), wallclock_knowledge)
        processed_dfs.append(processed_df)
        names.append('slurm')
    
    queue_size_df = slurm_df.submit_time.copy()
    queue_depth_df = slurm_df.submit_time.copy()
    queue_mem_df = slurm_df.submit_time.copy()
    
    for i, processed_df in enumerate(processed_dfs):
        if verbose: print(f'Merging {names[i]} group')
        if not processed_df.submit_time.is_monotonic_increasing:
            processed_df.sort_values(by='submit_time')
            
        queue_depth_df = pd.merge_asof(queue_depth_df, processed_df[['submit_time', 'queue_depth']], on='submit_time', direction='forward')
        queue_size_df = pd.merge_asof(queue_size_df, processed_df[['submit_time', 'queue_size']], on='submit_time', direction='forward')
        queue_mem_df = pd.merge_asof(queue_mem_df, processed_df[['submit_time', 'queue_mem']], on='submit_time', direction='forward')

        if knowledge_depth == 'node':
            queue_depth_df.rename(columns={'queue_depth': names[i]+'_node_group'}, inplace=True)
            queue_size_df.rename(columns={'queue_size': names[i]+'_node_group'}, inplace=True)
            queue_mem_df.rename(columns={'queue_mem': names[i]+'_node_group'}, inplace=True)
        elif knowledge_depth == 'partition':
            queue_depth_df.rename(columns={'queue_depth': names[i]+'_partition'}, inplace=True)
            queue_size_df.rename(columns={'queue_size': names[i]+'_partition'}, inplace=True)
            queue_mem_df.rename(columns={'queue_mem': names[i]+'_partition'}, inplace=True)
    
    queue_depth_df.ffill(inplace=True)
    queue_size_df.ffill(inplace=True)
    queue_mem_df.ffill(inplace=True)
    

    if knowledge_depth == 'node':
        queue_avg_size_df = queue_size_df[['submit_time']].copy()
        queue_avg_mem_df = queue_mem_df[['submit_time']].copy()
        # Get weighted average depth/size of relevant node-groups
        for partition, node_groups in partition_node_group_proportions.items():
            queue_depth_df[partition + '_mean'] = 0
            queue_size_df[partition + '_mean'] = 0
            queue_mem_df[partition + '_mean'] = 0
            queue_avg_size_df[partition + '_mean'] = 0
            queue_avg_mem_df[partition + '_mean'] = 0
            for node_group, proportion in node_groups.items():
                queue_depth_df[partition + '_mean'] += queue_depth_df[node_group + '_node_group'] * proportion
                queue_size_df[partition + '_mean'] += queue_size_df[node_group + '_node_group'] * proportion
                queue_mem_df[partition + '_mean'] += queue_mem_df[node_group + '_node_group'] * proportion
                queue_avg_size_df[partition + '_mean'] += ((queue_size_df[node_group + '_node_group'] / 
                                                            queue_depth_df[node_group + '_node_group']) * proportion).fillna(0)
                queue_avg_mem_df[partition + '_mean'] += ((queue_mem_df[node_group + '_node_group'] / 
                                                           queue_depth_df[node_group + '_node_group']) * proportion).fillna(0)

        # Get minimum depth/size of relevant node-groups
        for partition, node_groups in partition_node_group_proportions.items():
            non_zero_ngs = [node_group + '_node_group' for node_group, proportion in node_groups.items() if proportion != 0]
            queue_depth_df[partition + '_min'] = queue_depth_df[non_zero_ngs].min(axis=1)
            queue_size_df[partition + '_min'] = queue_size_df[non_zero_ngs].min(axis=1)
            queue_mem_df[partition + '_min'] = queue_mem_df[non_zero_ngs].min(axis=1)
            queue_avg_size_df[partition + '_min'] = (queue_size_df[non_zero_ngs] / queue_depth_df[non_zero_ngs]).min(axis=1).fillna(0)
            queue_avg_mem_df[partition + '_min'] = (queue_mem_df[non_zero_ngs] / queue_depth_df[non_zero_ngs]).min(axis=1).fillna(0)
    else:
        queue_avg_size_df = None
        queue_avg_mem_df = None

    queue_depth_df.to_parquet(f'../../data/system_state/queue_depth_{knowledge_depth}_{wallclock_knowledge}.parquet')
    queue_size_df.to_parquet(f'../../data/system_state/queue_size_{knowledge_depth}_{wallclock_knowledge}.parquet')
    queue_mem_df.to_parquet(f'../../data/system_state/queue_mem_{knowledge_depth}_{wallclock_knowledge}.parquet')
    if knowledge_depth == 'node':
        queue_avg_size_df.to_parquet(f'../../data/system_state/avg_queue_size_{knowledge_depth}_{wallclock_knowledge}.parquet')
        queue_avg_mem_df.to_parquet(f'../../data/system_state/avg_queue_mem_{knowledge_depth}_{wallclock_knowledge}.parquet')


def create_running_nodes_timeline(df, wallclock_knowledge='user', knowledge_depth='node'):
    if wallclock_knowledge == 'user':
        runtime = 'wallclock_req'
    elif wallclock_knowledge == 'perfect':
        runtime = 'wallclock_used'
    elif wallclock_knowledge == 'pred':
        runtime = 'wallclock_pred'
        
    if knowledge_depth == 'node':
        columns = ['partition', 'nodelist', 'mem_req', runtime]
        drop_subset = 'nodelist'
    elif knowledge_depth == 'partition':
        columns = ['partition', 'mem_req', 'nodes_used', runtime]
        drop_subset = 'partition'
    elif knowledge_depth == 'cluster':
        columns = ['partition', 'mem_req', 'nodes_used', runtime]
        
    df_submit = df[['job_array_id', 'submit_time'] + columns].copy()
    df_submit['event_type'] = 'submit'
    df_submit.rename(columns={'submit_time': 'event_time'}, inplace=True)

    if knowledge_depth != 'cluster':
        df_start = df[['job_array_id', 'start_time'] + columns].dropna(subset=[drop_subset]).copy()
    else:
        df_start = df[['job_array_id', 'start_time'] + columns].copy()
    df_start['event_type'] = 'start'
    df_start.rename(columns={'start_time': 'event_time'}, inplace=True)

    if knowledge_depth != 'cluster':
        df_end = df[['job_array_id', 'end_time'] + columns].dropna(subset=[drop_subset]).copy()
    else:
        df_end = df[['job_array_id', 'end_time'] + columns].copy()
    df_end['event_type'] = 'end'
    df_end.rename(columns={'end_time': 'event_time'}, inplace=True)

    event_timeline_df = pd.concat([df_submit, df_start, df_end])
    event_timeline_df['event_type'] = pd.Categorical(event_timeline_df['event_type'], ['submit', 'start', 'end'])
    
    event_timeline_df.sort_values(by=['event_time', 'event_type'], inplace=True)

    return event_timeline_df
        

def job_start(row, running_jobs_info, job_id, nodes_used, mem_req, t_predicted, t_start, partition_nodes, partition_to_jobs, knowledge_depth):
    running_jobs_info[job_id] = {
        't_predicted': t_predicted,
        't_start': t_start,
        't_remaining_est': t_predicted,
        'nodes': nodes_used,
        'mem': mem_req
    }
    
    if knowledge_depth == 'node':
        for partition, nodes in partition_nodes.items():
            intersection_size = len(set(nodes_used).intersection(nodes))
            if intersection_size > 0:
                partition_to_jobs[partition].add(job_id)
                running_jobs_info[job_id][partition + '_node_count'] = intersection_size
                n_nodes = len(running_jobs_info[job_id]['nodes'])
                total_mem = running_jobs_info[job_id]['mem']
                running_jobs_info[job_id][partition + '_mem_used'] = (intersection_size / n_nodes) * total_mem
    elif knowledge_depth == 'partition':
        partition = row.partition
        partition_to_jobs[partition].add(job_id)
        running_jobs_info[job_id]['partition'] = partition
        running_jobs_info[job_id][partition + '_node_count'] = running_jobs_info[job_id]['nodes']
        running_jobs_info[job_id][partition + '_mem_used'] = running_jobs_info[job_id]['mem']
    # Nothing more to do for cluster-level knowledge depth


def job_end(running_jobs_info, job_id, partition_nodes, partition_to_jobs, knowledge_depth): 
    if job_id in running_jobs_info:
        if knowledge_depth == 'node':
            for partition in partition_nodes:
                partition_to_jobs[partition].discard(job_id)
        elif knowledge_depth == 'partition':
            partition_to_jobs[running_jobs_info[job_id]['partition']].discard(job_id)
        del running_jobs_info[job_id]
        # Nothing more to do for cluster-level knowledge depth
    

def update_remaining_time(t_current, running_jobs_info, partition_to_jobs, partition):
    for job_id in partition_to_jobs[partition]:
        t_elapsed = t_current - running_jobs_info[job_id]['t_start']
        t_remaining_est = max(running_jobs_info[job_id]['t_predicted'] - t_elapsed.total_seconds(), 0)
        running_jobs_info[job_id]['t_remaining_est'] = t_remaining_est


def calculate_remaining_runtime(partition_nodes, partition_to_jobs, running_jobs_info, partition, knowledge_depth='node', this_job_id=0):
    if knowledge_depth in ['node', 'partition']:
        job_ids = partition_to_jobs[partition]
    elif knowledge_depth == 'cluster':
        job_ids = running_jobs_info.keys()
        
    n_nodes = np.zeros(len(job_ids), dtype=np.float32)
    time_remaining = np.zeros(len(job_ids), dtype=np.float32)
    mem_req = np.zeros(len(job_ids), dtype=np.float32)
    
    if knowledge_depth in ['node', 'partition']:
        T_remaining_est = {}
        
        for i, job_id in enumerate(job_ids):
            time_remaining[i] = running_jobs_info[job_id]['t_remaining_est']
            n_nodes[i] = running_jobs_info[job_id][partition + '_node_count']
            mem_req[i] = running_jobs_info[job_id]['mem']

        runtime_remaining = time_remaining * n_nodes
        mem_remaining = time_remaining * mem_req
        
        if len(runtime_remaining) == 0:
            T_remaining_est[partition + '_rt_mean'] = 0
            T_remaining_est[partition + '_rt_median'] = 0
            T_remaining_est[partition + '_rt_min'] = 0
            T_remaining_est[partition + '_mem_mean'] = 0
            T_remaining_est[partition + '_mem_median'] = 0
            T_remaining_est[partition + '_mem_min'] = 0
            T_remaining_est[partition + '_nnodes'] = 0
        else:
            T_remaining_est[partition + '_rt_mean'] = np.mean(runtime_remaining)
            T_remaining_est[partition + '_rt_median'] = np.median(runtime_remaining)
            T_remaining_est[partition + '_rt_min'] = np.min(runtime_remaining)
            T_remaining_est[partition + '_mem_mean'] = np.mean(mem_remaining)
            T_remaining_est[partition + '_mem_median'] = np.median(mem_remaining)
            T_remaining_est[partition + '_mem_min'] = np.min(mem_remaining)
            T_remaining_est[partition + '_nnodes'] = sum(n_nodes)
            
    elif knowledge_depth == 'cluster':
        job_ids = running_jobs_info.keys()
        n_nodes = np.zeros(len(job_ids), dtype=np.float32)
        time_remaining = np.zeros(len(job_ids), dtype=np.float32)
        mem_req = np.zeros(len(job_ids), dtype=np.float32)
        T_remaining_est = {}
        
        for i, job_id in enumerate(job_ids):
            time_remaining[i] = running_jobs_info[job_id]['t_remaining_est']
            n_nodes[i] = running_jobs_info[job_id]['nodes']
            mem_req[i] = running_jobs_info[job_id]['mem']
            
        runtime_remaining = time_remaining * n_nodes
        mem_remaining = time_remaining * mem_req
        
        if len(time_remaining) == 0:
            T_remaining_est['rt_mean'] = 0
            T_remaining_est['rt_median'] = 0
            T_remaining_est['rt_min'] = 0
            T_remaining_est['mem_mean'] = 0
            T_remaining_est['mem_median'] = 0
            T_remaining_est['mem_min'] = 0
            T_remaining_est['nnodes'] = 0
        else:
            T_remaining_est['rt_mean'] = np.mean(runtime_remaining)
            T_remaining_est['rt_median'] = np.median(runtime_remaining)
            T_remaining_est['rt_min'] = np.min(runtime_remaining)
            T_remaining_est['mem_mean'] = np.mean(mem_remaining)
            T_remaining_est['mem_median'] = np.median(mem_remaining)
            T_remaining_est['mem_min'] = np.min(mem_remaining)
            T_remaining_est['nnodes'] = sum(n_nodes)

    return T_remaining_est


def handle_event(row, running_jobs_info, partition_nodes, partition_to_jobs, wallclock_knowledge='user', knowledge_depth='node'):
    job_id = row.job_array_id
    event_type = row.event_type
    t_current = row.event_time
        
    if wallclock_knowledge == 'user':
        t_predicted = row.wallclock_req
    elif wallclock_knowledge == 'perfect':
        t_predicted = row.wallclock_used
    elif wallclock_knowledge == 'pred':
        t_predicted = row.wallclock_pred

    if event_type == 'submit': 
        update_remaining_time(t_current, running_jobs_info, partition_to_jobs, row.partition)
        return calculate_remaining_runtime(partition_nodes, partition_to_jobs, running_jobs_info, row.partition, knowledge_depth, job_id)
    elif event_type == 'start':
        if knowledge_depth == 'node':
            nodes_used = set(row.nodelist)
        else:
            nodes_used = row.nodes_used
        job_start(row, running_jobs_info, job_id, nodes_used, row.mem_req, t_predicted, 
                  t_current, partition_nodes, partition_to_jobs, knowledge_depth)
    elif event_type == 'end':
        job_end(running_jobs_info, job_id, partition_nodes, partition_to_jobs, knowledge_depth)

def calculate_system_utilization_features(partition_nodes, slurm_df, wallclock_knowledge='user', knowledge_depth='node'):
    partition_to_jobs = {partition: set() for partition in slurm_df.partition.unique()}
    running_jobs_info = {}
    
    event_timeline_df = create_running_nodes_timeline(slurm_df, wallclock_knowledge, knowledge_depth)

    if knowledge_depth in ['node', 'partition']:
        T_remaining = {}
        for partition in slurm_df.partition.unique():
            T_remaining[partition + '_rt_mean'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_rt_median'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_rt_min'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_mem_mean'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_mem_median'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_mem_min'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            T_remaining[partition + '_nnodes'] = np.zeros(len(event_timeline_df), dtype=np.float32)
            
        for i, row in tqdm(enumerate(event_timeline_df.itertuples()), total=event_timeline_df.shape[0], desc='Processing Events'):
            if row.event_type == 'submit':
                if row.partition == 'off3' and knowledge_depth == 'node':
                    continue # No node information for off3 cluster
                T_remaining_est = handle_event(row, running_jobs_info, partition_nodes, partition_to_jobs, wallclock_knowledge, knowledge_depth)
                #for partition in partition_nodes:
                partition = row.partition
                T_remaining[partition + '_rt_mean'][i] = T_remaining_est[partition + '_rt_mean']
                T_remaining[partition + '_rt_median'][i] = T_remaining_est[partition + '_rt_median']
                T_remaining[partition + '_rt_min'][i] = T_remaining_est[partition + '_rt_min']
                T_remaining[partition + '_mem_mean'][i] = T_remaining_est[partition + '_mem_mean']
                T_remaining[partition + '_mem_median'][i] = T_remaining_est[partition + '_mem_median']
                T_remaining[partition + '_mem_min'][i] = T_remaining_est[partition + '_mem_min']
                T_remaining[partition + '_nnodes'][i] = T_remaining_est[partition + '_nnodes']
            else:
                handle_event(row, running_jobs_info, partition_nodes, partition_to_jobs, wallclock_knowledge, knowledge_depth)
    elif knowledge_depth == 'cluster':
        T_remaining = {}
        T_remaining['runtime_mean'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['runtime_median'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['runtime_min'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['mem_mean'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['mem_median'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['mem_min'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        T_remaining['nnodes'] = np.zeros(len(event_timeline_df), dtype=np.float32)
        for i, row in tqdm(enumerate(event_timeline_df.itertuples()), total=event_timeline_df.shape[0], desc='Processing Events'):
            if row.event_type == 'submit':
                T_remaining_est = handle_event(row, running_jobs_info, partition_nodes, partition_to_jobs, wallclock_knowledge, knowledge_depth)
                T_remaining['runtime_mean'][i] = T_remaining_est['rt_mean']
                T_remaining['runtime_median'][i] = T_remaining_est['rt_median']
                T_remaining['runtime_min'][i] = T_remaining_est['rt_min']
                T_remaining['mem_mean'][i] = T_remaining_est['mem_mean']
                T_remaining['mem_median'][i] = T_remaining_est['mem_median']
                T_remaining['mem_min'][i] = T_remaining_est['mem_min']
                T_remaining['nnodes'][i] = T_remaining_est['nnodes']
            else:
                handle_event(row, running_jobs_info, partition_nodes, partition_to_jobs, wallclock_knowledge, knowledge_depth)

    running_nodes_df = pd.DataFrame(T_remaining)
    
    event_timeline_df = event_timeline_df.reset_index(drop=True)
    
    runtime_remaining_df = pd.concat([event_timeline_df, running_nodes_df], axis=1)

    runtime_remaining_df.to_parquet(f'../../data/system_state/runtime_remaining_{knowledge_depth}_{wallclock_knowledge}.parquet')
