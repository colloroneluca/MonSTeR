# Script to process the metrics YAML file and generate a CSV with the required columns

import argparse
import yaml
import csv

def main():
    parser = argparse.ArgumentParser(description='Process metrics from a YAML file and generate a CSV.')
    parser.add_argument('--input_file', help='Path to the input YAML file')
    args = parser.parse_args()
    
    input_file = args.input_file
    
    
    # Read the YAML file
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Initialize dictionaries to store metrics and protocols
    metrics = {}
    protocols = set()
    
    # Process the data
    for key, value in data.items():
        try:
            value = float(value)
        except ValueError:
            continue
        
        key_parts = key.split('/')
        if len(key_parts) != 3:
            continue
        protocol, rank, metric = key_parts
        protocols.add(protocol)
        if metric not in metrics:
            metrics[metric] = []
        metrics[metric].append(value)
    
    # Define the mapping from input metric names to output column names
    metric_mapping = {
        'st2m': 'st2m',
        'm2st': 'm2st',
        'sm2t': 'ms2t',   # Input 'sm2t' corresponds to output 'ms2t'
        't2sm': 't2sm',
        'tm2s': 'tm2s',
        's2tm': 's2mt',   # Input 's2tm' corresponds to output 's2mt'
        't2m': 't2m',
        'm2t': 'm2t',
        's2m': 's2m',
        'm2s': 'm2s',
        't2s': 't2s',
        's2t': 's2t'
    }

    # Define the output columns in the desired order
    output_columns = ['Protocol', 'Method', 'st2m', 'm2st', 'ms2t', 't2sm', 'tm2s',
                      's2mt', 't2m', 'm2t', 's2m', 'm2s', 't2s', 's2t']
    
    # Compute the mean for each metric
    metric_means = {}
    for metric_name, values in metrics.items():
        metric_means[metric_name] = round(sum(values) / len(values), 2)
    
    # Prepare the output data
    protocol = list(protocols)[0] if protocols else ''
    output_data = {'Protocol': protocol, 'Method': ''}
    
    # Map the computed means to the output columns
    for column in output_columns[2:]:
        input_metric = None
        for k, v in metric_mapping.items():
            if v == column:
                input_metric = k
                break
        if input_metric and input_metric in metric_means:
            output_data[column] = metric_means[input_metric]
        else:
            output_data[column] = ''
    
    # Write the results to a CSV file
    import os
    folder_name = os.path.dirname(input_file)
    file_name = os.path.basename(input_file).split('.')[0]
    output_file = 'avg_'+file_name+'.csv'
    output_path = os.path.join(folder_name,output_file)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_columns)
        writer.writeheader()
        writer.writerow(output_data)
    
    print(f"CSV file '{output_path}' has been created with the computed metrics.")

if __name__ == '__main__':
    main()
