import os
import yaml

def load_data_path(path):
    mapping_file = os.path.join(path, 'mapping.yaml')
    data_mapping = yaml.load(open(mapping_file), Loader=yaml.Loader)
    
    image_paths = data_mapping['image_paths']
    mapping = data_mapping['mapping']
    
    res = []
    for data_key, dataset_name in mapping.items():
        res.append([os.path.join(path, data_key + '.json'), image_paths[dataset_name]])
        
    return res