import json, os, yaml, argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Summary Data dir')
    parser.add_argument('input_dir', type=str, help='the input dir name')
    return parser.parse_args()


def main(args):
    input_dir = args.input_dir
    mapping_fn = os.path.join(input_dir, 'mapping.yaml')
    assert os.path.exists(mapping_fn)
    
    mapping = yaml.load(open(mapping_fn), Loader=yaml.Loader)
    
    count = {}
    for key in mapping['mapping'].keys():
        fn = os.path.join(input_dir, key + '.json')
        count[key] = len(json.load(open(fn)))
        print(key, count[key])
    
    mapping['count'] = count
    yaml.dump(mapping, open(mapping_fn, 'w'))


if __name__ == '__main__':
    main(parse_args())
