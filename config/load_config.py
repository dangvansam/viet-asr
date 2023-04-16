import yaml


def load_yaml(path):
    meta = dict()

    with open(path) as file:
        
        args = yaml.load(file, Loader=yaml.FullLoader)

        for key, value in args.items():
            temp = dict()
            for dictionary in value:
                temp.update(dictionary)

            meta[key] = temp
        print(meta)
    
    return meta
