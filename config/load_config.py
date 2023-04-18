import yaml


def load_yaml(path):
    with open(path) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)
        # for key, value in args.items():
        #     temp = dict()
        #     for dictionary in value:
        #         temp.update(dictionary)

        #     meta[key] = temp
        # # print(meta)
    return args
