import os, glob, ast, re
import argparse
from typing import List



def write_txt(
        data: List[str],
        path: str,
        mode: str = "w"
    ) -> None:

    with open(path, mode= mode, encoding='utf8') as fp:
        fp.writelines(data)


def search_number_in_string(string):
    return re.search('\d', string).group()


def normalize_label(label: str) -> str:
    label = label.upper()
    label = label.replace(" CK", "").replace("SPKT", "")
    label = re.sub("[^\ws]", " ", label)
    label = re.sub("\s+", " ", label)
    return label.strip()


def run(
        meta_folder: str,
        text_output_path: str
    ):

    meta_paths = glob.glob(os.path.join(meta_folder, "*.txt"))
    text = list()
    not_exist = dict()

    total_dur = 0
    for p in meta_paths:
        p_dur = 0
        with open(p, encoding= "utf8") as fp:
            for line in fp:

                try:
                    wav_path, label, dur = line.strip().split("|")
                except Exception as ex:
                    print(line)
                
                label = normalize_label(label)
                text.append("{}\n".format(label))
                p_dur += ast.literal_eval(dur)

                if not os.path.isfile(wav_path):
                    folder = wav_path.split("/")[1]

                    if folder in not_exist:
                        not_exist[folder] += 1
                    else:
                        not_exist[folder] = 1
    
        print("total duration of {}: {}".format(p, p_dur))
        total_dur += p_dur
    
    print("total duration of {} folder: {}".format(meta_folder, total_dur))
    print("not exist information: {}".format(not_exist))

    write_txt(text, text_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_folder', type=str, required=True)
    parser.add_argument('--text_output_path', type=str, required=True)

    args_input = parser.parse_args()
    run(**vars(args_input))

