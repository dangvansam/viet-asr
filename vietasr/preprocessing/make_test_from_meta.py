import os, ast, re, sys
import argparse

sys.path.append(os.getcwd())

from utils import save_json, write_txt



def normalize_label(label: str) -> str:
    label = label.upper()
    label = label.replace(" CK", "").replace("SPKT", "")
    label = re.sub("[^\ws]", " ", label)
    label = re.sub("\s+", " ", label)
    return label.strip()


def run(
        test_meta: str,
        json_output_path: str,
        scp_output_path: str,
        your_path: str
    ) -> None:

    json_output = dict()
    scp_output = list()
    total_dur = 0

    with open(test_meta, encoding='utf8') as fp:
        for line in fp:
            line = line.strip()

            try:
                wav_path, label, dur = line.split("|")
            except Exception as ex:
                print(line)
                continue

            wav_name = os.path.basename(wav_path)
            wav_path = os.path.join(your_path, wav_name)

            con1 = not os.path.isfile(wav_path)

            if '111' in label or con1: 
                print(wav_path, label)
                continue
            
            total_dur += ast.literal_eval(dur)

            scp_line = "{} {}\n".format(wav_name.split('.')[0], wav_path)
            scp_output.append(scp_line)

            label = normalize_label(label)
            json_output[wav_name] = label

    print("total test duration: {}".format(total_dur))
    save_json(json_output, json_output_path)
    write_txt(scp_output, scp_output_path)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_meta', type=str, required=True)
    parser.add_argument('--json_output_path', type=str, required=True)
    parser.add_argument('--scp_output_path', type=str, required=True)
    parser.add_argument('--your_path', type=str, required=True)

    args_input = parser.parse_args()
    run(**vars(args_input))