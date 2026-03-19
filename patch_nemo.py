import os
import tarfile
import shutil
import tempfile

nemo_path = 'models/multitalker-vietnamese.nemo'
temp_dir = tempfile.mkdtemp()
print("Extracting to", temp_dir)

with tarfile.open(nemo_path, 'r') as tar:
    tar.extractall(temp_dir)

# Swap the tokenizer model!
old_tokenizer_name = None
for f in os.listdir(temp_dir):
    if f.endswith('_tokenizer.model'):
        old_tokenizer_name = f
        break

if old_tokenizer_name:
    print("Replacing", old_tokenizer_name)
    shutil.copy('data/merged_tokenizer.model',
                os.path.join(temp_dir, old_tokenizer_name))
else:
    print("Could not find tokenizer model in tarball!")
    exit(1)

# Edit model_config.yaml
yaml_path = os.path.join(temp_dir, 'model_config.yaml')
with open(yaml_path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_vocab = False
for line in lines:
    if line.strip() == 'vocabulary:':
        skip_vocab = True
        continue
    if skip_vocab:
        if line.startswith('  -') or line.strip() == '':
            continue
        else:
            skip_vocab = False
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(yaml_path, 'w') as f:
    f.writelines(new_lines)

print("Repacking tarball...")
with tarfile.open(nemo_path, 'w') as tar:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, temp_dir)
            tar.add(full_path, arcname=arcname)

shutil.rmtree(temp_dir)
print("Finished patching models/multitalker-vietnamese.nemo")
