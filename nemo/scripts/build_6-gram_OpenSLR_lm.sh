#!/usr/bin/env bash
cd language_model2
# if [ ! -f "librispeech-lm-norm.txt.gz" ]; then
#   wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
# fi
# gzip -d librispeech-lm-norm.txt.gz
# convert all upper case characters to lower case
tr '[:upper:]' '[:lower:]' < alltext.txt > 4-gram.txt
cd ..
# build a language model
#pip install pandas
python build_lm_text.py language_model2/4-gram.txt --n 4
