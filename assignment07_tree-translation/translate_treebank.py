#!/usr/bin/env python3
#coding: utf-8

from collections import defaultdict, Counter
import numpy as np

conllu, para = "cs-ud-train.conllu", "monogr_cs-sk-tag.aligned"

lexicon = defaultdict(Counter)
pos = defaultdict()
with open(para, "r", encoding="utf-8") as translations:
    for line in translations.readlines()[:1000]:
        linesplit = line.split()
        if len(linesplit) == 4:
            src_word, src_pos, tgt_word = linesplit[0], linesplit[1], linesplit[3]
            lexicon[src_word][tgt_word] += 1
            if (src_word, src_pos) not in pos:
                pos[(src_word, src_pos)] = [tgt_word]
            else:
                if tgt_word not in pos[(src_word, src_pos)]:
                    pos[(src_word, src_pos)].append(tgt_word)
#print([pos[item] for item in pos if len(pos[item])>1])

results = []
exist, non_exist = 0, 0
with open(conllu) as treebank:
    for line in treebank:
        line = line.strip()
        fields = line.split('\t')
        if fields[0].isdigit():
            word = fields[1]
            src_pos = fields[3]
            top_translations, same_pos_translations, translation = [], [], word
            if word in lexicon:
                exist += 1
                top_translations = lexicon[word].most_common(5)
                top_1_translation = lexicon[word].most_common(1)[0][0]
                translation = top_1_translation
            else:
                non_exist += 1
            if (word, src_pos) in pos:
                same_pos_translations = pos[((word, src_pos))]
            translations = [top for top in top_translations for same in same_pos_translations if top[0]==same]
            if translations:
                ## actually no need of argmax because .common(3) already return in descending order of counts,
                ## so "max_index" is always 0
                max_index = np.argmax([top[1] for top in translations])
                translation = translations[max_index][0] 
            #print(translation)
            fields[1] = translation
            results.append("\t".join(fields))
            #print(*fields, sep='\t')
        else:
            #print(line)
            results.append(line)
# 157279, 1016003         
print(exist, non_exist)
with open("translated_sk.conllu", "w", encoding="utf-8") as f:
    for line in results:
        print(line, file=f)
