# coding:utf-8
"""
Filename : main.py
Role : TO-DO: Change role of main.py

@author : Sunwaee
"""

import pandas as pd

import databuilder
import train
import pipelines

# train_data = [["mid: What is FORTHEM?", "forthem.project"],
#               ["mid: What universities are involved in FORTHEM?", "forthem.universities"],
#               ["mid: Who are the associated partners of the project?", "forthem.partners"],
#               ["mid: How do I get involved in FORTHEM?", "forthem.involve"],
#               ["mid: Who can I contact to get more information?", 'forthem.contact'],
#               ["mid: What is the FORTHEM Digital Academy?", "fda.project"],
#               ["mid: How is the FORTHEM Digital Academy structured?", "fda.structure"],
#               ["mid: How do I enroll in FORTHEM courses and activities?", "fda.enrollment"],
#               ["mid: What recognition do I get for attending FORTHEM courses?", "fda.attendance"],
#               ["mid: What personal data is collected?", "fda.data"],
#               ["mid: Qu'est-ce que forthem?", "forthem.project"],
#               ["mid: Quelles universités font partie de forthem?", 'forthem.universities'],
#               ["mid: Quels sont les partenaires du projet?", 'forthem.partners'],
#               ["mid: Comment intégrer forthem?", 'forthem.involve'],
#               ["mid: Qui puis-je contacter?", 'forthem.contact'],
#               ["mid: Qu'est-ce que forthem digital academy?", 'fda.project'],
#               ["mid: Comment est structuré forthem?", 'fda.structure'],
#               ["mid: Comment s'inscrire à un cours?", 'fda.enrollment'],
#               ["mid: Quelle reconnaissance j'obtiens après la complétion d'un cours?", 'fda.attendance'],
#               ["mid: Quelles données sont collectées par forthem?", 'fda.data']]
# eval_data = [["mid: What is forthem?", 'forthem.project'],
#              ["mid: What universities are in forthem?", 'forthem.universities'],
#              ["mid: Who are the partners of the project?", 'forthem.partners'],
#              ["mid: How do I get in forthem?", 'forthem.involve'],
#              ["mid: Who can I contact?", 'forthem.contact'],
#              ["mid: What is forthem digital academy?", 'fda.project'],
#              ["mid: How is forthem digital academy structured?", 'fda.structure'],
#              ["mid: How do I enroll in courses and activities?", 'fda.enrollment'],
#              ["mid: What recognition do I get for attending courses?", 'fda.attendance'],
#              ["mid: What data is collected?", 'fda.data'],
#              ["mid: C'est quoi forthem?", 'forthem.project'],
#              ["mid: Quelles universités participent à forthem?", 'forthem.universities'],
#              ["mid: Quels sont les partenaires?", 'forthem.partners'],
#              ["mid: Comment participer à forthem?", 'forthem.involve'],
#              ["mid: Qui dois-je contacter?", 'forthem.contact'],
#              ["mid: C'est quoi forthem digital academy?", 'fda.project'],
#              ["mid: Quelle est la structure de forthem?", 'fda.structure'],
#              ["mid: Comment m'inscrire?", 'fda.enrollment'],
#              ["mid: Quelle reconnaissance j'obtiens?", 'fda.attendance'],
#              ["mid: Quelles données sont collectées?", 'fda.data']]
#
# train_df = pd.DataFrame(train_data, columns=['source_text', 'target_text'])
# eval_df = pd.DataFrame(eval_data, columns=['source_text', 'target_text'])
#
# train_df.to_csv("data/train.tsv", sep='\t')
# eval_df.to_csv("data/valid.tsv", sep='\t')
#
# databuilder_args = dict(
#     source_max_length=100,
#     target_max_length=20
# )
#
# # Running databuilder
# databuilder.run(args_dict=databuilder_args)
#
# model_args = dict(
#     output_dir="model",
#     do_train=True,
#     do_eval=True,
#     seed=42,
#     learning_rate=1e-4,
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     evaluation_strategy='epoch'
# )
#
# train.run(args_dict=model_args)

pipelines_args = dict(
    pipeline='classic'
)

pipeline = pipelines.run(pipelines_args)
res = pipeline(inputs='This is a test.')
print(res)


