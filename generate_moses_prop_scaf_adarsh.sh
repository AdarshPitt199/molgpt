#!/bin/bash

# Single property and scaffold based generation
python generate/generate.py --model_weight /home/furiosa/Documents/Adarsh/PolyAmor/cond_gpt/weights/scaf_tpsa/run.pt --props tpsa --scaffold --data_name datasets/moses_dataset_v2 --csv_name moses_scaf_tpsa_adarsh --gen_size 5000 --batch_size 512 --vocab_size 94 --block_size 54
python generate/generate.py --model_weight /home/furiosa/Documents/Adarsh/PolyAmor/cond_gpt/weights/scaf_sas/run.pt --props sas --scaffold --data_name datasets/moses_dataset_v2 --csv_name moses_scaf_sas_adarsh --gen_size 5000 --batch_size 512 --vocab_size 94 --block_size 54
python generate/generate.py --model_weight /home/furiosa/Documents/Adarsh/PolyAmor/cond_gpt/weights/scaf_logp/run.pt --props logp --scaffold --data_name datasets/moses_dataset_v2 --csv_name moses_scaf_logp_adarsh --gen_size 5000 --batch_size 512 --vocab_size 94 --block_size 54
python generate/generate.py --model_weight /home/furiosa/Documents/Adarsh/PolyAmor/cond_gpt/weights/scaf_qed/run.pt --props qed --scaffold --data_name datasets/moses_dataset_v2 --csv_name moses_scaf_qed_adarsh --gen_size 5000 --batch_size 512 --vocab_size 94 --block_size 54

# Two property and scaffold based generation
# python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_logp.pt --props tpsa logp --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_logp_temp1 --gen_size 10000 --batch_size 512
# python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_sas.pt --props tpsa sas --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_sas_temp1 --gen_size 10000 --batch_size 512
# python generate/generate.py --model_weight moses_scaf_wholeseq_sas_logp.pt --props sas logp --scaffold --data_name moses2 --csv_name moses_scaf_sas_logp_temp1 --gen_size 10000 --batch_size 512

# # Triple property and scaffold based generation
# python generate/generate.py --model_weight moses_scaf_wholeseq_tpsa_logp_sas.pt --props tpsa logp sas --scaffold --data_name moses2 --csv_name moses_scaf_tpsa_logp_sas_temp1 --gen_size 10000 --batch_size 512
