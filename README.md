
TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
====================

**TAT-QA** (**T**abular **A**nd **T**extual dataset for **Q**uestion **A**nswering) contains 16,552 questions associated with 2,757 hybrid contexts 
from real-world financial reports. 

You can download our TAT-QA dataset via [TAT-QA dataset](https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw).
                
For more information, please refer to our [TAT-QA website](https://nextplusplus.github.io/TAT-QA/) or read our ACL2021 paper [PDF](https://arxiv.org/pdf/2105.07624.pdf).

## TagOp Model

### Requirements

To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n tat-qa python==3.7
conda activate tat-qa
pip install -r requirement.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
```

We adopt `RoBERTa` as our encoder to develop our TagOp and use the following commands to prepare RoBERTa model 

```bash
cd dataset_tagop
mkdir roberta.large && cd roberta.large
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
```

### Training & Testing

#### Preprocessing dataset

We heuristicly generate the "facts" and "mapping" fields based on raw dataset, which are stored under the folder of `dataset_tagop`.


#### Prepare dataset

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --mode [train/dev/test]
```

Note: The result will be written into the folder `./tag_op/cache` default.

#### Train & Evaluation 
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/trainer.py --data_dir tag_op/cache/ \
--save_dir ./checkpoint --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 \
--weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 \
--log_per_updates 50 --eps 1e-6 --encoder roberta
```

#### Testing
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/cache/ --test_data_dir tag_op/cache/ \\
--save_dir tag_op/ --eval_batch_size 32 --model_path ./checkpoint --encoder roberta
```

Note: The training process may take around 2 days using a single 32GB v100.

### Citation
```bash
@misc{zhu2021tatqa,
      title={TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance}, 
      author={Fengbin Zhu and Wenqiang Lei and Youcheng Huang and Chao Wang and Shuo Zhang and Jiancheng Lv and Fuli Feng and Tat-Seng Chua},
      year={2021},
      eprint={2105.07624},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Any Question?

For any issues please create an issue [here](https://github.com/nextplusplus/tat-qa/issues) or kindly email us at:
Youcheng Huang [1361881994@qq.com](mailto:1361881994@qq.com) or Fengbin Zhu [zhfengbin@gmail.com](mailto:zhfengbin@gmail.com), thank you.
