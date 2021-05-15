
TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
====================

**TAT-QA** (**T**abular **A**nd **T**extual dataset for **Q**uestion **A**nswering) contains 16,552 questions associated with 2,757 hybrid contexts 
from real-world financial reports. For more details, please read our ACL 2021 paper [PDF]().
                
## The Dataset

### Data Description
```json
{
  "table": {
    "uid": "3ffd9053-a45d-491c-957a-1b2fa0af0570",
    "table": [
      [
        "",
        "2019",
        "2018",
        "2017"
      ],
      [
        "Fixed Price",
        "$  1,452.4",
        "$  1,146.2",
        "$  1,036.9"
      ],
      [
        "..."
      ]
    ]
  },
  "paragraphs": [
    {
      "uid": "f4ac7069-10a2-47e9-995c-3903293b3d47",
      "order": 1,
      "text": "Sales by Contract Type: Substantially all of our contracts are fixed-price type contracts. Sales included in Other contract types represent cost plus and time and material type contracts."
    },
    {
      "uid": "79e37805-6558-4a8c-b033-32be6bffef48",
      "order": 2,
      "text": "On a fixed-price type contract, ... The table below presents total net sales disaggregated by contract type (in millions)"
    }
  ],
  "questions": [
    {
      "uid": "f4142349-eb72-49eb-9a76-f3ccb1010cbc",
      "order": 1,
      "question": "In which year is the amount of total sales the largest?",
      "answer": [
        "2019"
      ],
      "derivation": "",
      "answer_type": "span",
      "answer_from": "text",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": true,
      "scale": ""
    },
    {
      "uid": "eb787966-fa02-401f-bfaf-ccabf3828b23",
      "order": 2,
      "question": "What is the change in Other in 2019 from 2018?",
      "answer": -12.6,
      "derivation": "44.1 - 56.7",
      "answer_type": "arithmetic",
      "answer_from": "table",
      "rel_paragraphs": [
        "2"
      ],
      "req_comparison": false,
      "scale": "million"
    }
  ]
}
```

- `table`: the tabular data in a hybrid context.
  - `uid`: the unique id of a table.
  - `table`: a 2d-array of the table content.
  
- `paragraphs`:the textual data in a hybrid context, the associated paragraphs to the table.
  - `uid`: the unique id of a paragraph.
  - `order`: the order of the paragraph in all associated paragraphs, starting from 1.
  - `text`: the content of the paragraph.
  
- `questions`: the generated questions according to the hybrid context.
  - `uid`: the unique id of a question. 
  - `order`: the order of the question in all generated questions, starting from 1.
  - `question`: the question itself.
  - `answer` : the ground-truth answer.
  - `derivation`: the derivation that can be executed to arrive at the ground-truth answer.
  - `answer_type`: the answer type, including `span`, `spans`, `arithmetic` and `counting`.
  - `answer_from`: the source of the answer, including `table`, `table` and `table-text`.
  - `rel_paragraphs`: the paragraphs that are relied to infer the answer if any.
  - `req_comparison`: a flag indicating if `comparison/sorting` is needed to arrive at the answer (`span` or `spans`).
  - `scale`: the scale of the answer, including `None`, `thousand`, `million`, `billion` and `percent`.

Please find our TAT-QA dataset under the folder `dataset_raw`.

## TagOp Model

## Requirements

To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n tat-qa python==3.7
conda activate tat-qa
pip install -r requirement.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
```

We adopt `RoBERTa` as our encoder to develop our TagOp and use the following commands to prepare RoBERTa model 

```bash
cd dataset_togop
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
{

}
```

### Any Question?

For any issues please create an issue [here](https://github.com/fengbinzhu/tat-qa/issues) or kindly email us at:
Youcheng Huang [1361881994@qq.com](mailto:1361881994@qq.com) or Fengbin Zhu [zhfengbin@gmail.com](mailto:zhfengbin@gmail.com), thank you.
