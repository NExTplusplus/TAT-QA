import os
import pickle
import torch
import random

class TaTQABatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            operator_labels = torch.tensor(item["operator_label"])
            scale_labels = torch.tensor(item["scale_label"])
            number_order_labels = torch.tensor(item["number_order_label"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                table_cell_index, tag_labels, operator_labels, scale_labels, number_order_labels, gold_answers,
                paragraph_tokens, table_cell_tokens, paragraph_numbers, table_cell_numbers, question_id))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQABatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                              self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, operator_labels_batch, scale_labels_batch, \
            number_order_labels_batch, gold_answers_batch, paragraph_tokens_batch,  \
            table_cell_tokens_batch, paragraph_numbers_batch, table_cell_numbers_batch, question_ids_batch = zip(*batch)
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            operator_labels = torch.LongTensor(bsz)
            scale_labels = torch.LongTensor(bsz)
            number_order_labels = torch.LongTensor(bsz)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []
            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                operator_labels[i] = operator_labels_batch[i]
                scale_labels[i] = scale_labels_batch[i]
                number_order_labels[i] = number_order_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids,
                "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                "operator_labels": operator_labels, "scale_labels": scale_labels, "number_order_labels": number_order_labels,
                "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens, "paragraph_numbers": paragraph_numbers,
                "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                "table_mask": table_mask, "table_cell_index":table_cell_index,
            }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch

class TaTQATestBatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta'):
        dpath =  f"tagop_{encoder}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        print(os.path.join(args.test_data_dir, dpath))
        with open(os.path.join(args.test_data_dir, dpath), 'rb') as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])
            paragraph_mask = torch.from_numpy(item["paragraph_mask"])
            table_mask = torch.from_numpy(item["table_mask"])
            paragraph_numbers = item["paragraph_number_value"]
            table_cell_numbers = item["table_cell_number_value"]
            paragraph_index = torch.from_numpy(item["paragraph_index"])
            table_cell_index = torch.from_numpy(item["table_cell_index"])
            tag_labels = torch.from_numpy(item["tag_labels"])
            gold_answers = item["answer_dict"]
            paragraph_tokens = item["paragraph_tokens"]
            table_cell_tokens = item["table_cell_tokens"]
            question_id = item["question_id"]
            all_data.append((input_ids, attention_mask, token_type_ids, paragraph_mask, table_mask, paragraph_index,
                             table_cell_index, tag_labels, gold_answers, paragraph_tokens, table_cell_tokens,
                             paragraph_numbers, table_cell_numbers, question_id))
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQATestBatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                               self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch, attention_mask_batch, token_type_ids_batch, paragraph_mask_batch, table_mask_batch, \
            paragraph_index_batch, table_cell_index_batch, tag_labels_batch, gold_answers_batch, paragraph_tokens_batch, \
            table_cell_tokens_batch, paragraph_numbers_batch, table_cell_numbers_batch, question_ids_batch = zip(*batch)
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, 512)
            attention_mask = torch.LongTensor(bsz, 512)
            token_type_ids = torch.LongTensor(bsz, 512).fill_(0)
            paragraph_mask = torch.LongTensor(bsz, 512)
            table_mask = torch.LongTensor(bsz, 512)
            paragraph_index = torch.LongTensor(bsz, 512)
            table_cell_index = torch.LongTensor(bsz, 512)
            tag_labels = torch.LongTensor(bsz, 512)
            paragraph_tokens = []
            table_cell_tokens = []
            gold_answers = []
            question_ids = []
            paragraph_numbers = []
            table_cell_numbers = []

            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]
                paragraph_mask[i] = paragraph_mask_batch[i]
                table_mask[i] = table_mask_batch[i]
                paragraph_index[i] = paragraph_index_batch[i]
                table_cell_index[i] = table_cell_index_batch[i]
                tag_labels[i] = tag_labels_batch[i]
                paragraph_tokens.append(paragraph_tokens_batch[i])
                table_cell_tokens.append(table_cell_tokens_batch[i])
                paragraph_numbers.append(paragraph_numbers_batch[i])
                table_cell_numbers.append(table_cell_numbers_batch[i])
                gold_answers.append(gold_answers_batch[i])
                question_ids.append(question_ids_batch[i])
            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "paragraph_mask": paragraph_mask, "paragraph_index": paragraph_index, "tag_labels": tag_labels,
                         "paragraph_tokens": paragraph_tokens, "table_cell_tokens": table_cell_tokens,
                         "paragraph_numbers": paragraph_numbers,
                         "table_cell_numbers": table_cell_numbers, "gold_answers": gold_answers, "question_ids": question_ids,
                         "table_mask": table_mask, "table_cell_index": table_cell_index,
                         # "paragraph_mapping_content": paragraph_mapping_content,
                         # "table_mapping_content": table_mapping_content,
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield  out_batch