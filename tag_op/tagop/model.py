import torch
import torch.nn as nn
from tag_op.tagop.optimizer import BertAdam as Adam
from tag_op.tagop.util import AverageMeter
from tqdm import tqdm


class TagopPredictModel():
    def __init__(self, args, network):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network

        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    @torch.no_grad()
    def evaluate(self, dev_data_list, epoch):
        dev_data_list.reset()
        self.network.eval()
        for batch in tqdm(dev_data_list):
            output_dict = self.network(**batch, mode="eval", epoch=epoch)
            loss = output_dict["loss"]
            self.dev_loss.update(loss.item(), 1)
        self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list):
        test_data_list.reset()
        self.network.eval()
        pred_json = {}
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(**batch, mode="eval")
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            question_id = output_dict["question_id"]
            for i in range(len(question_id)):
                pred_json[question_id[i]] = [pred_answer[i], pred_scale[i]]
        return pred_json

    def reset(self):
        self.mnetwork.reset()

    def get_df(self):
        return self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        return self.mnetwork.get_metrics(logger, True)


class TagopFineTuningModel():
    def __init__(self, args, network, state_dict=None, num_train_steps=1):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])
        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.named_parameters() if not n.startswith("encoder.")],
             "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]
        self.optimizer = Adam(optimizer_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup,
                              t_total=num_train_steps,
                              max_grad_norm=args.grad_clipping,
                              schedule=args.warmup_schedule)
        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    def update(self, tasks):
        self.network.train()
        output_dict = self.mnetwork(**tasks)
        loss = output_dict["loss"]
        self.train_loss.update(loss.item(), 1)
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()
        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_data_list, epoch):
        dev_data_list.reset()
        self.network.eval()
        for batch in dev_data_list:
            output_dict = self.network(**batch, mode="eval", epoch=epoch)
            loss = output_dict["loss"]
            self.dev_loss.update(loss.item(), 1)
        self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list):
        test_data_list.reset()
        self.network.eval()
        # pred_json = {}
        for batch in tqdm(test_data_list):
            self.network.predict(**batch, mode="eval")

    def reset(self):
        self.mnetwork.reset()

    def get_df(self):
        return self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        return self.mnetwork.get_metrics(logger, True)

    def save(self, prefix, epoch):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        other_params = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': epoch
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print('model saved to {}'.format(prefix))