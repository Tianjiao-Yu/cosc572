import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu, tanh, sigmoid
from pytorch_pretrained_bert import BertModel, modeling, TransfoXLModel, TransfoXLLMHeadModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from data_load import num_task, VOCAB, masking, hier, sig, rel

##
from transformers import RobertaModel, GPT2LMHeadModel
from pytorch_pretrained_bert.modeling_transfo_xl import TransfoXLPreTrainedModel
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel


class BertMultiTaskLearning(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, len(VOCAB[i])) for i in range(num_task)])
        #self.classifier = nn.ModuleList([nn.Linear(config.n_embd, len(VOCAB[i])) for i in range(num_task)])


        self.apply(self.init_bert_weights)
        self.masking_gate = nn.Linear(2, 1)

        ##### to do: improve network ############
        #self.masking_gate2  = nn.LSTM(2,1)


        if num_task == 2:
            self.merge_classifier_1 = nn.Linear(len(VOCAB[0])+len(VOCAB[1]), len(VOCAB[0]))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        if num_task == 1:
            logits = [self.classifier[i](sequence_output) for i in range(num_task)]

        elif num_task == 2 and masking:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)

            if sig:
                gate = sigmoid(self.masking_gate(sen_level))
                #gate = sigmoid(self.masking_gate2(self.masking_gate(sen_level)))
            else:
                gate = relu(self.masking_gate(sen_level))

            dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
            wei_token_level = torch.mul(dup_gate, token_level)

            logits = [wei_token_level, sen_level]

        elif num_task == 2 and hier:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            ##
            third_feature = self.classifier[1]

            dup_sen_level = sen_level.repeat(1, token_level.size()[1])
            dup_sen_level = dup_sen_level.view(sen_level.size()[0], -1, sen_level.size()[-1])
            logits = [self.merge_classifier_1(torch.cat((token_level, dup_sen_level),2)), self.classifier[1](pooled_output)]

        elif num_task == 2:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            logits = [token_level, sen_level]

        y_hats = [logits[i].argmax(-1) for i in range(num_task)]

        #print("logts:", logits)
        #print("logits shape:", logits[0].size())

        return logits, y_hats


class JointBertLearning(BertPreTrainedModel):
    def __init__(self, config):
        super(JointBertLearning, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.ModuleList([nn.RNN(config.hidden_size, len(VOCAB[i]), num_layers=3) for i in range(num_task)])
        self.classifier = nn.ModuleList([nn.RNN(config.hidden_size, len(VOCAB[i]), 2) for i in range(num_task)])

        self.apply(self.init_bert_weights)
        self.masking_gate = nn.Linear(2, 1)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        logits = [self.classifier[i](sequence_output) for i in range(num_task)]

        y_hats = [logits[i][0].argmax(-1) for i in range(num_task)]

        #print("logts:", logits)
        #print("logits shape:", logits[0].size())

        return logits, y_hats
