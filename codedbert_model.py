import torch, torchvision
import sys
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from utils import construct_bert_input, MultiModalBertDataset, PreprocessedADARI, save_json
from transformers import get_linear_schedule_with_warmup
import argparse
import datetime
from codedbert_adaptive_loss import adaptive_loss, adaptive_loss_4losses
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

class codedbertHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.im_dense = torch.nn.Linear(768, 2048)
        self.act_func = torch.nn.LeakyReLU()
        self.layer_norm = torch.nn.LayerNorm(2048, eps=config.layer_norm_eps)

    def forward(self, im_output):
        batch_size = im_output.shape[0]
        seq_len = im_output.shape[1]

        h = self.im_dense(im_output.reshape(-1, im_output.shape[2]))
        h = self.act_func(h)
        h = self.layer_norm(h)

        return h.view(batch_size, seq_len, -1)

    
class codedbertSentenceHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sent_cls = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, sent_output):
        return self.sent_cls(sent_output)  
    
class codedbert(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
   
        self.bert = BertModel(config)
    
        self.im_type_em = torch.nn.Embedding(1, config.hidden_size)
        self.im_position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        #self.sentence_alignment = torch.nn.Linear(config.hidden_size, 2)
        self.sentence_alignment = codedbertSentenceHead(config)
        self.cls = BertPreTrainingHeads(config)
        self.im_head = codedbertHead(config)

        self.init_weights()

    def forward(
        self,
        embeds,
        attention_mask,
        labels=None,
        unmasked_patch_features=None,
        is_paired=None,
    ):
        """
            Args: 
                embeds
                    hidden embeddings to pass to the bert model
                        batch size, seq length, hidden dim
                attention_mask
                    batch size, seq length
                labels
                    Unmasked tokenized token ids
                        batch size, word sequence length
                unmasked_patch_features
                    Unmasked image features
                        batch size, image sequence length, image embedding size
                is_paired
                    bool tensor, Whether the sample is aligned with the sentence
                        batch size, 1
        """
        batch_size = embeds.shape[0]
        seq_length = embeds.shape[1]
        hidden_dim = embeds.shape[2]

        outputs = self.bert(inputs_embeds=embeds, 
                            attention_mask=attention_mask,
                            return_dict=True)

        sequence_output = outputs.last_hidden_state
        pooler_output = outputs.pooler_output                 #[batch, 768]
       
        # hidden states corresponding to the text part
        text_output = sequence_output[:, :labels.shape[1], :] #[batch, 432, 768]

        # hidden states corresponding to the sentence part
        sentence_output = sequence_output[:, labels.shape[1]:labels.shape[1]+16, :] #[batch, 16, 768]

        # hidden states corresponding to the image part
        image_output = sequence_output[:, labels.shape[1]+16:, :] #[batch, 64, 768]

        # Predict the masked text tokens and alignment scores (whether image, text match)
        prediction_scores, alignment_scores = self.cls(text_output, pooler_output) # aligment scores [batch, 2]
        im_scores = self.im_head(image_output)

        # Predict sentence alignemnt
        sentence_alignment_pred = self.sentence_alignment(sentence_output).mean(1) # [batch, 16, 2] after mean [batch, 2]
        
        # We only want to compute masked losses w.r.t. aligned samples
        pred_scores_aligned = prediction_scores[is_paired.view(-1)]
        labels_aligned = labels[is_paired.view(-1)]
    
        # Compute masked language loss
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        if labels_aligned.shape[0] > 0:
            masked_lm_loss = loss_fct(pred_scores_aligned.view(-1, self.config.vocab_size), labels_aligned.view(-1))
        else:
            masked_lm_loss = torch.tensor(0.0).to(self.device)

        # Compute masked patch reconstruction loss
        # Only look at aligned images
        image_output_aligned = im_scores[is_paired.view(-1)]
        if image_output_aligned.shape[0] > 0:
            unmasked_patch_features_aligned = unmasked_patch_features[is_paired.view(-1)]

            pred_probs = torch.nn.LogSoftmax(dim=1)(image_output_aligned.view(-1, unmasked_patch_features_aligned.shape[2]))
            true_probs = torch.nn.Softmax(dim=1)(unmasked_patch_features_aligned.view(-1, unmasked_patch_features_aligned.shape[2]))
            loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
            masked_patch_loss = loss_fct(pred_probs, true_probs)
        else:
            masked_patch_loss = torch.tensor(0.0).to(self.device)
        
        # Compute alignment loss for text
        loss_fct = torch.nn.CrossEntropyLoss()
        alignment_loss_text = loss_fct(alignment_scores.view(-1, 2), is_paired.long().view(-1))

        # Compute alignment loss for sentence
        alignment_loss_sent = loss_fct(sentence_alignment_pred.view(-1, 2), is_paired.long().view(-1))

        return {
            "raw_outputs": outputs, 
            "masked_lm_loss": masked_lm_loss,
            "masked_patch_loss": masked_patch_loss,
            "alignment_loss_text": alignment_loss_text,
            "alignment_loss_sent": alignment_loss_sent,
            }

def train(coded_bert, dataset, params, device, random_patches=False):
    print('Random patches? ', random_patches)
    torch.manual_seed(0)
    train_size = int(len(dataset) * .8)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=params.batch_size, 
        shuffle=True,
        )

    coded_bert.to(device)
    coded_bert.train()
    opt = transformers.Adafactor(
        coded_bert.parameters(), 
        lr=params.lr, 
        beta1=params.beta1, 
        weight_decay=params.weight_decay,
        clip_threshold=params.clip,
        relative_step=False,
        scale_parameter=True,
        warmup_init=False
        )

    scheduler = get_linear_schedule_with_warmup(opt, params.num_warmup_steps, params.num_epochs * len(dataloader))

    for ep in range(params.num_epochs):
        avg_losses = {"masked_lm_loss": [], "masked_patch_loss": [], "alignment_loss_text": [], "alignment_loss_sent": [], "total": []}
        for patches, input_ids, is_paired, attention_mask, sents_embeds in dataloader:
            # sents_embeds = [batch, 16, 768]
            opt.zero_grad()

            # mask image patches with prob 10%
            im_seq_len = patches.shape[1]
            masked_patches = patches.detach().clone()
            masked_patches = masked_patches.view(-1, patches.shape[2])
            im_mask = torch.rand((masked_patches.shape[0], 1)) >= 0.1
            masked_patches *= im_mask
               
            try:
                masked_patches = masked_patches.view(params.batch_size, im_seq_len, patches.shape[2])
            except Exception as e:
                print(e)
                print(f"masked_patches: {masked_patches.shape}")
                print(f"im_mask: {im_mask.shape}")
                print(f"patches: {patches.shape}")
                continue

            # mask tokens with prob 15%, note id 103 is the [MASK] token
            token_mask = torch.rand(input_ids.shape)
            masked_input_ids = input_ids.detach().clone()
            masked_input_ids[token_mask < 0.15] = 103

            input_ids[token_mask >= 0.15] = -100
            input_ids[attention_mask == 0] = -100
            
            masked_patches = masked_patches.to(device)
            embeds = construct_bert_input(masked_patches, masked_input_ids, coded_bert, sentences=sents_embeds, device=device, random_patches=random_patches)
            # pad attention mask with 1s so model pays attention to the image parts
            attention_mask = F.pad(attention_mask, (0, embeds.shape[1] - input_ids.shape[1]), value = 1) # still works with sentences since we are using 1s for sent and imgs

            outputs = coded_bert(
                embeds=embeds.to(device), 
                attention_mask=attention_mask.to(device), 
                labels=input_ids.to(device), 
                unmasked_patch_features=patches.to(device), 
                is_paired=is_paired.to(device)
                )
            
            loss = adaptive_loss_4losses(outputs)

            loss.backward()
            opt.step()
            scheduler.step()

            for k, v in outputs.items():
                if k in avg_losses:
                    avg_losses[k].append(v.cpu().item())
            avg_losses["total"].append(loss.cpu().item())
        
        print("***************************")
        print(f"At epoch {ep+1}, losses: ")
        for k, v in avg_losses.items():
            print(f"{k}: {sum(v) / len(v)}")
        print("***************************")

class TrainParams:
    lr = 2e-5
    batch_size = 4
    beta1 = 0.95
    beta2 = .999
    weight_decay = 1e-4
    num_warmup_steps = 5000
    num_epochs = 10
    clip = 1.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train codedbert.')
    #parser.add_argument('--path_to_images', help='Absolute path to image directory', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/v2/full')
    #parser.add_argument('--path_to_data_dict', help='Absolute path to json containing img name, sentence pair dict', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/ADARI_furniture_pairs.json')
    parser.add_argument('--path_to_dataset', help='Absolute path to .pkl file', default='/home/ubuntu/preprocessed_patches.pkl')
    parser.add_argument('--random_patches', help='Whether the dataset patches are random, 1 if so, 0 if not random')
    args = parser.parse_args()

    params = TrainParams()

    coded_bert = codedbert.from_pretrained('bert-base-uncased', return_dict=True)
    #dataset = MultiModalBertDataset(
    #    args.path_to_images, 
    #    args.path_to_data_dict,
    #    device=device,
    #    )
    dataset = PreprocessedADARI(args.path_to_dataset)

    try:
        train(coded_bert, dataset, params, device, args.random_patches)
    except KeyboardInterrupt:
        pass
    model_time = datetime.datetime.now().strftime("%X")
    model_name = f"codedbert_{model_time}"
    print(f"Saving trained model to directory {model_name}...")
    coded_bert.save_pretrained(model_name)
    save_json(f"{model_name}/train_params.json", params.__dict__)
