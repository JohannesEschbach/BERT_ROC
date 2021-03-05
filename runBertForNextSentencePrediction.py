import csv
import torch
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# underlying pretrained LM
BASE_MODEL = 'bert-large-uncased-whole-word-masking'


class BertForMultiChoiceNSP(BertForNextSentencePrediction):
    def __init__(self, config):
        super().__init__(config)
        
        self.loss_fct = torch.nn.BCEWithLogitsLoss() # seems to be default loss function for NSP
        self.softmax = torch.nn.Softmax(dim=0)
        self.tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

    def forward(self, input_ids, token_type_ids = None, labels=None, attention_mask=None, verbose=False):
        
        # Parent class provides all needed output
        output = super().forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = output[0]
        logits = output[1]        
                
        predictions = list()
        # iterate over elements in batch
        for i, element_input_ids in enumerate(input_ids):

            # Model predicts sentence-pair as correct if True-logit > False-logit
            predictions.append(int(logits[i][0] > logits[i][1]))
            
            # Extra info print() if verbose
            if verbose:
                prob = self.softmax(logits[i])
                prob = prob.cpu().detach().numpy()
                print(self.tokenizer.decode(element_input_ids))
                print("Probability:", prob[0]*100)
                print("Predicted: ", bool(predictions[i]))
                print("True label: ", bool(labels[i]))
             
        return loss, predictions


class ClozeTest(torch.utils.data.Dataset):
    def __init__(self, dev=True):
        
        dataset = []

        # if dev=True, we load the dev set for testing
        if dev:
            with open('cloze_test.csv', 'r', encoding='utf-8') as d:
                reader = csv.reader(d, quotechar='"', delimiter=',' , quoting=csv.QUOTE_ALL, skipinitialspace=True)                
                for line in reader:
                    dataset.append(line)                

        self.data = []
        self.labels = []


        #i = 0 #for debugging purposes
        for sample in dataset:
            
            #shorter dataset for debugging
            #if i > 200: break
            #i += 1
            
            start = " ".join(sample[1:-3])
            end1 = sample[-3]
            end2 = sample[-2]
            right_ending = sample[-1]

            self.data.append([start, end1])
            self.labels.append(int(right_ending == "1"))

            self.data.append([start, end2])
            self.labels.append(int(right_ending == "2"))

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]        
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)



def test(verbose = False):
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForMultiChoiceNSP.from_pretrained(BASE_MODEL)

    #Send to GPU and allow Evaluation
    model = model.to(device)
    model.eval()

    #Dataloader
    devloader = torch.utils.data.DataLoader(ClozeTest(), batch_size=10)

    pred_list, label_list = list(), list()

    for stories, labels in tqdm(devloader, disable=verbose):
        
        start = stories[0]
        end = stories[1]
        
        #Tokenize sentence pairs. (All sequences in batch processing must be same length. Therefore we use padding to fiill shorter sequences with uninterpreted [PAD] tokens)
        tokenized_batch = tokenizer(start, padding = True, text_pair = end, return_tensors='pt').to(device)        

        input_ids = tokenized_batch['input_ids'] #IDs of Tokens
        token_type_ids = tokenized_batch['token_type_ids'] #1 if Token in second sentence, otherwise 0
        attention_mask = tokenized_batch['attention_mask'] #0 if [PAD] token, 1 otherwise
        
        #Send to GPU
        labels.to(device)

        loss, predictions = model(input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, labels = labels, verbose=verbose)

        pred_list.extend(predictions)
        label_list.extend(labels.tolist())

    print(confusion_matrix(label_list, pred_list))
    print(classification_report(label_list, pred_list))

if __name__ == "__main__":
    test(verbose=False)