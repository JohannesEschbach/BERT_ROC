# -*- coding: utf-8 -*-
"""StoriesSemantics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/JohannesEschbach/BERT_ROC/blob/main/StoriesSemantics.ipynb
"""

!pip install transformers

from google.colab import drive

current_directory = '/content/drive/My Drive/Semantics/BERT_ROC/'
drive.mount('/content/drive')

"""# Headers and Global Variables"""

import csv
import torch
from torch.nn.functional import softmax
from torch.nn.functional import relu
from transformers import BertForNextSentencePrediction, BertTokenizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from matplotlib import pyplot as plt


device = "cuda:0" if torch.cuda.is_available() else "cpu"

CLOZE_MODEL = 'bertfornsp_cloze_finetuned'
ROC_MODEL = 'bertfornsp_roc_finetuned'
# underlying pretrained LM
BASE_MODEL = 'bert-large-uncased-whole-word-masking'

BATCH_SIZE = 12
WARMUP_EPOCHS = 1
TRAIN_EPOCHS = 10
LAST_EPOCH = -1

"""# Datasets"""

class RocStories(torch.utils.data.Dataset):
    def __init__(self):    
        dataset = []       
        with open(current_directory + 'roc_stories.csv', 
                  'r', encoding='utf-8') as d:
            
            reader = csv.reader(d, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)                
            for line in reader:
                dataset.append(line)  

        self.data = []
        self.labels = []

        stories = []
        endings = []
        for sample in dataset:           
            start = " ".join(sample[2:-1])
            stories.append(start)            
            end = sample[-1]                        
            endings.append(end)

        from random import shuffle
        wrong_endings = endings.copy()
        shuffle(wrong_endings)

        assert len(stories) == len(endings)
        for i, story in enumerate(stories):
            
            #True Ending
            self.data.append([story, endings[i]])
            self.labels.append(0)

            #Wrong Ending
            self.data.append([story, wrong_endings[i]])
            self.labels.append(1)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]        
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)

class ClozeTest(torch.utils.data.Dataset):
    def __init__(self, dev=True, hypothesis_only=False, hard = False):
        """
        :param hypothesis_only: Replaces story with empty string. Only Keeps endings as they are.
        :param hard: For future hard_test_set.csv
        """

        dataset = []

        # if dev=True, we load the dev set for testing
        dir = ""
        if dev:
            if hard:
                dir = current_directory + 'hard_test_set.csv'
            else: 
                dir = current_directory + 'cloze_test.csv'
        else:
            dir = current_directory + 'cloze_train.csv'

        with open(dir, 'r', encoding='utf-8') as d:
            reader = csv.reader(d, quotechar='"', delimiter=',', 
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)                
            for line in reader:
                dataset.append(line) 
            dataset.pop(0)

        self.data = []
        self.labels = []

        for sample in dataset:
            
            start = " ".join(sample[1:-3])
            if hypothesis_only: start = ""
            end1 = sample[-3]
            end2 = sample[-2]
            right_ending = sample[-1]

            self.data.append([start, end1])
            self.labels.append(0 if "1" == right_ending else 1)

            self.data.append([start, end2])
            self.labels.append(0 if "2" == right_ending else 1)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]        
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)

"""# Auxiliary Functions"""

def getModelFileName(model_name, last_epoch):
    return current_directory + model_name + str(last_epoch)

def weight_diff(model1, model2):
    diff = torch.nn.MSELoss() # diff(a, b) = ((a - b) ** 2).mean()

    xweights, yweights, xbiases, ybiases = dict(), dict(), dict(), dict()
    layer_names = set()

    for (name, parameter1), parameter2 in zip(
        model1.bert.encoder.layer.named_parameters(),
        model2.bert.encoder.layer.parameters()
    ):

        difference = diff(parameter1, parameter2).item()

        name = name.split(".")
        xtick = float(name[0])
        layer_name = ".".join(name[1:-1])
        parameter_type = name[-1]

        if layer_name not in layer_names:
            layer_names.add(layer_name)
            xweights[layer_name], xbiases[layer_name] = list(), list()
            yweights[layer_name], ybiases[layer_name] = list(), list()

        if parameter_type == "weight":
            yweights[layer_name].append(difference)
            xweights[layer_name].append(xtick + 0.0)
        else: # if parameter_type == "bias"
            ybiases[layer_name].append(difference)
            xbiases[layer_name].append(xtick + 0.5)

    for name in layer_names:
        plt.bar(xweights[name], yweights[name], width=0.4, label="weight")
        plt.bar(xbiases[name], ybiases[name], width=0.4, label="bias")
        plt.xticks(xweights[name])
        plt.legend()
        plt.title(name)
        plt.show()

"""# Functions for Training and Testing"""

def train(cloze_test, model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False):
    
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    model_old = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Training
    model = model.to(device)
    model.train()

    trainloader = torch.utils.data.DataLoader(
        ClozeTest(dev=False) if cloze_test else RocStories(),
        batch_size=batch_size, shuffle=True
    )

    #LR maybe needs to be optimized
    optimizer = AdamW(model.parameters(), lr=1e-5)
    n_batches =  len(trainloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(warmup_epochs * n_batches),
        num_training_steps=(train_epochs * n_batches),
        last_epoch=max(-1, last_epoch * n_batches) # actually, last_step
    )
    losses = []

    epochs_range = range(last_epoch + 1, train_epochs)
    for epoch in tqdm(epochs_range):
        
        for batchId, (stories, labels) in zip(range(n_batches), trainloader):
            # this is PyTorch-specific as gradients get accumulated        
            optimizer.zero_grad()

            start = stories[0]
            end = stories[1]

            labels = labels.to(device)
           
            # Tokenize sentence pairs.
            # All sequences in batch processing must be same length.
            # Therefore we use padding to fill shorter sequences
            # with uninterpreted [PAD] tokens)
            tokenized_batch = tokenizer(start, padding = True, text_pair = end,
                                        return_tensors='pt').to(device)
            
            loss = model(**tokenized_batch, labels = labels).loss
            if verbose:
                print("Epoch " + str(epoch + 1) + 
                      " Batch " + batchId + " of " + n_batches + 
                      " Loss: " + loss.item())
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step() # Huggingface specific: step = epoch

        model.save_pretrained(
            getModelFileName((CLOZE_MODEL if cloze_test 
                              else ROC_MODEL), epoch + 1)
        )
    
    # Loss function change over steps is plotted below.
    plt.plot(losses)
    plt.xticks(
        ticks=[(i - last_epoch - 1) * n_batches for i in epochs_range],
        labels=epochs_range
    )
    plt.title(("Story Cloze" if cloze_test else "ROCStories") + " Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Models are compared
    weight_diff(model, model_old)

def test(model_file=BASE_MODEL, verbose = False, cloze_test = ClozeTest()):
    softmax = torch.nn.Softmax(dim=1)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Evaluation
    model = model.to(device)
    model.eval()

    #Dataloader
    devloader = torch.utils.data.DataLoader(cloze_test, batch_size=10)

    pred_list, label_list = list(), list()

    for stories, labels in tqdm(devloader, disable=verbose):
        
        start = stories[0]
        end = stories[1]
        
        # Tokenize sentence pairs.
        # All sequences in batch processing must be same length.
        # Therefore we use padding to fill shorter sequences
        # with uninterpreted [PAD] tokens)
        tokenized_batch = tokenizer(start, padding = True, text_pair = end,
                                    return_tensors='pt').to(device)

        #Send to GPU
        labels = labels.to(device)

        outputs = model(**tokenized_batch, labels = labels)
        logits = outputs.logits

        # Model predicts sentence-pair as correct if True-logit > False-logit
        predictions = logits.argmax(dim=1).int()
        probs = softmax(logits).cpu().detach()

        # Extra info print() if verbose
        if verbose:
            # iterate over elements in batch
            for i, element_input_ids in enumerate(tokenized_batch.input_ids):
                print(tokenizer.decode(element_input_ids))
                print("Probability:", probs[i][0].item() * 100)
                print("Predicted: ", bool(predictions[i]))
                print("True label: ", bool(labels[i]))

        pred_list.extend(predictions.tolist())
        label_list.extend(labels.tolist())

    print(confusion_matrix(label_list, pred_list))
    print(classification_report(label_list, pred_list))

    return confusion_matrix(label_list, pred_list).ravel()

"""# Testing Pretrained BertForNextSentencePrediction"""

test()

"""# Training and Testing the Model on ROCStories Dataset"""

train_epochs_roc = 1
#train(train_epochs=train_epochs_roc, cloze_test=False)
test(getModelFileName(ROC_MODEL, train_epochs))

"""# Training and Testing the Model Further on StoryCloze Dataset"""

train_epochs_cloze = 10
#train(model_file=getModelFileName(ROC_MODEL, train_epochs_roc), cloze_test=True)
test(getModelFileName(CLOZE_MODEL, train_epochs_cloze))

"""#Saliency Maps"""

#Old Version. Check Kamal's branch for up to date version

def saliency_map(model, tokenizer, input, ending, label):

    
    # Activations are saved.
    acts = dict() # one-key dictionary. Doesn't work otherwise.
    def get_acts(name):
        def hook(module, input, output):
			      acts[name] = output.detach()
        return hook

	  # Gradients are saved.
    grads = dict() # same as for activations
    def get_grads(name):
		    def hook(module, input, output):
			      grads[name] = output[0].detach() # 'output' is a tuple
		    return hook

    frw_handle = model.bert.embeddings.register_forward_hook(get_acts("emb"))
    bck_handle = model.bert.embeddings.register_backward_hook(get_grads("emb"))
    
    tokens = tokenizer(input, text_pair=ending, return_tensors='pt').to(device)

    token_names = tokenizer.decode(tokens.input_ids[0])


    model.eval()
    model.zero_grad()
    model = model.to(device)

    model(**tokens, labels = torch.tensor(label).to(device)).loss.backward()

    frw_handle.remove()
    bck_handle.remove()

    saliencies = (-grads["emb"] * acts["emb"]).sum(dim=-1)
    saliencies = relu(saliencies) # relu'd saliencies
    saliencies = saliencies / saliencies.max() # normalization
    saliencies = saliencies[0] # squeezing the batch of one
    return list(zip(token_names.split(), saliencies.tolist())) # token-saliency pairs

model = BertForNextSentencePrediction.from_pretrained(getModelFileName(CLOZE_MODEL, "10"))
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

cloze_test = ClozeTest(dev=True)
story, label = cloze_test[43] #Example data point
input, ending = story

saliency_map(model = model, tokenizer=tokenizer, input = input, ending = ending, label = label)

"""#Trigger Words"""

def vocab_distribution(dev_only=True, train_only=False, hard = True, token_ids = False):
    """    
    :param token_ids: Return words when False, token_ids when True
    :param dev_only: Identify words with high class likelihood in test-set endings
    :param train_only: Identify words with high class likelihood in train-set endings (this is where the model gets biased). 
    :param hard: Use hard test-set (Doesnt exist yet)
    """
    data = []
    labels = []
    
    if dev_only:
        cloze = ClozeTest(dev=False, hard = hard)
        data.extend(cloze.data)
        labels.extend(cloze.labels)

    if train_only:
        clozedev = ClozeTest(dev=True)
        data.extend(clozedev.data)
        labels.extend(clozedev.labels)
    
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)


    ending_tokens = {}
    word_count = 0

    for i, story in enumerate(data):
        label = labels[i]
        end = story[1]    
        tokens = tokenizer(end).input_ids
        tokens.pop(0)
        tokens.pop(-1)
        
        for token in tokens:
            if not token_ids: token = tokenizer.decode(token).replace(" ", "")
            word_count += 1
            if token not in ending_tokens:
                ending_tokens[token] = [0,0]
            ending_tokens[token][label] += 1

    return ending_tokens, word_count

def pmi(class_count, other_class_count, word_count):
    """
    :param class_count: Number of occurences in the class you want to calculate the pmi with
    :param other_class_count: Number of occurences in the other class
    :param word_count: Total word count
    """
    import math
    if class_count < 1:
        return 0
    return math.log((class_count / word_count) / ((class_count + other_class_count)/(word_count*2)))

def class_prob(class_count, other_class_count):
    return class_count/(class_count + other_class_count)

def get_trigger_words(hard = False, dev_only = True, train_only = True, min_occurences = 30, token_ids = False):
    """    
    :param token_ids: Return words when False, token_ids when True
    :param dev_only: Identify words with high class likelihood in test-set endings
    :param train_only: Identify words with high class likelihood in train-set endings (this is where the model gets biased). 
    :param hard: Use hard test-set (Doesnt exist yet)
    :param min_occunrences: Only return trigger words minimally occuring this often
    """
    
    vocab_dis, word_count = vocab_distribution(dev_only=dev_only, train_only=train_only, hard=hard, token_ids=token_ids)

    pos_triggers = []
    neg_triggers = []

    for word, dis in vocab_dis.items():
        if(dis[0]+dis[1] >= min_occurences):      
            pmi_pos = pmi(dis[0], dis[1], word_count)    
            pmi_neg = pmi(dis[1], dis[0], word_count)

            class_prob_pos = class_prob(dis[0], dis[1])
            class_prob_neg = class_prob(dis[1], dis[0])

            pos_triggers.append([word, dis[0], pmi_pos, class_prob_pos])
            neg_triggers.append([word, dis[1], pmi_neg, class_prob_neg])

    pos_triggers.sort(key=lambda x: x[2], reverse = True)
    neg_triggers.sort(key=lambda x: x[2], reverse = True)


    from tabulate import tabulate
    print(tabulate(pos_triggers[:100], headers=['Token', 'n', 'pmi', 'pos_class_likelihood']))
    print("\n")
    print(tabulate(neg_triggers[:100], headers=['Token', 'n', 'pmi', 'neg_class_likelihood']))

    return pos_triggers, neg_triggers

get_trigger_words(hard = False, dev_only = False, train_only = True, min_occurances = 10, token_ids = False)

"""#Hard Testset (OUTDATED, SOME TRIES, NOTHING RELEVANT, MAYBE GOOD FOR COPYING SOME CODE FOR FUTURE PURPOSES)"""

def eliminate_hyp_only_stories(cloze_test = ClozeTest(dev=True, hypothesis_only=True)):
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(getModelFileName(CLOZE_MODEL, 10))

    #Send to GPU and allow Evaluation
    model = model.to(device)
    model.eval()
   
    labels = cloze_test.labels

    biased = []

    for i, sample in enumerate(cloze_test.data):
        label = torch.LongTensor([labels[i]])
        label = label.to(device)
        certainty = predict("", sample[1], label, tokenizer, model)
        biased.append([i, certainty]) #append index

    biased.sort(key=lambda x: x[1], reverse = True)

    from tabulate import tabulate
    print(tabulate(biased, headers=['Story', 'certainty']))


    biased_endings = [elem[0] for elem in biased]

    biased_stories = []
    for i in biased_endings:
        biased_stories.append(i)
        if i % 2 == 0: biased_stories.append(i+1)
        else: biased_stories.append(i-1)

    return biased_stories

def remove_words(bias_tokens, file):
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    with open(current_directory + 'hard_test_set.csv', 'a') as f:
        with open(current_directory + file, 'r', encoding='utf-8') as d:
            reader = csv.reader(d, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)                
            for sample in reader:
                ends = [sample[-3],sample[-2]]
                new_ends = []
                for end in ends:
                    tokens = tokenizer(end).input_ids

                    reduced_end = []
                    for token in tokens:
                        if token not in bias_tokens:
                            word = tokenizer.decode(token).replace(" ", "")   
                            reduced_end.append(word)
                    new_ends.append(" ".join(reduced_end))
                #...


                    
                    
            
                



    return

def create_reduced_testset(remove_ind):
    with open(current_directory + 'hard_test_set.csv', 'a') as f:
    with open(current_directory + 'cloze_test.csv', 'r', encoding='utf-8') as d:
        lines = d.readlines()
        lines.pop(0)
        f.write("header line \n")
        for i, line in enumerate(lines):
            if i * 2 not in remove_ind:
                f.write(line)

triggers = get_trigger_words(hard = False, dev_only = False, train_only = True, min_occurances = 3, token_ids = True)

story_indices = eliminate_hyp_only_stories()

cloze_test = ClozeTest(dev=True,hypothesis_only=True)
remove_ind = story_indices[:2000]
cloze_test.data = [x for i, x in enumerate(cloze_test.data) if i not in remove_ind]
cloze_test.labels = [x for i, x in enumerate(cloze_test.labels) if i not in remove_ind]

#tn, fp, fn, tp = test(model_file = getModelFileName(CLOZE_MODEL, "10"), cloze_test=cloze_test)

create_reduced_testset(remove_ind)

cloze_test_hard = ClozeTest(dev=True, hard=True, hypothesis_only=True)
test(model_file = getModelFileName(CLOZE_MODEL, "10"), cloze_test=cloze_test_hard)
cloze_test_hard = ClozeTest(dev=True, hard=True)
test(model_file = getModelFileName(CLOZE_MODEL, "10"), cloze_test=cloze_test_hard)

def train_bias_reduced(cloze_test, model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False):
    
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    model_old = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Training
    model = model.to(device)
    model.train()

    trainloader = torch.utils.data.DataLoader(
        ClozeTest(dev=False) if cloze_test else RocStories(),
        batch_size=batch_size, shuffle=True
    )

    #LR maybe needs to be optimized
    optimizer = AdamW(model.parameters(), lr=1e-5)
    n_batches =  len(trainloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(warmup_epochs * n_batches),
        num_training_steps=(train_epochs * n_batches),
        last_epoch=max(-1, last_epoch * n_batches) # actually, last_step
    )
    losses = []

    epochs_range = range(last_epoch + 1, train_epochs)
    for epoch in tqdm(epochs_range):
        
        for batchId, (stories, labels) in zip(range(n_batches), trainloader):
            # this is PyTorch-specific as gradients get accumulated        
            optimizer.zero_grad()

            start = stories[0]
            end = stories[1]

            labels = labels.to(device)
           
            # Tokenize sentence pairs.
            # All sequences in batch processing must be same length.
            # Therefore we use padding to fill shorter sequences
            # with uninterpreted [PAD] tokens)
            tokenized_batch = tokenizer(start, padding = True, text_pair = end,
                                        return_tensors='pt').to(device)
            
            loss = model(**tokenized_batch, labels = labels).loss
            if verbose:
                print("Epoch " + str(epoch + 1) + 
                      " Batch " + batchId + " of " + n_batches + 
                      " Loss: " + loss.item())
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step() # Huggingface specific: step = epoch

        model.save_pretrained(
            getModelFileName((CLOZE_MODEL if cloze_test 
                              else ROC_MODEL), epoch + 1)
        )
    
    # Loss function change over steps is plotted below.
    plt.plot(losses)
    plt.xticks(
        ticks=[(i - last_epoch - 1) * n_batches for i in epochs_range],
        labels=epochs_range
    )
    plt.title(("Story Cloze" if cloze_test else "ROCStories") + " Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Models are compared
    weight_diff(model, model_old)