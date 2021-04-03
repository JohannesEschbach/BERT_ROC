import csv
import os
import torch
from transformers import BertForNextSentencePrediction, BertTokenizer
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from matplotlib import pyplot as plt

if "ROC_DATA_DIR" in os.environ:
    current_directory = os.environ["ROC_DATA_DIR"]
else:
    current_directory = os.getcwd()
if not current_directory.endswith("/"):
    current_directory += "/"


device = "cuda:0" if torch.cuda.is_available() else "cpu"

CLOZE_MODEL = 'bertfornsp_cloze_finetuned'
ROC_MODEL = 'bertfornsp_roc_finetuned'
# underlying pretrained LM
BASE_MODEL = 'bert-large-uncased-whole-word-masking'

BATCH_SIZE = 12
WARMUP_EPOCHS = 1
TRAIN_EPOCHS = 10
LAST_EPOCH = -1


class RocStories(torch.utils.data.Dataset):
    def __init__(self, short = False):
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
        for i, sample in enumerate(dataset):
            if short is True:
                if i > 10000:
                    break
            start = " ".join(sample[2:-1])
            stories.append(start)
            end = sample[-1]
            endings.append(end)

        from random import shuffle
        wrong_endings = endings.copy()
        shuffle(wrong_endings)

        assert len(stories) == len(endings)
        for i, story in enumerate(stories):

            # True Ending
            self.data.append([story, endings[i]])
            self.labels.append(0)

            # Wrong Ending
            self.data.append([story, wrong_endings[i]])
            self.labels.append(1)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


# %% codecell
class ClozeTest(torch.utils.data.Dataset):
    def __init__(self, dev=True, hypothesis_only=False, file=None):
        """
        :param hypothesis_only: Replaces story with empty string.
                                Only Keeps endings as they are.
        :param hard: For future hard_test_set.csv
        """

        dataset = []

        # if dev=True, we load the dev set for testing
        dir = ""

        if file is None:
          if dev:
              dir = current_directory + 'cloze_test.csv'
          else:
              dir = current_directory + 'cloze_train.csv'
        else: dir = current_directory + file

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
            if hypothesis_only:
                start = ""
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


# %% codecell
class ClozeTest_negated(torch.utils.data.Dataset):
    def __init__(self):
        """
        :param hypothesis_only: Replaces story with empty string.
                                Only keeps endings as they are.
        :param hard: For future hard_test_set.csv
        """

        dataset = []

        # if dev=True, we load the dev set for testing
        dir = current_directory + "cloze_test_negated.csv"


        with open(dir, 'r', encoding='utf-8') as d:
            reader = csv.reader(d, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for line in reader:
                dataset.append(line)
            dataset.pop(0)

        self.data = []
        self.labels = []

        for sample in dataset[:100]:

            start = " ".join(sample[1:-3])
            end1 = sample[-3]
            end2 = sample[-2]
            right_ending = sample[-1]

            self.data.append([start, end1])
            self.labels.append(0 if "2" == right_ending else 1)

            self.data.append([start, end2])
            self.labels.append(0 if "1" == right_ending else 1)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


# %% codecell
class ClozeTest_MC(torch.utils.data.Dataset):
    def __init__(self, dev=True, hypothesis_only=False, file=None):

        dataset = []

        dir = ""

        if file is None:
            if dev:
                dir = current_directory + 'cloze_test.csv'
            else:
                dir = current_directory + 'cloze_train.csv'
        else:
            dir = current_directory + file

        # if dev=True, we load the dev set for testing
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
            if hypothesis_only:
                start = ""
            end1 = sample[-3]
            end2 = sample[-2]
            right_ending = sample[-1]

            self.data.append([start, end1, end2])
            self.labels.append(0 if "1" == right_ending else 1)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


# %% codecell
class ClozeTest_negated_MC(torch.utils.data.Dataset):
    def __init__(self, dev=True):

        dataset = []

        # if dev=True, we load the dev set for testing
        dir = current_directory + "cloze_test_negated.csv"

        with open(dir, 'r', encoding='utf-8') as d:
            reader = csv.reader(d, quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for line in reader:
                dataset.append(line)
            dataset.pop(0)

        self.data = []
        self.labels = []

        for sample in dataset[:100]:

            start = " ".join(sample[1:-3])
            end1 = sample[-3]
            end2 = sample[-2]
            right_ending = sample[-1]

            self.data.append([start, end1, end2])
            self.labels.append(1 if "1" == right_ending else 0)  # inversed

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.labels)


# %% markdown
# # Auxiliary Functions
# %% codecell
def getModelFileName(model_name, last_epoch, subdir = "models"):
    path = os.path.join(current_directory, subdir, model_name
                        + str(last_epoch))
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


# %% codecell
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
# %% markdown
# # Functions for Training and Testing
# %% codecell
def train(cloze_test, model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False, model_name=None):

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    #model_old = BertForNextSentencePrediction.from_pretrained(model_file)

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

        name = ""
        if model_name != None: name = model_name
        else: name = CLOZE_MODEL if cloze_test else ROC_MODEL

        model.save_pretrained(
            getModelFileName(name, epoch + 1)
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


# %% codecell
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

    #print(confusion_matrix(label_list, pred_list))
    print(classification_report(label_list, pred_list))

    #return confusion_matrix(label_list, pred_list).ravel()


# %% codecell
def train_MC(cloze_test = ClozeTest_MC(dev=False), model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False, model_name=None):

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    #model_old = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Training
    model = model.to(device)
    model.train()

    trainloader = torch.utils.data.DataLoader(cloze_test, batch_size=batch_size, shuffle=True)

    #LR maybe needs to be optimized
    optimizer = AdamW(model.parameters(), lr=1e-5)
    n_batches =  len(trainloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(warmup_epochs * n_batches),
        num_training_steps=(train_epochs * n_batches),
        last_epoch=max(-1, last_epoch * n_batches) # actually, last_step
    )

    loss_fct = torch.nn.CrossEntropyLoss()

    losses = []
    epochs_range = range(last_epoch + 1, train_epochs)
    for epoch in tqdm(epochs_range):

        for batchId, (stories, labels) in zip(range(n_batches), trainloader):
            # this is PyTorch-specific as gradients get accumulated
            optimizer.zero_grad()

            start = stories[0]
            end1 = stories[1]
            end2 = stories[2]

            tokenized_batch_end1 = tokenizer(start, padding = True, text_pair = end1,
                                        return_tensors='pt').to(device)

            tokenized_batch_end2 = tokenizer(start, padding = True, text_pair = end2,
                                        return_tensors='pt').to(device)

            #Send to GPU
            labels = labels.to(device)


            logits0 = model(**tokenized_batch_end1).logits
            logits1 = model(**tokenized_batch_end2).logits

            logits_combined = logits0 + logits1.flip(-1)
            loss = loss_fct(logits_combined.view(-1,2), labels.view(-1))
            losses.append(loss.item())

            """
            loss = 0
            for i in range(len(labels.data)): #Iterate through batch
                log0 = logits0.data[i].to(device)
                log1 = logits1.data[i].to(device)
                if labels[i].item() == 0:
                    label0 = torch.tensor([1,0])
                    label1 = torch.tensor([0,1])
                else:
                    label0 = torch.tensor([0,1])
                    label1 = torch.tensor([1,0])

                label0 = label0.to(device)
                label1 = label1.to(device)

                logits_combined = (log0 + log1.flip(-1)) * (label0 + label1.flip(-1))

                loss += loss_fct(logits_combined.unsqueeze(0), labels[i].unsqueeze(0))

            """

            loss.backward()
            optimizer.step()
            scheduler.step() # Huggingface specific: step = epoch

        name = ""
        if model_name != None: name = model_name
        else: name = CLOZE_MODEL if cloze_test else ROC_MODEL

        model.save_pretrained(
            getModelFileName(name, epoch + 1)
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
    #weight_diff(model, model_old)


# %% codecell
def test_MC(model_file=BASE_MODEL, verbose = False, cloze_test = ClozeTest_MC(dev=True)):
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
        end1 = stories[1]
        end2 = stories[2]

        tokenized_batch_end1 = tokenizer(start, padding = True, text_pair = end1,
                                    return_tensors='pt').to(device)

        tokenized_batch_end2 = tokenizer(start, padding = True, text_pair = end2,
                                    return_tensors='pt').to(device)

        #Send to GPU
        labels = labels.to(device)

        logits0 = model(**tokenized_batch_end1).logits
        logits1 = model(**tokenized_batch_end2).logits

        logits = logits0 + logits1.flip(-1)

        # Model predicts sentence-pair as correct if True-logit > False-logit
        predictions = logits.argmax(dim=1).int()
        #probs = softmax(logits).cpu().detach()

        """
        predictions = []

        for i in range(len(labels.data)):
            end1_likelih = logits_end1.data[i][0]
            end2_likelih = logits_end2.data[i][0]
            likelihoods = torch.tensor([end1_likelih, end2_likelih])
            pred = likelihoods.argmax(dim=0).int()
            predictions.append(pred)

        """

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

    #print(confusion_matrix(label_list, pred_list))

    print(classification_report(label_list, pred_list))


# %% codecell
def train_mixed(model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False, model_name=None):

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    model_old = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Training
    model = model.to(device)
    model.train()

    cloze = ClozeTest(dev=False)
    roc = RocStories(short = True)
    cloze.data.extend(roc.data)
    cloze.labels.extend(roc.labels)


    trainloader = torch.utils.data.DataLoader(cloze, batch_size=batch_size, shuffle=True)


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

        name = ""
        if model_name != None: name = model_name
        else: name = CLOZE_MODEL if cloze_test else ROC_MODEL

        model.save_pretrained(
            getModelFileName(name, epoch + 1)
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


# %% codecell
#IGNORE THIS. BAD IDEA
def train_bias_reduced(cloze_test, model_file=BASE_MODEL, batch_size=BATCH_SIZE,
          warmup_epochs=WARMUP_EPOCHS, train_epochs=TRAIN_EPOCHS,
          last_epoch=LAST_EPOCH, verbose=False):

    softmax = torch.nn.Softmax(dim=1)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForNextSentencePrediction.from_pretrained(model_file)
    # The old weights are saved in model_old to be used to compare to model
    model_old = BertForNextSentencePrediction.from_pretrained(model_file)

    #Send to GPU and allow Training
    model = model.to(device)
    model.train()

    cloze = ClozeTest(dev=False)
    cloze_hyp = ClozeTest(dev=False, hypothesis_only=True)
    cloze.data.extend(cloze_hyp.data)
    cloze.labels.extend(cloze_hyp.labels)


    trainloader = torch.utils.data.DataLoader(cloze, batch_size=batch_size, shuffle=True)

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

            logits = model(**tokenized_batch, labels = labels).logits
            probs = softmax(logits)

            loss_fct = torch.nn.MSELoss()
            #loss_fct = torch.nn.CrossEntropyLoss()

            loss = 0
            for i, dp_start in enumerate(start):

                if dp_start == "":
                    target = torch.tensor([0.5, 0.5], dtype=torch.float)
                else:
                    if labels[i] == 0: target = torch.tensor([1,0], dtype=torch.float)
                    else: target = torch.tensor([0,1], dtype=torch.float)
                target = target.to(device)
                loss += loss_fct(probs[i], target)



            if verbose:
                print("Epoch " + str(epoch + 1) +
                      " Batch " + batchId + " of " + n_batches +
                      " Loss: " + loss.item())
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step() # Huggingface specific: step = epoch

        model.save_pretrained(
            getModelFileName((CLOZE_MODEL + "_bias_reduced_better"), epoch + 1)
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
    #weight_diff(model, model_old)


# %% markdown
# # Create Dataset with Noise
# %% codecell
def insert_noise(file, endings_included, temporals, locations, conjunctives):
    # Creates Testset file with noise
    import random

    with open((current_directory + 'noise_test_set_endingsincluded.csv') if endings_included else (current_directory + 'noise_test_set.csv'), 'a') as f:
        writer = csv.writer(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        with open(current_directory + file, 'r', encoding='utf-8') as d:
            reader = csv.reader(d, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for j, sample in enumerate(reader):
                sentences = sample[1:-3]
                endings = sample[-3:-1]

                loc_used = False
                temp_used = False
                for i, sentence in enumerate(sentences):
                    choice = random.randrange(0,3)
                    if choice == 0 and not temp_used:
                        sentences[i] = random.choice(temporals) + " " + sentence
                        temp_used = True
                    elif choice == 1 and not loc_used:
                        sentences[i] = f"In {random.choice(locations)}," + " " + sentence
                        loc_used = True
                    elif i != 0: sentences[i] = random.choice(conjunctives) + " " + sentence

                if endings_included:
                    conjunctive = random.choice(conjunctives)
                    endings[0] = conjunctive + " " + endings[0]
                    endings[1] = conjunctive + " " + endings[1]
                    row = [sample[0]] + sentences + endings + [sample[-1]]

                else: row = [sample[0]] + sentences + sample[-3:]
                writer.writerow(row)
