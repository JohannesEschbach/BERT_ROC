import torch
import matplotlib
from IPython.display import display, HTML
from transformers import BertTokenizer
from tabulate import tabulate

from models import ClozeTest, BASE_MODEL

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Saliency Maps
# %% codecell
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

    logits = model(**tokens, labels=torch.tensor([label]).to(device)).logits.view(-1, 2)
    # Gradient of loss as per Han et al. 2020 calculated
    torch.nn.CrossEntropyLoss()(logits, logits.argmax(dim=-1)).backward()

    frw_handle.remove()
    bck_handle.remove()

    saliencies = (-grads["emb"] * acts["emb"]).sum(dim=-1)
    norm = torch.linalg.norm(saliencies, ord=1, dim=-1, keepdims=True)
    saliencies = saliencies / norm # normalizing the saliencies
    saliencies = saliencies[0] # squeezing the batch of one

    # Visualization. Courtesy of https://gist.github.com/ihsgnef
    colors = saliencies / max(abs(saliencies.min()), abs(saliencies.max())) * 0.5 + 0.5
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for token_name, color in zip(token_names.split(), colors.tolist()):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + token_name + '&nbsp')

    display(HTML(colored_string))


# %% markdown
# # Trigger Words
# %% codecell
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


# %% codecell
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

    print(tabulate(pos_triggers[:100], headers=['Token', 'n', 'pmi', 'pos_class_likelihood']))
    print("\n")
    print(tabulate(neg_triggers[:100], headers=['Token', 'n', 'pmi', 'neg_class_likelihood']))

    return pos_triggers, neg_triggers
