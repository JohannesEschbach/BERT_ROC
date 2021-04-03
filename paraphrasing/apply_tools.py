
# %% markdown
# ## ClozeOnly
# %% codecell
train_epochs_cloze = 10
train(train_epochs = train_epochs_cloze, cloze_test=True, model_name="bertfornsp_clozeonly_finetuned")
# %% markdown
# ## Cloze + 5000Roc
# %% codecell
train_epochs_cloze = 5
train_mixed(model_name = "bertfornsp_mixed_more_roc", train_epochs = train_epochs_cloze)
# %% markdown


# %% codecell
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
model = BertForNextSentencePrediction.from_pretrained(getModelFileName(CLOZE_MODEL, "10"))
model2 = BertForNextSentencePrediction.from_pretrained(getModelFileName("bertfornsp_mixed", "5"))
model3 = BertForNextSentencePrediction.from_pretrained(getModelFileName("bertfornsp_mixed_more_roc", "5"))

# %% codecell
cloze_test = ClozeTest(dev=True)
story, label = cloze_test[100] #Example data point
input, ending = story

saliency_map(model = model, tokenizer=tokenizer, input = input, ending = ending, label = label)
saliency_map(model = model2, tokenizer=tokenizer, input = input, ending = ending, label = label)
saliency_map(model = model3, tokenizer=tokenizer, input = input, ending = ending, label = label)

# %% codecell
get_trigger_words(hard = False, dev_only = False, train_only = True, min_occurences = 5, token_ids = False)


# %% codecell
file = "cloze_test.csv"
locations = "Tiko;Cahersiveen;Yomju-up;Culebra;Isernia;Zhangxi;Straldzha;Clayton;Dernekpazari;Kueps;Oberfranken;Pagalungan;Sulbiate;Alto;Ruhstorf;Sighetu Marmatiei;Satuek;Tisina;Kalvia;Batan;Algodre;Herrlisheim;Franklin;Achim;Krasnoshchekovo;Luoping;Villalago;Robinwood;Xuanma;Isole Tremiti;Jaisinghnagar;Omu Aran;Zamania;Mouila;Palacios del Sil;Nova Olimpia;Argelaguer;Bolangitang;Cohasset;Fermanville;Aunon;Kujama;Xingping;Darfield;Shangxing;Brices Creek;Geghamasar;Grobengereuth;Radevormwald;Jastrebarsko;Pignataro Interamna;Fairview Shores;Fruitland;Charleston;Agios Loukas;Tiruppuvanam;Effingham;Valdunciel;Kruft;Gornyye Klyuchi;Bou Arfa;Villabraz;Bokhorst;Yulee;Pouembout;Ettenstatt;Tambo;Sinteu;Erzhausen;Bai'e;Santopadre;Puquio;Huai'an;Tuotang;Giriawas;Dayingzi;Wolpertswende;Vassouras;Vari;Lapua;Cameroon;Ireland;Korea;Puerto Rico;Italy;China;Bulgaria;America;Turkey;Germany;Italy;Germany;Romania;Thailand;Slovenia;Finland;Costa Rica;Spain;France;England;Germany;Russia;China;Italy;United States;China;Italy;India;Nigeria;India;Gabon;Spain;Brazil;Spain;Indonesia;United States;France;Spain;Nigeria;China;China;Armenia;Germany;Germany;Croatia;Italy;Greece;India;Spain;Germany;Russia;Algeria;Spain;Germany;the United States;New Caledonia;Germany;Philippines;Romania;Germany;China;Italy;Peru;China;China;Indonesia;China;Germany;Brazil;Greece;Finland"
locations = locations.split(";")
temporals = ["Last week","Yesterday","Last summer", "Last winter", "Last spring", "Last autumn", "Last year", "Recently,", "Some time ago,", "A long time ago", "Today", "Last Monday", "Last Tuesday", "Last Sunday", "Last Wednesday", "Last Thursday", "Last Friday", "Last Saturday", "Just now", "Lately", "Not long ago", "A decade ago", "Ages ago"]
conjunctives = ["Furthermore,","Incidentally", "Moreover,", "Also,", "Consequently,", "However,", "Indeed,","Then,","Thus,", "Anyway,","Certainly,","Finally,","Meanwhile,", "Now,", "Afterwards,","Thereafter,","Well,", "What's more,", "Ultimately,", "Hence", "Therefore"]

insert_noise(file, True, temporals, locations, conjunctives)
insert_noise(file, False, temporals, locations, conjunctives)
