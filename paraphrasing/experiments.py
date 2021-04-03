# %% markdown
# Go sure current directory contains test sets and models
#
# Models:
#
# ClozeOnly: https://drive.google.com/drive/folders/119WnpHBmM637M0SVk3buy-KPI5aWpsF0?usp=sharing
#
# RocOnly: ...still training
#
# Cloze + 5000 Roc: https://drive.google.com/drive/folders/1-XfWuEsxEAKUby35Zz_y9zEo6kyDuSRF?usp=sharing

# %% codecell
import csv
import paraphrasing.models as m

# %% markdown
# # Training
#
# ## RocOnly

# %% codecell
# train_epochs_roc = 10
# train(train_epochs=train_epochs_roc, cloze_test=False, batch_size=32, warmup_epochs=0, model_name=ROC_MODEL)
# fn = m.getModelFileName(m.CLOZE_MODEL, train_epochs_roc)
# m.test(fn)
# here was the content which was moved to apply_tools.py

# %% markdown
# # Experiments

# %% codecell
def test_testset(file, hypothesis_only):
    if file =="cloze_test_negated.csv":
        cloze_test = m.ClozeTest_negated()
        cloze_test_mc = m.ClozeTest_negated_MC()
    else:
        cloze_test = m.ClozeTest(dev=True, hypothesis_only = hypothesis_only, file=file)
        cloze_test_mc = m.ClozeTest_MC(dev=True, hypothesis_only=hypothesis_only, file=file)


    print("\nBert\n")
    m.test(m.BASE_MODEL, cloze_test = cloze_test)
    m.test_MC(m.BASE_MODEL, cloze_test = cloze_test_mc)

    print("\nRocOnly\n")
    m.test(m.getModelFileName(m.ROC_MODEL, ""), cloze_test = cloze_test)
    m.test_MC(m.getModelFileName(m.ROC_MODEL, ""), cloze_test = cloze_test_mc)

    print("\nClozeOnly\n")
    m.test(m.getModelFileName("bertfornsp_clozeonly_finetuned", "10"), cloze_test = cloze_test)
    m.test_MC(m.getModelFileName("bertfornsp_clozeonly_finetuned", "10"), cloze_test = cloze_test_mc)

    print("\nRocCloze\n")
    m.test(m.getModelFileName("bertfornsp_cloze_finetuned", "10"), cloze_test = cloze_test)
    m.test_MC(m.getModelFileName("bertfornsp_cloze_finetuned", "10"), cloze_test = cloze_test_mc)

    print("\nCloze + 5000 Roc\n")
    m.test(m.getModelFileName("bertfornsp_mixed", "5"), cloze_test = cloze_test)
    m.test_MC(m.getModelFileName("bertfornsp_mixed", "5"), cloze_test = cloze_test_mc)

test_testset("paraphrasing/data/cloze_test.hu-HU_sl-SL_en-GB.csv", False)

# %% codecell
#Not complete yet
def test_model(model, name):
    #cloze_test = ClozeTest(dev=True, hypothesis_only=False)
    #cloze_test_hyp = ClozeTest(dev=True, hypothesis_only=True)
    cloze_test_negated = ClozeTest_negated()
    #cloze_test_triggers_only = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_triggers_only.csv")
    #cloze_test_no_triggers_only = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_no_triggers_only.csv")
    #cloze_test_cz = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_cz.csv")
    #cloze_test_cz_es = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_cz_es.csv")
    #cloze_test_cz_ja_pl = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_cz_ja_pl.csv")
    #cloze_test_cz_fi = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_cz_fi.csv")
    #cloze_test_ch = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_ch.csv")
    #cloze_test_ru = ClozeTest(dev=True, hypothesis_only=False, file="cloze_test_ru.csv")


    #cloze_test_mc = ClozeTest_MC(dev=True, hypothesis_only=False)
    #cloze_test_mc_hyp = ClozeTest_MC(dev=True, hypothesis_only=True)
    cloze_test_mc_negated = ClozeTest_negated_MC()
    #cloze_test_mc_triggers_only = ClozeTest_MC(dev=True, hypothesis_only=False, file="cloze_test_triggers_only.csv")
    #cloze_test_mc_no_triggers_only = ClozeTest_MC(dev=True, hypothesis_only=False, file="cloze_test_no_triggers_only.csv")
    #cloze_test_mc_paraphrased = ClozeTest_MC(dev=True, hypothesis_only=False, file="cloze_test_paraphrased.csv")

    print(name, "\n")

    """
    print("\nCloze:\n")
    test(model, cloze_test = cloze_test)
    print("\nCloze Hypothesis Only:\n")
    test(model, cloze_test = cloze_test_hyp)
    """
    print("\nCloze Negated:\n")
    test(model, cloze_test = cloze_test_negated)
    """
    print("\nCloze Triggers Only")
    test(model, cloze_test = cloze_test_triggers_only)
    print("\nCloze No Triggers Only")
    test(model, cloze_test = cloze_test_no_triggers_only)
    print("\nCloze Paraphrased")
    test(model, cloze_test = cloze_test_paraphrased)

    print("\nCloze_Choice:\n")
    test_MC(model, cloze_test = cloze_test_mc)
    print("\nCloze_Choice Hypothesis Only:\n")
    test_MC(model, cloze_test = cloze_test_mc_hyp)
    """
    print("\nCloze_Choice Negated_MC:\n")
    test_MC(model, cloze_test = cloze_test_mc_negated)
    """
    print("\nCloze_Choice Triggers Only")
    test(model, cloze_test = cloze_test_mc_triggers_only)
    print("\nCloze_Choice No Triggers Only")
    test(model, cloze_test = cloze_test_mc_no_triggers_only)
    print("\nCloze_Choice Paraphrased")
    test(model, cloze_test = cloze_test_mc_paraphrased)
    """
