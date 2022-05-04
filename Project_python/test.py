from logging import error
import data
import conf
import pandas as pd
import tools
import model
import json
import os


def test_text_preprocess(texts, n_texts=20):

    if len(texts) < n_texts: error("Hey, check input texts")
    outPath = "test/"
    texts = texts[0:(n_texts - 1)]
    
    print("#### No lemmatize, no stem")
    with open(outPath + "text_preprocess.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write("###################################### \r")
            f.write("##### ORIGINAL: \r" + text + "\r\r")
            f.write("##### NO_LEM_NO_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=False, stem=False)) + "\r\r")
            f.write("##### LEM_NO_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=True, stem=False)) + "\r\r")
            f.write("##### NO_LEM_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=False, stem=True)) + "\r\r")
            f.write("##### LEM_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=True, stem=True)) + "\r\r")
        f.close()
    
def test_individual_models_nmf():
    paths = conf.get_paths()
    models_nmf = model.get_individual_models_per_sdg(flag_train=True)
    sdgsFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
    
    
#%% Data loading
paths = conf.get_paths()
# natureAbstracts = data.get_nature_abstracts(paths["Nature"])
# natureFiles = data.get_nature_files(paths["Nature"])
# abstracts = data.get_previous_classified_abstracts(paths["Abstracts"])
# sdgsFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
# textsPathfinder = data.get_sdgs_pathfinder(paths["ref"])


# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')

raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=False, kw=True, intro=True, body=True, concl=True)
raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"], n_files=100)

topWords = 25

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)

# TRAINING SECTION
print('######## TRAINING MODELS...')
# pam = model.PAM_classifier(paths)
# pam.train_global_model(trainFiles, k1=17, k2=20, rm_top=0)
# pam.global_model.infer(trainFiles)

# lda = model.LDA_classifier(tp.LDAModel.make_doc(trainFiles[0].split(' ')))
# lda.train_individual_model_per_sdg()
# # lda.export_individual_model_topics_to_csv("out/topics_lda_individual_models_monogram.csv", n_top_words=topWords)
# lda.train_global_model(trainFilesCompact, n_topics=17)
# lda.export_global_model_topics_to_csv("out/topics_lda_global_unordered.csv", n_top_words=topWords)
# lda.map_model_topics_to_sdgs(n_top_words=topWords, path_csv="out/topics_lda_global_monogram.csv")

top2vec = model.Top2Vec_classifier(paths)
trainData = [raw_orgFiles + raw_extraFiles + raw_healthcare, sdgs_orgFiles + sdgs_extra + sdgs_healthcare]

def test_model(model, path_csv_topics="", path_test_excel=""):
    model.map_model_topics_to_sdgs(associated_sdgs=trainData[1], path_csv=path_csv_topics, num_docs=-1, normalize=True)# out/topics_top2vec.csv"

    model.test_model(corpus=raw_natureShort, stat_topics=-1, associated_SDGs=sdgs_nature,
                     path_to_excel=path_test_excel, 
                     only_bad=True, score_threshold=20
                     )
    
    # model.test_model(corpus=raw_natureExt, stat_topics=-1, associated_SDGs=sdgs_natureAll,
    #                  path_to_excel=path_test_excel, 
    #                  only_bad=False, score_threshold=2
    #                  )

# top2vec.train_global_model(train_data=trainData, embedding_model="all-MiniLM-L6-v2", method="learn", ngram=True, min_count=2, workers=8, embedding_batch_size=10, tokenizer=False, split=False, nSplit=25) #"all-MiniLM-L6-v2", universal-sentence-encoder
top2vec.load_global_model()

# test_model(top2vec, 
#            path_csv_topics="out/topics_top2vec_ext.csv",
#            path_test_excel="out/test_abstract_top2vec_ext2.xlsx"
#            )

if 1:
    nmf = model.NMF_classifier(paths)
    nmf.train_individual_model_per_sdg(multigrams=(1,1))
    # nmf.load_individual_model_per_sdg()
    # nmf.export_individual_model_topics_to_csv("out/topics_nmf_individual_models_monogram.csv", n_top_words=topWords)

    nmf.train_global_model(orgFiles, n_topics=16, multigrams=(1,2))
    # nmf.load_global_model(n_topics=17)
    nmf.map_model_topics_to_sdgs(n_top_words=topWords, path_csv="out/topics_nmf_global_bigram2.csv")

    # # TESTING SECTION
    print('###### NMF models...')
    nmf.test_model(corpus=natureShort, associated_SDGs=sdgs_nature, valid_threshold=0.2, path_to_excel="out/test_abstract_nmf2.xlsx")
    nmf.test_model(corpus=natureExt, associated_SDGs=sdgs_natureAll, valid_threshold=0.2, path_to_excel="out/test_full_nmf.xlsx")
    # nmf.test_model(corpus=raw_pathFinder, associated_SDGs=sdgs_pathFinder, path_to_excel="out/results6.xlsx")