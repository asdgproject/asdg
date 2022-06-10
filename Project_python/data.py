# File that contains the functions for accesing the required data: training and validation

import os
from numpy import empty
import pandas as pd
import json
import conf
import tools

def get_sdg_titles(refPath):
    # returns the title of each SDG as a dictionary, with key: SDGx, value = title.
    # for example: "SDG1":"No poverty"
    f = open(refPath + "SDG_titles.json")
    sdgs_title = json.load(f)
    f.close()
    return sdgs_title


def get_sdgs_seed_list(refPath):
    with open(refPath + "seed_list_sdgs.json", 'r') as f:
        text = f.read()
        f.close()
        dict = json.loads(text)
    return list(dict.values())
        
# DATASET: the role of artificial intelligence in achieving the sustainable development goals. NATURE PAPER. The user can select independently: abstract, keywords, introduction, body or conclusions.
def get_nature_files(abstract=True, kw=False, intro=False, body=False, concl=False):
    paths = conf.get_paths()
    with open(paths["ref"] + "cleaned_database.json", "r") as f:
        json_dump = f.read()
        f.close()
    database = json.loads(json_dump)
    
    corpus = []; associatedSDGs = []; indexes = []
    for file, index in zip(database, range(len(database))):
        text = ""
        sdgs = database[file]["SDG"]
        if 17 in sdgs:
            continue
        if abstract:
            if len(database[file]["abstract"]) > 50:
                text += " " + database[file]["abstract"]
        if kw:
            text += " " + database[file]["keywords"]
        if intro:
            text += " " + database[file]["introduction"]
        if body:
            text += " " + database[file]["body"]
        if concl:
            text += " " + database[file]["conclusions"]
        corpus.append(text)
        associatedSDGs.append(sdgs)
        indexes.append(index)
    print("- {} nature files were found".format(len(corpus)))
    return [corpus, associatedSDGs, indexes]

def get_nature_abstracts():
    paths = conf.get_paths()
    with open(paths["ref"] + "cleaned_database.json", "r") as f:
        json_dump = f.read()
        f.close()
    database = json.loads(json_dump)
    
    corpus = []; associatedSDGs = []; indexes = []
    for (file, index) in zip(database, range(len(database))):
        sdgs = database[file]["SDG"]
        if 17 in sdgs:
            continue
        if len(database[file]["abstract"].split(' ')) > 50:
            corpus.append(database[file]["abstract"])
            associatedSDGs.append(sdgs)
            indexes.append(index)
    print("- {} nature abstracts were found".format(len(corpus)))
    return [corpus, associatedSDGs, indexes]

def get_nature_abstracts_filtered():
    paths = conf.get_paths()
    excel = pd.read_excel(paths["ref"] + "test2set_thresholds.xlsx")
    texts = list(excel["text"]); sdgsAscii = list(excel["sdgs"])
    sdgs = tools.parse_sdgs_ascii_list(sdgsAscii)
    return [texts, sdgs]

# DATASET: https://sdgs.un.org/
# - Goals definition
# - Goals progress - evolution section
def get_sdgs_org_files(refPath, sdg_query=-1):
    # Returns an array where each elements consist of an array with the fields:
    # [0] abstract or text related to a SDG, [1]: array with the associated SDGs.
    path = refPath + "sdg_texts.xlsx"
    df = pd.read_excel(path)
    texts = list(df["text"]); sdgs = tools.parse_sdgs_ascii_list(list(df["sdg"]))
    
    corpus = []; associatedSDGs = []
    for text, sdg in zip(texts, sdgs):
        if sdg[0] == 17: continue # not included
        if sdg_query > 0:
            if  sdg_query == sdg[0]:
                corpus.append(text)
                associatedSDGs.append(sdg)
        else:
            corpus.append(text)
            associatedSDGs.append(sdg)
                            
    nFiles = len(corpus)
    print("- {} sdgs files were found".format(nFiles))    
    return [corpus, associatedSDGs]

# DATASET: https://www.kaggle.com/datasets/xhlulu/medal-emnlp
def get_health_care_files(refPath):
    sdgsPath = [refPath + "Extra_files/SDG3/"]
    corpus = []; associatedSDGs = []
    for path in sdgsPath:
        for file in os.listdir(path):
            f = open(path + file, 'r')
            text = f.read()
            f.close()
            fileSDG = [3]
            corpus.append(text)
            associatedSDGs.append(fileSDG)
    nFiles = len(corpus)
    print("- {} health care files (SDG3) were found".format(nFiles))   
    return [corpus, associatedSDGs]

# DATASET: files from scopus classified as related to a sdg previously by the algorithm
def get_previous_classified_abstracts(refPath):
    # returns files that were previously classifies by the algorithm as valid
    absPath = refPath + "Abstracts/"
    abstracts = []
    for file in os.listdir(absPath):
        if file.endswith(".txt"):
            f = open(absPath + file, 'r', encoding="utf-8")
            text = f.read()
            f.close()
            abstracts.append(text)   
            
# DATASET: https://sdg-pathfinder.org/ files related to each SDG
def get_sdgs_pathfinder(refPath, min_words=150):
    csv = pd.read_csv(refPath + "ds_sdg_path_finder.csv")
    corpus = []; sdgs = []
    for text, sdgsAscii in zip(csv["text"], csv["SDGs"]):
        sdgsInt = [int(sdg) for sdg in (sdgsAscii.replace("[","").replace("]","")).split(",")]
        if 17 in sdgsInt:
            continue
        if len(text.split(' ')) > min_words:
            corpus.append(text)
            sdgs.append(sdgsInt)
    print("- {} texts in the pathfinder dataset".format(len(corpus)))
    return [corpus, sdgs]

# MANUAL SELECTED files
def get_extra_manual_files(refPath, sdg_query=[]):
    # Returns an array where each elements consist of an array with the fields:
    # [0] abstract or text related to a SDG, [1]: array with the associated SDGs.
    sdgsPaths = [refPath + "Manual_selected/"]
    corpus = []; associatedSDGs = []
    for path in sdgsPaths:
        for file in os.listdir(path):
            f = open(path + file, 'r')
            text = f.read()
            f.close()
            fileSDG = []
            for sdg in file.split("_"):
                if sdg.isdigit():
                    if int(sdg) == 17: continue
                    fileSDG.append(int(sdg))
            ok = 0
            if len(sdg_query) > 0:
                for sdg in fileSDG:
                    if sdg in sdg_query: ok += 1
            else: ok = 1
            
            if ok > 0:
                corpus.append(text)
                associatedSDGs.append(fileSDG)
        
    nFiles = len(corpus)
    print("- {} manual files were found".format(nFiles))    
    return [corpus, associatedSDGs]
 

def get_iGEM_files(ref_path, verbose=True):
    path = ref_path + "iGEM_2004_2021/"
    fieldsSeparator = ":::"
    with open(path + "00Header.txt", 'r') as hd:
        text = hd.read()[:-1]; hd.close()
        fields = text.split(fieldsSeparator)
        
    abstracts = []; extInformation = []; not_valid = []
    for folder in os.listdir(path=path):
        if folder.startswith("iGEM"):
            for file in os.listdir(path=(path + folder)):
                if file.startswith("0"): continue # it is not a valid file
                try: 
                    fp = open(path + folder + "/" +  file, 'r', encoding='utf8')
                    text = fp.read()[:-1]; fp.close()
                    fieldsValue = text.split(fieldsSeparator)
                    data = dict()
                    for fieldValue, fieldName in zip(fieldsValue, fields):
                        data[fieldName] = fieldValue
                    
                    # append the data to the lists if OK
                    if data['Application'] == 'Accepted' and len(data['Abstract'].split(' ')) > 10:
                        data["Abstract"] =  data["Abstract"].encode("ascii", "ignore")
                        abstracts.append(data["Abstract"])
                        extInformation.append(data)
                except: 
                    not_valid.append(path + folder + "/" +  file)
                
    if verbose:
        print('## {} accepted texts with abstract were found'.format(len(abstracts)))   
        for file in not_valid:
            print('# Revise: ' + file)       
    
    return [abstracts, extInformation]
                

    
# paths = conf.get_paths()
# abstracts, information = get_iGEM_files(ref_path=paths["ref"])
# a= 2