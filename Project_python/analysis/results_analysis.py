# Analysis of the results obtained separately with the NMF, LDA and Top2Vec models
# from cProfile import label
# from curses import raw
from turtle import color
import pandas as pd
import numpy as np
import conf
import matplotlib.pyplot as plt
import time

paths = conf.get_paths()
pathFolder = paths["out"] + "Global/"

training_set = 0 # 0: training files, 1: validation files
abstracts_cases = [1,0] # 0: abstracts, 1: full texts
model = 4 # 1:nmf, 2:lda, 3:top2vec, 4: all
plot = 0 # 0:no plot, 1: plot

print('######### Loading texts...')
def parse_ascii_sdgs_association(sdgs_ascii):
    sdgs = sdgs_ascii.split('|')
    scores = []
    for sdg in sdgs:
        score = float(sdg.split(':')[1])
        scores.append(score)
    return np.array(scores)

plt.figure(figsize=(8, 8))
for abstracts, dir, colors in zip(abstracts_cases, [-1,1], ["green", "red"]):
    if training_set:
        top2vec = pd.read_excel(paths["out"] + "Top2vec/" + "test_top2vec_training_files0.xlsx")
        nmf = pd.read_excel(paths["out"] + "NMF/" + "test_nmf_training_files.xlsx")
        lda = pd.read_excel(paths["out"] + "LDA/" + "test_lda_training_files0.xlsx")
    else:
        if abstracts:
            nmf = pd.read_excel(paths["out"] + "Send/" + "test_nmf_abstracts.xlsx")
            lda = pd.read_excel(paths["out"] + "Send/" + "test_lda_abstracts.xlsx")
            top2vec = pd.read_excel(paths["out"] + "Send/" + "test_top2vec_abstracts.xlsx")
        else:
            nmf = pd.read_excel(paths["out"] + "Send/" + "test_nmf_full.xlsx")
            lda = pd.read_excel(paths["out"] + "Send/" + "test_lda_full.xlsx")
            top2vec = pd.read_excel(paths["out"] + "Send/" + "test_top2vec_full.xlsx")
    
    # Abstracts data
    texts = list(top2vec["text"])
    labeledSDGs = list(top2vec["real"])

    sdgs_ascii_nmf = list(nmf["sdgs_association"])
    sdgs_nmf = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_nmf]

    sdgs_ascii_lda = list(lda["sdgs_association"])
    sdgs_lda = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_lda]

    sdgs_ascii_top2vec = list(top2vec["sdgs_association"])
    sdgs_top2vec = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_top2vec]

    def filter_nmf(raw_sdgs, min_valid):
        potential_sdgs = np.zeros(17)
        for sdg, sdgIndex in zip(raw_sdgs, range(17)):
            if sdg >= min_valid:
                potential_sdgs[sdgIndex] = sdg
        discarded_sdgs = raw_sdgs - potential_sdgs

        return (potential_sdgs, discarded_sdgs)

    def filter_lda(raw_sdgs, min_valid):
        potential_sdgs = np.zeros(17)
        for sdg, sdgIndex in zip(raw_sdgs, range(17)):
            if sdg >= min_valid:
                potential_sdgs[sdgIndex] = sdg
        discarded_sdgs = raw_sdgs - potential_sdgs

        return (potential_sdgs, discarded_sdgs)

    def filter_top2vec(raw_sdgs, min_valid):
        # threshold = np.mean(raw_sdgs) 
        
        potential_sdgs = np.zeros(17)
        for sdg, sdgIndex in zip(raw_sdgs, range(17)):
            if sdg >= min_valid:
                potential_sdgs[sdgIndex] = sdg
        discarded_sdgs = raw_sdgs - potential_sdgs

        return (potential_sdgs, discarded_sdgs)

    def identify_sdgs(sdgs_nmf, sdgs_lda, sdgs_top2vec):
        sdgs_new = np.zeros(17)
        
        for index, nmf, lda, top2vec in zip(range(len(sdgs_nmf)), sdgs_nmf, sdgs_lda, sdgs_top2vec):
            sdgs = np.array([nmf, lda, top2vec])
            if (sdgs >= 0.3).any():
                tmp = [ii for ii in sdgs if ii >= 0.3]
                sdgs_new[index] = np.mean(tmp)
            # elif index == 2 and top2vec >=0.2:
            #     tmp = [ii for ii in sdgs if ii >= 0.2]
            #     sdgs_new[index] = np.mean(tmp)
            else:
                count1 = [ii for ii in sdgs if ii >= 0.2]
                count2 = [ii for ii in sdgs if ii >= 0.15]
                if len(count1) >= 1 and len(count2) >= 2:
                    tmp = [ii for ii in sdgs if ii >= 0.2]
                    sdgs_new[index] = np.mean(tmp)
        return sdgs_new
            
    xlabel_sdgs = range(1,18)
    xlabel_np = np.array(xlabel_sdgs)
    valid_any = 0; valid_all = 0; count_any = 0; count_all = 0
    identified_sdgs = []; scores_sdgs = []
    count_per_sdg = np.zeros(17); id_ok_per_sdg = np.zeros(17)

    for textIndex in range(len(sdgs_ascii_top2vec)):
        print('Text {} of {}'.format(textIndex + 1, len(sdgs_ascii_top2vec)))
        if plot: plt.figure(figsize=(8, 8))
        
        if model == 1:
            potential_sdgs, discarded_sdgs = filter_nmf(sdgs_nmf[textIndex], min_valid=0.20)
            path_save = pathFolder + "Images/NMF/"
            path_resume = pathFolder + "Images/n_files_classified_ok_nmf.png"
            if plot:
                plt.bar(xlabel_np - 0.1, potential_sdgs, width=0.15, label='valid-nmf')
                plt.bar(xlabel_np + 0.1, discarded_sdgs, width=0.15, label='disc-nmf')
            
        elif model == 2:    
            potential_sdgs, discarded_sdgs = filter_lda(sdgs_lda[textIndex], min_valid=0.20)
            if plot: plt.bar(xlabel_np, sdgs_lda[textIndex], width=0.15, label='lda')
            path_save = pathFolder + "Images/LDA/"
            path_resume = pathFolder + "Images/n_files_classified_ok_lda.png"
                

        elif model == 3:
            potential_sdgs, discarded_sdgs = filter_top2vec(sdgs_top2vec[textIndex], min_valid=0.2)
            path_save = pathFolder + "Images/TOP2VEC/"
            path_resume = pathFolder + "Images/n_files_classified_ok_top2vec.png"
                

            if plot:
                plt.bar(xlabel_np - 0.1, potential_sdgs, width=0.15, label='valid-top2vec')
                plt.bar(xlabel_np + 0.1, discarded_sdgs, width=0.15, label='disc-top2vec')
            
        elif model == -1:
            path_save = pathFolder + "Images/ALL/"
            if plot:
                plt.bar(xlabel_np - 0.15, sdgs_nmf[textIndex], width=0.15, label='nmf')
                plt.bar(xlabel_np + 0.0, sdgs_lda[textIndex], width=0.15, label='lda')
                plt.bar(xlabel_np + 0.15, sdgs_top2vec[textIndex], width=0.15, label='top2vec')
        else:
            potential_sdgs = identify_sdgs(sdgs_nmf[textIndex], sdgs_lda[textIndex], sdgs_top2vec[textIndex])
            path_resume = pathFolder + "Images/n_files_classified_ok.png"

            if abstracts:
                path_save = pathFolder + "Images/ALL_FILTER_ABSTRACTS/"
            else:
                path_save = pathFolder + "Images/ALL_FILTER_FULL/"
            if plot: plt.bar(xlabel_np, potential_sdgs, width=0.3, label='all')
            
        if plot:
            plt.xticks(xlabel_sdgs)
            plt.xlabel('SDGS')
            plt.ylabel("Score")
            plt.ylim(top=0.5)
            plt.title('SDGs to identify: {}'.format(labeledSDGs[textIndex]))
            plt.legend()
            plt.savefig(path_save + "text{}.png".format(textIndex))
            # print('################# \r')
            # print(texts[textIndex])
            # plt.show()
            plt.close()
            
        filtered_sdgs = ["{:.2f}".format(sdg) for sdg in potential_sdgs if sdg > 0.1]    
        scores_sdgs.append('|'.join(filtered_sdgs))
        id_sdgs = [list(potential_sdgs).index(sdg) + 1 for sdg in potential_sdgs if sdg > 0.1]
        identified_sdgs.append(id_sdgs)
        labeledSDGs_int = [int(labeled) for labeled in labeledSDGs[textIndex][1:-1].split(',')]
        for label in labeledSDGs_int:
            count_per_sdg[label - 1] += 1
            if label in id_sdgs:
                id_ok_per_sdg[label - 1] += 1
        
        tmp = 0
        count_all += 1
        ass_sdgs = labeledSDGs[textIndex][1:-1].split(',')
        ass_sdgs = [int(sdg) for sdg in ass_sdgs]
        predict_sdgs = []
        for sdg in potential_sdgs:
            if sdg > 0.1:
                predict_sdgs.append(list(potential_sdgs).index(sdg) + 1)
        for sdg in ass_sdgs:
            count_any += 1
            if sdg in predict_sdgs:
                tmp += 1
                valid_any += 1
        if tmp == len(ass_sdgs):
            valid_all += 1
        
    perc_any = valid_any / count_any * 100
    perc_all = valid_all / count_all * 100
    print('Summary -> any: {:.2f} %, all: {:.2f} %'.format(perc_any, perc_all))


    no_identified = np.array(count_per_sdg) - np.array(id_ok_per_sdg)
    plt.bar(xlabel_np + dir * 0.15, no_identified + id_ok_per_sdg, width=0.2, label='nmf', color=colors, alpha=0.5)
    plt.bar(xlabel_np  + dir * 0.15, id_ok_per_sdg, width=0.2, label='nmf', color=colors)
    plt.xticks(xlabel_sdgs)
    plt.xlabel('SDGS')
    plt.ylabel("Number of texts")
    # if abstracts:titleStart = "Abstracts. "
    # else: titleStart = "Full paper. "
    # if model == 4:
    #     plt.title(titleStart + 'Voted method. Green: ok, Red: not identified')
    # else:
    #     plt.title(titleStart + 'Threshold = 0.2. Green: ok, Red: not identified')


    df = pd.DataFrame()
    for case in range(len(texts)):
        df["texts"] = texts
        df["labeled_Sdgs"] = labeledSDGs
        df["identified_sdgs"] = identified_sdgs
        df["scores_sdgs"] = scores_sdgs
    if abstracts:
        pathOut = paths["out"] + "Global/" + "test_results_filtered_abstracts.xlsx"
    else:
        pathOut = paths["out"] + "Global/" + "test_results_filtered_full.xlsx"
    df.to_excel(pathOut)
    
plt.savefig(path_resume)
# print('################# \r')
# print(texts[textIndex])
plt.show()
plt.close()