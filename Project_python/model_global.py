# functions used for testing different model configurations
from logging import error
import data
import conf
import pandas as pd
import numpy as np
import tools
from scipy.special import softmax
         
class Global_Classifier:
    paths=[]
    nmf=[]; lda=[]; top2vec=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_models(self):
        self.nmf = tools.load_obj(self.paths["model"] + "nmf.pickle")
        print('# Loaded nmf...')
        
        self.lda = tools.load_obj(self.paths["model"] + "lda.pickle")
        print('# Loaded lda...')
        
        self.top2vec = tools.load_obj(self.paths["model"] + "top2vec.pickle")
        print('# Loaded top2vec...')
        
    def test_model(self, raw_corpus, corpus, associated_SDGs=[], path_to_plot="", path_to_excel="", only_bad=False,
                   score_threshold=3.0,  only_positive=False, filter_low=False):
        rawSDG = []; realSDGs = []; predic = []; scores = []; texts = []
       
        def parse_line(sdgs):
            sdgsAscii = ["x{}: {:.3f}".format(xx, topic) for topic, xx in zip(sdgs, range(1,18))]
            sdgsAscii = "|".join(sdgsAscii)
            sdgsAscii += "\n"
            return sdgsAscii
        if len(associated_SDGs) == 0: associated_SDGs = [[-1] for ii in range(len(corpus))]
        
        for raw_text, text, sdgs in zip(raw_corpus, corpus, associated_SDGs):
            [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs] = self.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive, version=1, filter_low=filter_low, normalize=False, normalize_threshold=-1)  
            
            concat_array = np.array([nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs])
            filt_mean = np.zeros(17)
            for ii in range(17):
                counter = 0.0; tmp = 0.0
                for val in concat_array[:, ii]:
                    if val >= 0.0:
                        if val >= 0.5: val = 0.5
                        counter += 1; tmp += val
                if counter > 0: tmp /= counter
                filt_mean[ii] = tmp
                
            # predict_sdgs, scores_sdgs = self.get_identified_sdgs(nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs)
            predict_sdgs, scores_sdgs = self.get_identified_sdgs_mean(nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs, filt_mean)
            
            filt_mean_softmax = softmax(filt_mean)
                       
            rawSDG.append("NMF -> "+ parse_line(nmf_raw_sdgs) + "LDA -> " + parse_line(lda_raw_sdgs) + "TOP2VEC -> " + parse_line(top_raw_sdgs) + "MEAN -> " + parse_line(filt_mean) + "SOFTMAX -> " + parse_line(filt_mean_softmax))
            predic.append(predict_sdgs); scores.append(scores_sdgs)
            realSDGs.append(sdgs)
            
            if len(raw_corpus) == 0: texts.append(text)
            else: texts.append(raw_text)

        # oks = [ok for ok in valids if ok == True]
        # oksSingle = [ok for ok in validsAny if ok == True]
        # perc_global = len(oks) / len(valids) * 100
        # perc_single = len(oksSingle) / len(valids) * 100
        # print("- {:.2f} % valid global, {:.3f} % valid any, of {} files".format(perc_global, perc_single, len(valids)))
        # print('Max found: {:.3f}'.format(maxSDG))
        
        # for probs, index in zip(probs_per_sdg, range(len(probs_per_sdg))):
        #     probs_per_sdg[index] = np.mean(probs_per_sdg[index])
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            df["predict"] = predic
            df["sdgs_association"] = rawSDG
            # df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
            df.to_excel(path_to_excel)
            
        self.get_statistics_from_test(actual_sdgs=associated_SDGs, predict_sdgs=predic)
            
        return predic, scores
    
    def get_statistics_from_test(self, actual_sdgs, predict_sdgs):
        count = 0; ok = 0; wrong = 0
        for act_sdg, pred_sdg in zip(actual_sdgs, predict_sdgs):
            for sdg in act_sdg:
                count += 1
                if sdg in pred_sdg: ok +=1
                else: wrong += 1
        print('## RESULTS OF TEST: OK: {:.2f} %'.format(ok / float(count) * 100.0))
        
    def get_identified_sdgs(self, nmf, lda, top2vec):
        identified = []; scores = []
        for sdg in range(1, 18):
            index = sdg - 1; 
            predic = np.array([nmf[index], lda[index], top2vec[index]])
            # if any(predic >= 0.3):
            #     identified.append(sdg)
            if np.count_nonzero(predic >= 0.15) >= 2:
                values = [value for value in predic if value >= 0.15]
                identified.append(sdg)
                scores.append(np.mean(values))
            else: pass # not identified
        return identified, scores
    
    def get_identified_sdgs_mean(self, nmf, lda, top2vec, mean_vec):
        identified = []; scores = []
        for sdg, predic in zip(range(1, 18), mean_vec):
            index = sdg - 1; 
            tmp = np.array([nmf[index], lda[index], top2vec[index]])
            
            flag_mean = predic >= 0.2
            flag_count_low = np.count_nonzero(tmp >= 0.2) >= 2
            flag_coun_high = np.count_nonzero(tmp >= 0.35) >= 1 and np.count_nonzero(tmp >= 0.1) >= 2
            
            # if flag_mean or flag_count_low or flag_coun_high:
            if flag_mean:
                identified.append(sdg)
                scores.append(predic)
            else: pass # not identified
        return identified, scores
            
           
    def map_text_to_sdgs(self, text, score_threshold, only_positive=False, version=1, filter_low=True, normalize=True, normalize_threshold=0.25):
        scale_factor = 1.4
        top_raw_sdgs, top_predic, top_score, top_raw_topicsScores = self.top2vec.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive, version=version, expand_factor=1.2*scale_factor, filter_low=filter_low, normalize=normalize, normalize_threshold=normalize_threshold)  
            
        nmf_raw_sdgs = self.nmf.map_text_to_sdgs(text, filter_low=filter_low, normalize=normalize, expand_factor=4.0*scale_factor)  
        
        lda_raw_sdgs = self.lda.map_text_to_sdgs(text, only_positive=only_positive, filter_low=filter_low, normalize=normalize, expand_factor=1.3*scale_factor) 

        return [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs]
    