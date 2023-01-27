import json

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))

def load_jsonl(input_path):
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(tri_fpr,tri_tpr, tri_fpr_m, tri_tpr_m, cls_fpr, cls_tpr, m_cls_fpr, m_cls_tpr, a_cls_fpr, a_cls_tpr, tri_fpr_a,tri_tpr_a): 
    
    legendt = "ROC CURVE, STRATEGY 1 : MUTATION"
    plt.rcParams.update({'figure.figsize':(12,7), 'figure.dpi':100})
    
    plt.plot(tri_fpr,tri_tpr, '-.',color = 'green', label = 'Triplet model with treshold, only the condition mutated')
    plt.plot(tri_fpr_m,tri_tpr_m, '-.', color = 'blue', label = 'Triplet model with treshold, only the message mutated')
    
    plt.plot(cls_fpr, cls_tpr, color = 'green', label = 'BILSTM classification model, only the condition mutated')
    plt.plot(m_cls_fpr, m_cls_tpr, color = 'blue', label = 'BILSTM classification model, only the message mutated')
    
    plt.plot(a_cls_fpr, a_cls_tpr, color= 'orange', label = 'BILSTM classification model, condition and message')
    plt.plot(tri_fpr_a,tri_tpr_a, '-.',color = 'orange', label = 'Triplet model with treshold, condition and message')
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.plot([0,1],[0,1], '--', color='black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()

from sklearn import metrics
def plot_roc_curve_all_strategies(fprs_tprs_bilstm, fprs_tprs_codet5, fprs_tprs_triplet, labels, legendt = "ROC CURVE MIXED DATA"): 
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize = (5,5.0), dpi =100)
    plt.rcParams.update({'figure.figsize':(16,9), 'figure.dpi':100})
    
    plt.plot(fprs_tprs_bilstm[0][0], fprs_tprs_bilstm[0][1],color = 'green', label = labels[0][0])

    plt.plot(fprs_tprs_codet5[0][0], fprs_tprs_codet5[0][1], '-*', color = 'blue', label = labels[1][0])

    plt.plot(fprs_tprs_triplet[0][0], fprs_tprs_triplet[0][1],'-.', color = 'orange', label = labels[2][0])

    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.plot([0,1],[0,1], '--', color='black')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0., 1.01, 0.1))
    plt.yticks(np.arange(0., 1.01, 0.1))
    #plt.savefig("test_sets/RQ1_new.pdf", dpi=100,bbox_inches='tight')
    plt.show()

def plot_f1_curve_all_strategies(fprs_tprs_bilstm, fprs_tprs_triplet, labels, legendt = "F1 CURVE MIXED DATA"): 
       
    plt.rcParams.update({'figure.figsize':(16,9), 'figure.dpi':100})
    
    pre = np.array(fprs_tprs_bilstm[0][1])/(np.array(fprs_tprs_bilstm[0][1])+np.array(fprs_tprs_bilstm[0][0]))
    f1 = pre * np.array(fprs_tprs_bilstm[0][1]) * 2. / (pre + np.array(fprs_tprs_bilstm[0][1]))
    plt.plot(fprs_tprs_bilstm[0][0], f1,color = 'green', label = labels[0][0])
    
    pre = np.array(fprs_tprs_bilstm[1][1])/(np.array(fprs_tprs_bilstm[1][1])+np.array(fprs_tprs_bilstm[1][0]))
    f1 = pre * np.array(fprs_tprs_bilstm[1][1]) * 2. / (pre + np.array(fprs_tprs_bilstm[1][1]))
    plt.plot(fprs_tprs_bilstm[1][0], f1,color = 'red', label = labels[0][1])
    plt.plot(fprs_tprs_bilstm[2][0], fprs_tprs_bilstm[2][1],color = 'orange', label = labels[0][2])
    plt.plot(fprs_tprs_bilstm[3][0], fprs_tprs_bilstm[3][1],color = 'blue', label = labels[0][3])
    plt.plot(fprs_tprs_bilstm[4][0], fprs_tprs_bilstm[4][1],color = 'pink', label = labels[0][4])
    
    
    pre = np.array(fprs_tprs_triplet[0][1])/(np.array(fprs_tprs_triplet[0][1])+np.array(fprs_tprs_triplet[0][0]))
    f1 = pre * np.array(fprs_tprs_triplet[0][1]) * 2. / (pre + np.array(fprs_tprs_triplet[0][1]))
    plt.plot(fprs_tprs_triplet[0][0], f1,'-.', color = 'green', label = labels[1][0])
    plt.plot(fprs_tprs_triplet[1][0], fprs_tprs_triplet[1][1],'-.', color = 'red', label = labels[1][1])
    plt.plot(fprs_tprs_triplet[2][0], fprs_tprs_triplet[2][1],'-.', color = 'orange', label = labels[1][2])
    plt.plot(fprs_tprs_triplet[3][0], fprs_tprs_triplet[3][1],'-.', color = 'blue', label = labels[1][3])
    plt.plot(fprs_tprs_triplet[4][0], fprs_tprs_triplet[4][1],'-.', color = 'pink', label = labels[1][4])
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('F1 Score')
    plt.plot([0,1],[0.5,0.5], '--', color='black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    minor_ticks = np.arange(0, 1.01, 0.01)
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()

def plot_pr_curve_all_strategies(fprs_tprs_bilstm, fprs_tprs_triplet, labels, legendt = "PRECISION/RECALL CURVE MIXED DATA"): 
       
    plt.rcParams.update({'figure.figsize':(16,9), 'figure.dpi':100})
    
    ######ST1
    pre = np.array(fprs_tprs_bilstm[0][1])/(np.array(fprs_tprs_bilstm[0][1])+np.array(fprs_tprs_bilstm[0][0]))
    f1 = pre * np.array(fprs_tprs_bilstm[0][1]) * 2. / (pre + np.array(fprs_tprs_bilstm[0][1]))
    plt.plot(fprs_tprs_bilstm[0][1], pre, color = 'green', label = labels[0][0])
    
    
    ######ST2
    pre = np.array(fprs_tprs_bilstm[1][1])/(np.array(fprs_tprs_bilstm[1][1])+np.array(fprs_tprs_bilstm[1][0]))
    f1 = pre * np.array(fprs_tprs_bilstm[1][1]) * 2. / (pre + np.array(fprs_tprs_bilstm[1][1]))
    plt.plot(fprs_tprs_bilstm[1][1], pre, color = 'red', label = labels[0][1])
    
    
    
    plt.plot(fprs_tprs_bilstm[2][0], fprs_tprs_bilstm[2][1],color = 'orange', label = labels[0][2])
    plt.plot(fprs_tprs_bilstm[3][0], fprs_tprs_bilstm[3][1],color = 'blue', label = labels[0][3])
    plt.plot(fprs_tprs_bilstm[4][0], fprs_tprs_bilstm[4][1],color = 'pink', label = labels[0][4])
    
    #------------------------------------
    
    pre = np.array(fprs_tprs_triplet[0][1])/(np.array(fprs_tprs_triplet[0][1])+np.array(fprs_tprs_triplet[0][0]))
    f1 = pre * np.array(fprs_tprs_triplet[0][1]) * 2. / (pre + np.array(fprs_tprs_triplet[0][1]))
    plt.plot(fprs_tprs_triplet[0][1], pre, '-.', color = 'green', label = labels[1][0])
    
    
    plt.plot(fprs_tprs_triplet[1][0], fprs_tprs_triplet[1][1],'-.', color = 'red', label = labels[1][1])
    plt.plot(fprs_tprs_triplet[2][0], fprs_tprs_triplet[2][1],'-.', color = 'orange', label = labels[1][2])
    plt.plot(fprs_tprs_triplet[3][0], fprs_tprs_triplet[3][1],'-.', color = 'blue', label = labels[1][3])
    plt.plot(fprs_tprs_triplet[4][0], fprs_tprs_triplet[4][1],'-.', color = 'pink', label = labels[1][4])
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('RECALL') 
    plt.ylabel('PRECISION')
    plt.plot([0,1],[0.5,0.5], '--', color='black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    minor_ticks = np.arange(0, 1.01, 0.01)
    plt.xticks(np.arange(0., 1.01, 0.05))
    plt.yticks(np.arange(0., 1.01, 0.05))
    plt.show()

def plot_roc_curve_f1_beta(tri_fpr,tri_tpr, tri_fpr_m, tri_tpr_m, cls_fpr, cls_tpr, m_cls_fpr, m_cls_tpr, a_cls_fpr, a_cls_tpr, tri_fpr_a,tri_tpr_a): 
    
    legendt = "F1 SCORE CURVE, STRATEGY 1 : MUTATION"
    plt.rcParams.update({'figure.figsize':(12,7), 'figure.dpi':100})
    
    
    pre = np.array(tri_tpr)/(np.array(tri_tpr)+np.array(tri_fpr))
    f1 = pre * np.array(tri_tpr) * 2. / (pre + np.array(tri_tpr))
    plt.plot(tri_fpr,f1, '-.',color = 'green', label = 'Triplet model with treshold, only the condition mutated')
    
    pre = np.array(tri_tpr_m)/(np.array(tri_tpr_m)+np.array(tri_fpr_m))
    f1 = pre * np.array(tri_tpr_m) * 2. / (pre + np.array(tri_tpr_m))
    plt.plot(tri_fpr_m,f1, '-.', color = 'blue', label = 'Triplet model with treshold, only the message mutated')
    
    pre = np.array(cls_tpr)/(np.array(cls_tpr)+np.array(cls_fpr))
    f1 = pre * np.array(cls_tpr) * 2. / (pre + np.array(cls_tpr))
    plt.plot(cls_fpr, f1, color = 'green', label = 'BILSTM classification model, only the condition mutated')
    
    pre = np.array(m_cls_tpr)/(np.array(m_cls_tpr)+np.array(m_cls_fpr))
    f1 = pre * np.array(m_cls_tpr) * 2. / (pre + np.array(m_cls_tpr))
    plt.plot(m_cls_fpr, f1, color = 'blue', label = 'BILSTM classification model, only the message mutated')
    
    pre = np.array(a_cls_tpr)/(np.array(a_cls_tpr)+np.array(a_cls_tpr))
    f1 = pre * np.array(a_cls_tpr) * 2. / (pre + np.array(a_cls_tpr))
    plt.plot(a_cls_fpr, f1, color= 'orange', label = 'BILSTM classification model, condition and message')
    
    pre = np.array(tri_tpr_a)/(np.array(tri_tpr_a)+np.array(tri_tpr_a))
    f1 = pre * np.array(tri_tpr_a) * 2. / (pre + np.array(tri_tpr_a))
    plt.plot(tri_fpr_a, f1, '-.',color = 'orange', label = 'Triplet model with treshold, condition and message')
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('F1 Score')
    plt.plot([0,1],[0,1], '--', color='black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()

def treshold_classifier(dist, t):
    if dist < t:
        return 0.
    elif dist > t:
        return 1.

def calc_tpr_fpr(zeros, ones, classifier, n_tresh= 10, max_t = 1):

    treshes= [max_t*i/n_tresh for i in range(n_tresh+1)]
    pos_pred_by_tresh = []
    neg_pred_by_tresh = []

    fprs = []
    tprs = []

    for t in treshes:
        pos_pred = np.array([classifier(d, t) for d in ones], dtype=np.float32)
        neg_pred = np.array([classifier(d, t) for d in zeros], dtype=np.float32)

        tpr = sum(pos_pred>0.5)/pos_pred.shape[0]
        fpr = sum(neg_pred>0.5)/neg_pred.shape[0]

        fprs.append(fpr)
        tprs.append(tpr)

        pos_pred_by_tresh.append(pos_pred)
        neg_pred_by_tresh.append(neg_pred)
    
    return fprs, tprs

def plot_roc_curve_2(cls_fpr_1, cls_tpr_1, tri_fpr_1, tri_tpr_1, cls_fpr_2, cls_tpr_2, tri_fpr_2, tri_tpr_2): 
    legendt = "ROC CURVE, STRATEGY 2 : RANDOM MATCHING"
    plt.rcParams.update({'figure.figsize':(12,7), 'figure.dpi':100})
    
   
    plt.plot(cls_fpr_1, cls_tpr_1,color = 'green', label = 'Non Semantic Mutation, BILSTM Classifier')
    
    plt.plot(tri_fpr_1, tri_tpr_1, '-.', color = 'green', label = 'Non Semantic Mutation, Triplet Treshold Classifier')
    
    plt.plot(cls_fpr_2, cls_tpr_2, color = 'orange', label = 'Random Matching, BILSTM classification model')
    #plt.plot(m_cls_fpr, m_cls_tpr, color = 'blue', label = 'BILSTM classification model, only the message mutated')
    
    #plt.plot(a_cls_fpr, a_cls_tpr, color= 'orange', label = 'BILSTM classification model, condition and message')
    #plt.plot(tri_fpr_a,tri_tpr_a, '-.',color = 'orange', label = 'Triplet model with treshold, condition and message')
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.plot([0,1],[0,1], '--', color = 'black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()

legendt = "F1 SCORE, ALL STRATEGIES"
def plot_roc_curve_df1(cls_fpr_1, cls_tpr_1, tri_fpr_1, tri_tpr_1, cls_fpr_2, cls_tpr_2, tri_fpr_2, tri_tpr_2): 
    plt.rcParams.update({'figure.figsize':(12,7), 'figure.dpi':100})
    
    pre = np.array(cls_tpr_1)/(np.array(cls_tpr_1)+np.array(cls_fpr_1))
    f1 = pre * np.array(cls_tpr_1) * 2. / (pre + np.array(cls_tpr_1))
    plt.plot(cls_fpr_1, cls_tpr_1,color = 'green', label = 'Non Semantic Mutation, BILSTM Classifier')
    
    pre = np.array(tri_tpr_1)/(np.array(tri_tpr_1)+np.array(tri_fpr_1))
    f1 = pre * np.array(tri_tpr_1) * 2. / (pre + np.array(tri_tpr_1))
    plt.plot(tri_fpr_1, tri_tpr_1, '-.', color = 'green', label = 'Non Semantic Mutation, Triplet Treshold Classifier')
    
    pre = np.array(cls_tpr_2)/(np.array(cls_tpr_2)+np.array(cls_fpr_2))
    f1 = pre * np.array(cls_tpr_2) * 2. / (pre + np.array(cls_tpr_2))
    plt.plot(cls_fpr_2, cls_tpr_2, color = 'orange', label = 'Random Matching, BILSTM classification model')
    #plt.plot(m_cls_fpr, m_cls_tpr, color = 'blue', label = 'BILSTM classification model, only the message mutated')
    
    #plt.plot(a_cls_fpr, a_cls_tpr, color= 'orange', label = 'BILSTM classification model, condition and message')
    #plt.plot(tri_fpr_a,tri_tpr_a, '-.',color = 'orange', label = 'Triplet model with treshold, condition and message')
    
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.plot([0,1],[0,1], '--', color = 'black')
    plt.gca().set(title=legendt)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0., 1.1, 0.1))
    plt.yticks(np.arange(0., 1.1, 0.1))
    plt.show()