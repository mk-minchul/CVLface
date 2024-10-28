import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.metrics import roc_curve


def image2template_feature(img_feats=None, templates=None, medias=None, dummy=False):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    if dummy:
        template_feats = np.random.randn(len(unique_templates), img_feats.shape[1])
        template_norm_feats = sklearn.preprocessing.normalize(template_feats)
        return template_norm_feats, unique_templates

    for count_template, uqt in tqdm(enumerate(unique_templates), total=len(unique_templates), desc='image2template_feature'):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [ total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize) ]
    for c, s in tqdm(enumerate(sublists), total=len(sublists), desc='verification'):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
    return score



def evaluate(embeddings, faceness_scores, templates, medias, label, p1, p2, dummy=False):

    infernece_configs = [{'use_norm_score': True, 'use_detector_score': True},
                         {'use_norm_score': True, 'use_detector_score': False},
                         {'use_norm_score': False, 'use_detector_score': True}, ]

    scores = {}
    for config in infernece_configs:
        use_norm_score = config['use_norm_score']
        use_detector_score = config['use_detector_score']

        img_input_feats = embeddings.copy()
        if not use_norm_score:
            # normalise features to remove norm information
            img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))
        if use_detector_score:
            img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]

        template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias, dummy=dummy)
        score = verification(template_norm_feats, unique_templates, p1, p2)
        method = f"Norm:{use_norm_score}_Det:{use_detector_score}"
        scores[method] = score

    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    result = {}
    for method in scores.keys():
        fpr, tpr, thresholds = roc_curve(label, scores[method])
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr
        thresholds = np.flipud(thresholds)
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
            best_thresh = thresholds[min_index]
            _fpr_val = x_labels[fpr_iter]
            _tpr_val = tpr[min_index] * 100
            result[f'{method}_tpr_at_fpr_{_fpr_val}'] = _tpr_val
            result[f'{method}_thresh_at_fpr_{_fpr_val}'] = best_thresh
    return result
