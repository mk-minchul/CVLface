import os
import numpy as np
from .metrics import DIR_FAR


def evaluate(
        all_features,
        image_paths,
        meta,
        ranks=[1, 5, 20]
):
    evaluator = TinyFaceTest(meta)
    results = evaluator.test_identification(all_features, image_paths, ranks)
    results = {k: v for k, v in zip(['rank-{}'.format(r) for r in ranks], results)}
    results = {k: v * 100 for k, v in results.items()}
    return results


class TinyFaceTest:
    def __init__(self, meta):
        self.meta = meta

    def get_key(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def get_label(self, image_path):
        return int(os.path.basename(image_path).split('_')[0])

    def init_proto(self, image_paths, probe_paths, match_paths, distractor_paths):
        index_dict = {}
        for i, image_path in enumerate(image_paths):
            index_dict[self.get_key(image_path)] = i

        self.indices_probe = np.array([index_dict[self.get_key(img)] for img in probe_paths])
        self.indices_match = np.array([index_dict[self.get_key(img)] for img in match_paths])
        self.indices_distractor = np.array([index_dict[self.get_key(img)] for img in distractor_paths])

        self.labels_probe = np.array([self.get_label(img) for img in probe_paths])
        self.labels_match = np.array([self.get_label(img) for img in match_paths])
        self.labels_distractor = np.array([-100 for img in distractor_paths])

        self.indices_gallery = np.concatenate([self.indices_match, self.indices_distractor])
        self.labels_gallery = np.concatenate([self.labels_match, self.labels_distractor])

    def test_identification(self, features, image_paths, ranks=[1, 5, 20]):
        assert len(image_paths) == len(features)
        assert len(image_paths) == len(self.meta['image_paths'])
        self.init_proto(image_paths,
                        self.meta['probe_paths'],
                        self.meta['gallery_paths'],
                        self.meta['distractor_paths'])

        feat_probe = features[self.indices_probe]
        feat_gallery = features[self.indices_gallery]
        compare_func = inner_product
        score_mat = compare_func(feat_probe, feat_gallery)

        label_mat = self.labels_probe[:, None] == self.labels_gallery[None, :]

        results, _, __ = DIR_FAR(score_mat, label_mat, ranks)

        return results


def inner_product(x1, x2):

    # normalize
    x1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    x2 = x2 / np.linalg.norm(x2, axis=1, keepdims=True)

    return np.dot(x1, x2.T)
