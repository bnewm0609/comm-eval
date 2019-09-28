"""This file is going to hold some useful functions/classes used in the
Replication Evaluation Notebook that describes how the results were obtained"""
from itertools import chain # flatten
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr, kendalltau

from nlgeval import NLGEval # for running baseline eval metrics

import caption_featurizers
import color_featurizers
from experiment import FeatureHandler
from models import LiteralListener, CaptionEncoder
from models import LiteralSpeakerScorer, CaptionGenerator
from monroe_data import MonroeData, MonroeDataEntry, Color

########################## UTILITY FUNCTIONS ##########################

def normalize(caption):
    """
    Makes caption text lowercase and removes uneeded white space.

    Args:
        caption: a candidate caption string.

    Returns:
        caption lowercased, without extra whitespace or double quotes
    """
    return caption.lower().strip().replace('"', "")

def flatten(ctxs):
    """
    Turns 2-d list of lists into 1-d list.

    Equivalent to np.flatten but for non-square 2-d lists.

    Args:
        ctxs: a 2-D list of lists of contexts.
            [[ctx1.1, ctx1.2], [ctx2.1, ctx2.2, ctx2.3], ...]
    Returns:
        1-D list of the same contexts.
        [ctx1.1, ctx1.2, ctx2.1, ctx2.2, ctx2.3, ...]
    """
    return list(chain.from_iterable(ctxs))

def target_col_for_context_id(df, context_id):
    """
    Gets the target color given a context id and a dataframe to search in.

    Args:
        df: the dataframe that holds the target color.
        context_id: the id of the context whose target color is desired.
    Returns:
        Tuple of 3 floats representing the HSL coordinates of the target color.
    """
    ctx_entry = df[df["contextId"] == context_id].iloc[0]
    statuses = ['click', 'alt1', 'alt2']
    color = (None, None, None)
    for status in statuses:
        if ctx_entry[f"{status}Status"] == "target":
            color = (ctx_entry[f"{status}ColH"],
                     ctx_entry[f"{status}ColS"],
                     ctx_entry[f"{status}ColL"])
            break
    return color

########################## PLOTTING FUNCTIONS ###########################

def plot_score_dists_for_metric(scores, metric):
    """
    Plots the scores assigned by a given metric on a violin plot.

    Args:
        scores: the scores to plot the distributions of. List of three
            dictionaries that contain all the scores. One descriptive, one
            ambiguous, and one misleading.
        metric: name of metric that assigned the scores to label the graph.

    Returns:
        None. Just prints correlations and displays graph.
    """
    plot_data = [score[metric][1] for score in scores]

    # let's also get the correlation coefficients
    # domain is [1, 1, 1, ...], [2, 2, 2, ...], [3, 3, 3, 3, ...]
    correlation_support = []
    for i, d in enumerate(plot_data):
        correlation_support.extend([i+1] * len(d))

    plot_data_flat = flatten(plot_data)

    pearson_r = pearsonr(correlation_support, plot_data_flat)
    spearman_r = spearmanr(correlation_support, plot_data_flat)
    kendall_t = kendalltau(correlation_support, plot_data_flat)
    print("Pearson r: {:.3f}, p = {:.3f}".format(pearson_r[0], pearson_r[1]))
    print("Spearman r: {:.3f}, p = {:.3f}".format(spearman_r[0], spearman_r[1]))
    print("Kendall τ: {:.3f}, p = {:.3f}".format(kendall_t[0], kendall_t[1]))

    # now plot the violin plots
    vpdata = plt.violinplot(plot_data, bw_method=0.2,
                            showmeans=True, showextrema=True)
    vpdata["cbars"].set(linewidths=1)
    vpdata["cmeans"].set(linewidths=1)
    vpdata["cmins"].set(linewidths=1)
    vpdata["cmaxes"].set(linewidths=1)

    # add scatter plots on top of vilions
    plt.scatter(correlation_support, plot_data_flat, alpha=0.01, s=5)

    plt.xticks([1, 2, 3], ["Descriptive", "Ambiguous", "Misleading"])
    plt.title("{} scores of captions".format(metric))
    plt.show()

def plot_score_dists(*scores, metrics=['Bleu_1', 'ROUGE_L', 'METEOR', 'CIDEr']):
    """Plots distributions of scores asssigned from all the metrics given."""
    for metric in metrics:
        plot_score_dists_for_metric(scores, metric)


######################## METRIC CALCULATION WRAPPERS #########################

class NgramMetrics:
    """
    Wrapper for calculating Ngram Overlap metrics.

    Attributes:
        nlgeval_metrics: the NLGEval object (from Sharma et al., 2018) that
            performs all of the evaluations. We do not load the glove or
            skipthought evaluaations and only use the n-gram overlap metrics.
    """

    def __init__(self):
        """
        Loads metrics without extra (slow) models.

        Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOER, and CIDEr.
        """
        self.nlgeval_metrics = NLGEval(no_overlap=False,
                                       no_glove=True,
                                       no_skipthoughts=True)

    def get_overlap_scores(self, refs, hyps):
        """
        Calculates n-gram scores given references and candidates.

        Args:
            refs: list of lists of all reference caption strings for each
                candidate caption.
            hyps: list of hypothesis or candidate captions.

        Returns:
            Dictionary of metric names to a tuple containing the mean score
            and the list of all scores. This list is the same size as the
            hyp list.

            {
                "BLUE_1" : (0.23, [0.41, 0.84, 0.31, ...]),
                "ROUGE_L" : (0.45, [0.12, 0.68, 0.24, ...]),
                ...
            }
        """
        refs = {idx: lines for (idx, lines) in enumerate(refs)}
        hyps = {idx: [line] for (idx, line) in enumerate(hyps)}
        ret_scores = {}
        for scorer, method in self.nlgeval_metrics.scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = (sc, scs)
            else:
                ret_scores[method] = (score, scores)
        return ret_scores


class ListenerMetrics:
    """

    Attributes:
        train_data: MonroeData object that holds all of the model's training
            data. It is used to fix the vocabulary for the model, but probably
            should be swapped out for a lighter vocabuary file.
        caption_phi: the caption feature function that takes care of tokenizing
            and normalizing captions and converting their tokens to indices.
        color_phi_ll: the color feautre function used by the literal listener.
            It is based on a fourier feature representation derived from RGB
            color space.
        color_phi_pl: the color feature function used by the pragmatic listener.
            It is also based on a fourier feature representation but is in the
            HSV color space.
        model_ll: the pretrained literal listener model that selects a color
            given a caption.
        model_pl: the pretrained literal speaker model that gives the
            probability of a caption given a color context. It's used in the
            pragmatic listener model.

    """

    def __init__(self):
        """
        Sets up training data, color/caption feature functions, and pretrained
        models for evaluation.
        """
        # Data used to generate vocab (this is suboptimal)
        self.train_data = MonroeData("../data/csv/train_corpus_monroe.csv",
                                     "../data/entries/train_entries_monroe.pkl")

        self.caption_phi = caption_featurizers.CaptionFeaturizer(
            tokenizer=caption_featurizers.EndingTokenizer)
        # Literal Listener uses RGB for color features
        self.color_phi_ll = color_featurizers.ColorFeaturizer(
            color_featurizers.color_phi_fourier, "rgb", normalized=True)
        # Pragmatic Listener uses HSV for color features
        self.color_phi_pl = color_featurizers.ColorFeaturizer(
            color_featurizers.color_phi_fourier, "hsv", normalized=True)

        # placeholder feature handler to load vocab (this is ugly) TODO
        fh = FeatureHandler(self.train_data, self.train_data,
                            self.caption_phi, self.color_phi_ll)

        # load the listener models
        self.model_ll = LiteralListener(CaptionEncoder)
        self.model_ll.init_model(
                    embed_dim=100,
                    hidden_dim=100,
                    vocab_size=fh.caption_featurizer.caption_indexer.size,
                    color_dim=54)

        self.model_ll.load_model(
            "../model/literal_listener_5epoch_endings_tkn.params")
        
        self.model_pl = LiteralSpeakerScorer(CaptionGenerator)
        self.model_pl.init_model(
                    color_in_dim=54,
                    color_dim=100,
                    vocab_size=fh.caption_featurizer.caption_indexer.size,
                    embed_dim=100,
                    speaker_hidden_dim=100)

        self.model_pl.load_model("../model/literal_speaker_30epochGLOVE.params")

    def get_lit_listener_scores(self, eval_contexts):
        """
        Queries Literal Listener model for target colors given a caption.

        Args:
            eval_contexts: list of caption-color contexts to evaluate with
                the literal listener model.
        Returns:
            log-probability of the target color given the caption.
        """
        feature_handler = FeatureHandler(self.train_data, # for vocab :(
                                         eval_contexts,
                                         self.caption_phi,
                                         self.color_phi_ll,
                                         randomized_colors=True)
        X_assess = feature_handler.test_features()
        y_assess = feature_handler.test_targets()

        preds = self.model_ll.predict(X_assess)
        scores = preds[np.arange(len(preds)), y_assess]
        return scores


    def speaker_target(self, data_entry):
        """
        Gets gold tokens for finding caption probabilities.

        Args:
            data_entry: i.e. a color-caption context

        Returns:
            Tokenized caption with end token but no start token.
        """
        _, caption_ids = self.caption_phi.to_string_features(data_entry.caption)
        target = caption_ids[1:]
        return target

    def speaker_predictions_to_scores(self, results, targets):
        """
        Converts speaker's predictions over tokens to a score for each color.

        Args:
            results: the outputted log-probs from the speaker models.
            targets: the gold token sequences used for calculating the
                probabilities of the captions.

        Returns:
            The log-probabilities the pragmatic speaker model assigns to the
            true targets being the target color.
        """
        # Gets the probability of the caption given each color is the target.
        all_scores = []
        target_lens = np.array([len(target) for target in targets])
        for i, predictions in enumerate(results):
            scores = [0, 0, 0]
            for j, prediction in enumerate(predictions):
                scores[j] = np.sum(
                    prediction[np.arange(target_lens[i]), targets[i]].numpy()
                ) # end tokens are already cut off by LSS model

            all_scores.append(scores)

        # Apply Bayes Rule in log space to get prob of color given caption
        all_scores = np.array(all_scores)
        all_scores_dist = (all_scores.T - logsumexp(all_scores, axis=1)).T
        return all_scores_dist[:, 0] # the target is at index 0

    def get_prag_listener_scores(self, eval_contexts):
        """
        Queries Pragmatic Listener model to score each color given caption.

        Args:
            eval_contexts: the color-caption contexts being evaluated

        Returns:
            Log probabilites for all possible tokens for each position in the
            gold caption for each color.
        """
        feature_handler = FeatureHandler(self.train_data,
                                         eval_contexts,
                                         self.caption_phi,
                                         self.color_phi_pl,
                                         randomized_colors=False,
                                         target_fn=self.speaker_target)
        X_assess = feature_handler.test_features()
        y_assess = feature_handler.test_targets()

        preds = self.model_pl.predict(X_assess)
        scores = self.speaker_predictions_to_scores(preds, y_assess)
        return scores

    def get_listener_scores(self, eval_contexts):
        """
        Runs all listener eval metrics and return scores in a dictionary.

        Wrapper function for handling the calling of the literal listener and
        pragmatic listener evaluation functions.

        Args:
            eval_contexts: the color-caption contexts being scored.

        Returns:
            Dictionary containing the names of the two listener evaluation
            methods as keys and a tuple of mean score and list of scores
            assigned by those methods as values.
        """

        ll_scores = np.exp(self.get_lit_listener_scores(eval_contexts))
        pl_scores = np.exp(self.get_prag_listener_scores(eval_contexts))
        results = {
            "Literal Listener": (np.mean(ll_scores), ll_scores),
            "Pragmatic Listener": (np.mean(pl_scores), pl_scores)
        }
        return results
