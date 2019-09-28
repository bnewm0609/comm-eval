import torch
import torch.nn as nn
import numpy as np
from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
from caption_featurizers import CaptionFeaturizer
from color_featurizers import ColorFeaturizer, color_phi_fourier
from evaluation import score_model

import time
import math

def evaluate_model(assess_data, feature_handler, model, predictions_to_scores, model_scorer,
                    feature_handler_2=None, model_scorer_kwargs={}, accuracy=True):
    assess_features = feature_handler.test_features() # ~6 sec
    if feature_handler_2 != None:
        assess_features_2 = feature_handler_2.test_features()
        assess_features = list(zip(assess_features, assess_features_2))
    assess_targets = feature_handler.test_targets()
    model_outputs =  model.predict(assess_features)
    model_scores = predictions_to_scores(model_outputs, assess_targets) # decide what the score we're going to use is
    result = model_scorer(assess_data, model_scores, **model_scorer_kwargs)

    if accuracy: # also report accuracy
        model_predictions = np.argmax(model_outputs, axis=1)
        accuracy_val = sum(model_predictions == assess_targets) / len(assess_targets)
        print("Accuracy:", accuracy_val)
        result = ((result[0], accuracy_val), result[1:]) # assumes first thing in result is the coorelation
        

    return result


class FeatureHandler:
    """
    This class handles the interface between the data, the feature functions, and the model.
    Basically what it does is the following:

    1. Convert MonroeDataEntry to a list of np.array's per speaker using the caption and color feature
       functions along with any other feature functions that the model needs
    2. Converts the MonroeDataEntry to the prediction targets also by applying some user-specified function
       to each of the data entries
    3. Handles color order randomization

    It does this for both the train and assessment datasets. From here, you should be able
    to call a model's fit method with the resutls of a `train_features` as X and `train_targets` as y
    """
    def __init__(self, train_data, test_data, caption_phi, color_phi, extra_featurizers=[],
                 target_fn=None, randomized_colors=True):
        """
        tain_data - training data (type: MonroeData)
        test_data - assessment data (type: MonroeData)
        caption_phi - caption feature function (type: CaptionFeaturizer). Should not have called `construct_featurizer`
                      method yet
        color_phi   - color feature function (type: ColorFeaturizer)
        extra_featurizers - list any other feature functions to include. Each should have a `to_features`
                            method that takes a MonroeDataEntry to a feature
        target_fn - function for mapping MonroeDataEntry (and permuation if randomized_colors is true)
                    to a *single value*
        randomized_colors - True if color order should be randomized, False if should be fixed with target first.

        """
        self.caption_featurizer = caption_phi
        # constructs indexer by default now
        self.caption_featurizer.construct_featurizer(train_data)

        self.color_featurizer = color_phi

        self.extra_featurizers = extra_featurizers
        self.train_data = train_data
        self.test_data = test_data
        self.randomized_colors = randomized_colors

        if target_fn is None:
            # by default just use the color at the target index as the target
            if self.randomized_colors:
                self.target_fn = lambda de, color_perm: np.where(color_perm==de.target_idx)[0][0] # first idx for tuple, second for getting raw number
            else:
                self.target_fn = lambda de: de.target_idx # should be 0, but just in case
        else:
            self.target_fn = target_fn

        # for keeping track of where colors ended up if randomized
        self.train_color_permutations = []
        self.test_color_permutations = []

        # only construct caption index once
        self.constructed_index = False


    def get_features(self, data, construct=False):
        """
        Extracts caption features and color features from data entries using their respective
        feature functions. Randomize and store color order if required
        """
        features = []
        for data_entry in data:
            entry_features = []
            # Get caption features
            if self.caption_featurizer is not None:
                _, idx_features = self.caption_featurizer.to_string_features(data_entry.caption)
                entry_features.append(idx_features)

            # Get color features (and randomize order if needed)
            if self.color_featurizer is not None:
                color_features = self.color_featurizer.to_color_features(data_entry.colors)

                if self.randomized_colors:
                    color_features, permutations = self.color_featurizer.shuffle_colors(color_features)
                    if construct:
                        self.train_color_permutations.append(permutations)
                    else:
                        self.test_color_permutations.append(permutations)

                entry_features.append(color_features)

            # Get any other features
            for featurizer in self.extra_featurizers:
                entry_features.append(featurizer.to_features(data_entry))
            features.append(entry_features)

        return features

    def train_features(self):
        """
        Wrapper function for get_features that calls specifically with train data. Should
        only be called once because it also constructs the index
        """
        features = self.get_features(self.train_data, construct=(not self.constructed_index)) # will only construct index the first time
        self.constructed_index = True
        return features


    def test_features(self):
        """
        Wrapper function for get_features that calls specifically with assess data
        """
        if self.randomized_colors:
            # reset permutations for another round
            self.test_color_permutations = []
        return self.get_features(self.test_data)

    def get_targets(self, data, permutations=[]):
        """
        Given data, iterates through it and extracts whatever the target of the model prediction
        will be by calling self.target_fn on the entry. If we are going to be predicting color
        indices we need to know where the target color index ended up. To this end, we also pass
        in the permuted indices list in permutations.

        A way around this would be to actually change the raw entry and get the target index with
        entry.target_idx, but that kind of scares me...

        Returns a (len(data),) shape np.array with the targets
        """
        if len(permutations) == 0 and self.randomized_colors:
            print("Make sure to call feature function before target function so color permutations can be used when generating targets")
            return

        targets = []
        # we need to pass in the permutations to the target functions
        # so we know where each color ended up - this is kind of ugly
        # but needed if our task is predicting colors (not needed otherwise)
        # but included when color is randomized
        for i, data_entry in enumerate(data):
            if len(permutations) == 0:
                targets.append(self.target_fn(data_entry))
            else:
                targets.append(self.target_fn(data_entry, permutations[i]))
        targets = np.array(targets)
        if (targets.shape[-1] == 1):
            targets = np.array(targets).flatten()
        return targets


    def train_targets(self):
        """
        Wrapper function for get_targets that calls specifically with train data
        """
        return self.get_targets(self.train_data, self.train_color_permutations)


    def test_targets(self):
        """
        Wrapper function for get_targets that calls specifically with assess data
        """
        return self.get_targets(self.test_data, self.test_color_permutations)


if __name__ == "__main__":
    # just load a pretrained model and evaluate it on the dev set
    from monroe_data import MonroeData, MonroeDataEntry, Color # last two for reading pkl file
    from caption_featurizers import CaptionFeaturizer
    from color_featurizers import ColorFeaturizer, color_phi_fourier
    from models import CaptionEncoder, LiteralListener
    import sys

    print("Loading training and dev data")
    train_data = MonroeData("data/csv/train_corpus_monroe.csv", "data/entries/train_entries_monroe.pkl")
    dev_data = MonroeData("data/csv/dev_corpus_monroe.csv", "data/entries/dev_entries_monroe.pkl")

    print("Initializing featurizers")
    caption_phi = CaptionFeaturizer()
    color_phi = ColorFeaturizer(color_phi_fourier, "rgb", normalized=True)
    feature_handler = FeatureHandler(train_data, dev_data, caption_phi, color_phi) # target function is initialized by default

    print("Obtaining training features")  # have to get train featurs to get vocab size (EVEN IF YOU'RE RUNNING PRETRAINED MODEL)
    train_features = feature_handler.train_features()
    train_targets = feature_handler.train_targets()
    # print(train_targets[:10])

    print("Initializing model")
    # model parameters
    embed_dim = 100; hidden_dim = 100; color_dim= 54;# hard coded for example - 54 comes from color fourier phi
    model = LiteralListener(CaptionEncoder, num_epochs=5)
    model.init_model(embed_dim = embed_dim, hidden_dim = hidden_dim, vocab_size = feature_handler.caption_featurizer.caption_indexer.size,
                 color_dim = color_dim)

    # to train: (probably takes about 15 min - 2 hrs) depending on # of epochs (5 - 30)
    # print("Training model")
    # model.fit(train_features, train_targets)
    # model.save_model("model/literal_listener_5epoch-2.params")

    print("Loading and evaluating pretrained model")
    # model.load_model("model/literal_listener_5epoch-2.params")
    model.load_model("model/literal_listener_5epoch.params")

    # convert the model output to a score for that particular round
    output_to_score = lambda model_outputs, targets: np.exp(model_outputs[np.arange(len(model_outputs)), targets]) # get the model's predicted probablity at each target index and use that as the score
    evaluate_model(dev_data, feature_handler, model, output_to_score, score_model)



