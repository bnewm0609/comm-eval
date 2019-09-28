# Holds objects used for accessing and processing data from the Monroe et al., 2017 color datasets
# (None of this is at all efficient at the moment, but has been constructed for clarity)


import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import colorsys as cs
import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter
import nltk


# Note: Monroe et al., 2017 does not include any listener utterances, so the total
# of these partitions is 46994 rather than 57946

monroe_train_dev_test_partitions = {'train':15665, 'dev':15670, 'test': 15659} 

class MonroeData():
    """
    Wraps the CSV file containing the filtered data from Monroe et al., 2017 and the data we collect.
    """
    
    def __init__(self, data_filename, entries_filename=None, single_speaker=True, ss_method="pool"):
        """
        Initializes the MonroeData object with the CSV passed.

        MonroeData allows for storage, access, and display of data from the Monroe et al, 2017
        color experiments. Each csv row is stored in a MonroeDataEntry object and MonroeData
        stores a python list of these objects as well as the underlying dataframe.

        Args:
            data_filename - csv file that holds the Monroe et al data

            entries_filename - If the entries_filename is not None, it should be a pickle file that can be unpacked
                to get at the MonroeDatakEntry objects in the entries without taking the 3 or so minutes
                to generate them.

            single_speaker - If single_speaker is True, the dataframe created from data_filename is filtered to include
                only utterances from the speaker according to one of two methods (specified by the ss_method
                parameter).
                - pool: take every utterance from a given game and round that the speaker uttered and group
                        them together as a single utterance. These utterances are space separated. [Default]
                - final: take the last utterance the speaker made from a given game and round use that as
                        representative of the speaker's utterance for the entire game

            ss_method - See single_speaker. Ignored if single_speaker is false

        Returns:
            None
        """
        self.data = pd.read_csv(data_filename)
        if single_speaker:
            data = self.data[self.data["role"] == "speaker"]

            if ss_method == "pool":
                pool_msg_times = data.groupby(['gameid', 'roundNum'])
                concat_rows = data.groupby(['gameid', 'roundNum']).contents.transform(' ~ '.join)
                data['contents'] = concat_rows # throws warning, but isn't actually be a problem

            # if ss_method is pool, this is arbitrary, taking the last of the identical speaker utterances.
            # if ss_method is final, this choice is not arbitrary because the speaker utterances differ
            max_msg_times = data.groupby(['gameid', 'roundNum']).msgTime.transform(max)
            self.data = data.loc[max_msg_times == data.msgTime]
            # reset index so indices line up with entries list
            self.data.reset_index(drop=True, inplace=True)
        
        if entries_filename is None:
            self.entries = []
        else:
            with open(entries_filename, "rb") as pkl_file:
                self.entries = pkl.load(pkl_file)

            # helps make sure entries_filename and data_filename match up
            assert len(self.entries) == self.data.shape[0]

        self.vocab = Counter();
        
    def read_data(self):
        """
        Iterates over the CSV file rows adding them to entries list.

        This method is a generator for getting training examples from
        self.entries (if they have been read in from a pkl file) or from
        the dataframe rows. Constructs the vocab from the tokens in the entries
        """
        for k, row in self.data.iterrows():
            if len(self.entries) < self.data.shape[0]:
                self.entries.append(MonroeDataEntry(k, row))
            yield self.entries[k]

        self.read_vocab()

    def read_vocab(self):
        """
        Initializes the self.vocab counter with word counts.
        """
        pass
    
    def train_dev_test_split(self, partitions=monroe_train_dev_test_partitions, filenames={'train': 'train_corpus.csv', 'dev': 'dev_corpus.csv', 'test': 'test_corpus.csv'}, random=False):
        """
        Creates a random partition of entries in the self.data to form a train, dev, test split.

        The number of entries in each of the train, dev, and test splits is given. According to
        the partitions argument. new dataframes are saved to their respective files in filenames.

        Args:
            partitions - Dictionary containing how many entries are each of the splits
                {'train': ###, 'dev': ###, 'test': ###} [default from Monroe et al., 2017 paper]
            filenames - The names of the files that the partioned data will be in.
                Default is {'train': 'train_corpus.csv', 'dev': 'dev_corpus.csv', 'test': 'test_corpus.csv'}

        Returns:
            train, dev, and test Pandas Dataframes.
        """
        num_train = partitions['train']
        num_dev = partitions['dev']
        num_test= partitions['test']

        # first randomly order indices
        permutation_indices = list(range(len(self)))
        if random:
            np.random.shuffle(permutation_indices)

        # next create new empty dataframes
        train_df = pd.DataFrame(index = list(range(num_train)), columns=self.data.columns)
        dev_df = pd.DataFrame(index = list(range(num_dev)), columns=self.data.columns)
        test_df = pd.DataFrame(index = list(range(num_test)), columns=self.data.columns)

        # now populate those dataframes according to the index they were randomly assigned to in permutation_indices
        train_i = 0
        dev_i   = 0
        test_i  = 0
        for i, row in self.data.iterrows():
            if permutation_indices[i] < num_train: # training
                train_df.loc[train_i] = row
                train_i += 1
            elif permutation_indices[i] < num_train + num_dev: # dev
                dev_df.loc[dev_i] = row
                dev_i += 1
            else: # test
                test_df.loc[test_i] = row
                test_i += 1

        # now save these to filenames (if not None)
        if filenames:
            train_df.to_csv(filenames['train'], index=False)
            dev_df.to_csv(filenames['dev'], index=False)
            test_df.to_csv(filenames['test'], index=False)

        return train_df, dev_df, test_df


    def save_entries(self, filename):
        """
        Save the contents of the entries list to the specified pkl file.

        Args:
            filename - the name of the pickle file holding the entries
        """
        with open(filename, "wb") as pkl_file:
            pkl.dump(self.entries, pkl_file)

    def display_target(self, color, caption):
        """
        Displays the color in a rectangle with the caption above it.

        Args:
            color - list with rgb values - all values normalized between 0 and 1
            caption - string displayed above image
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        target_rect = matplotlib.patches.Rectangle((0,0), 800, 800, facecolor=color, edgecolor="black")
        
        ax.add_patch(target_rect)
        ax.set_yticks([])
        ax.set_xticks([])
        print(caption)
        plt.show()

    def display_custom_game(self, colors, caption):
        """
        Displays color contexts in a rectangle and caption.

        These colors and captions can be completely independent of
        the entries in the MonroeData object

        Args:
            colors - A list of Colors that to be displayed along with the captions.
            caption - A string containing the caption to displayed with the colors.
        """
        # width and height of each rectangle
        width = 400
        height = 800

        # bottom corners of all the rectangles
        bottom_corners = [(x - 600, -400) for x in range(0, width*len(colors), width)]

        color_rects = []
        for i, corner in enumerate(bottom_corners):
            # target color will always be on left, we want clicked color
            # to stand out with a thick (linewidth=5), red outline that is
            # at the top of the rectangle stack (zorder = 5) displayed
            outline_color, stack_order, line_width = ("black", 0, None)
            color_rects.append(matplotlib.patches.Rectangle(corner, width, height,
                                                            facecolor=colors[i].rgb_norm, 
                                                            edgecolor=outline_color,
                                                            zorder = stack_order,
                                                            linewidth=line_width))
        # add rectangles to figures
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for rect in color_rects:
            ax.add_patch(rect)

        # hide axis labels
        ax.set_yticks([])
        ax.set_xticks([])
        
        # display caption and game
        print(caption)
        plt.xlim([-600, width*len(colors) - 600])
        plt.ylim([-400, 400])
        plt.show()
        
    def display_game_for_gameid(self, game_id, round_num = None):
        """
        Displays the color context and caption for a particular game and round.
        
        The Monroe et al., 2017 data comes in games of 50 rounds. If no round
        number is specified, al 50 rounds will be displayed. In the data we
        collect we also put in game ids but this method is intended for use
        with the Monroe data specifically. For our data use display_target_for_idx
        or display_custom_game.

        Args:
            game_id - the unique game identifier from the Monroe et al., Dataset.
            round_num - one of three values:
                1. the number of the specific context within the game to display.
                2. a list of contexts numbers to display.
                3. None (by default) - all of the rounds in the game are displayed.
        """
        df_indices = []
        round_num_list = []
        if round_num is None:
            # add all the indices!
            round_num_list = self.data[self.data['gameid'] == game_id].roundNum.unique()
        elif isinstance(round_num, list):
            round_num_list = round_num
        else:
            # round_nums is a single integer, so we just convert it to a list
            round_num_list = [round_num]
        
        
        for round_num in round_num_list:
            df_indices.extend(
                self.data[(self.data['gameid'] == game_id) & (self.data['roundNum'] == round_num)].index.values
            )
        
        for i in df_indices:
            self.display_game(i)
        
    
    def display_target_for_idx(self, row_index):
        """
        Displays just the target color and caption for the entry at the given index.

        Args:
            row_index - the index into the dataframe of the target color to display.
        """
        data_entry = self.entries[row_index]
        self.display_target(data_entry.colors[data_entry.target_idx].rgb_norm, data_entry.caption)
    
    def display_game(self, row_index):
        """
        Displays the full color context and caption for a index into the dataframe.

        Takes an index and displays an annotated version of the round at that index.
        The target color is on left, and the clicked color is outlined in red.
        
        Args:
            row_index - the index into the dataframe of the round to display.
        """
        data_entry = self.entries[row_index]        
        bottom_corners = [(-600,-400), (-200,-400), (200,-400)]
        width = 400
        height = 800
        
        # generate rectangles with their colors
        color_rects = []
        for i, corner in enumerate(bottom_corners):
            # target color will always be on left, we want clicked color
            # to stand out with a thick (linewidth=5), red outline that is
            # at the top of the rectangle stack (zorder = 5) displayed
            outline_color, stack_order, line_width =\
                ("red", 5, 5) if i == data_entry.click_idx else ("black", 0, None)
            color_rects.append(matplotlib.patches.Rectangle(corner, width, height,
                                                            facecolor=data_entry.colors[i].rgb_norm, 
                                                            edgecolor=outline_color,
                                                            zorder = stack_order,
                                                            linewidth=line_width))
        # add rectangles to figures
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for rect in color_rects:
            ax.add_patch(rect)

        # hide axis labels
        ax.set_yticks([])
        ax.set_xticks([])
        
        # display caption and game
        print("{}) Game: {} Round: {}\n{}".format(row_index, self.data.iloc[row_index]['gameid'],
                                                          self.data.iloc[row_index]['roundNum'], data_entry.caption))
        plt.xlim([-600, 600])
        plt.ylim([-400, 400])
        plt.show()
    
    def __getitem__(self, key):
        return self.entries[key]
    
    def __len__(self):
        return len(self.entries)



class MonroeDataEntry():
    """
    Class used for storing and providing access to each row in the dataframe in a more
    intuitive way. Stores colors such that target color is always in position 0 then
    distr1 and then distr2 regardless of how they were presented to the subjects. 
    Also stores tokenized captions as well as which color was selected

    Note: this is intended more for useful exploration than for speed and efficiency
    """
    def __init__(self, k, raw_row):
        """
        Constructs MonroeDataEntry Object

        Args:
            k - the index of the entry in the MonroeData Pandas Dataframe.
            raw_row - the Pandas Dataframe row containing all of the data to parse.
        """
        # color order: [target, distr1, distr2]
        self.click_to_idx = {'target': 0, 'distr1':1, 'distr2':2}
        
        self.index = k
        self.colors = self.parse_colors(raw_row)
        self.caption = raw_row["contents"]
        self.tokens = nltk.word_tokenize(self.caption)
        self.target_idx = 0
        self.click_idx = self.click_to_idx[raw_row['clickStatus']]
        self.outcome = raw_row["outcome"]
        self.condition = raw_row["condition"]
    
    def _parse_color(self, raw_row, field):
        # get the dataframe keys by splitting a string - maintain HSL order
        color_ids = "{0}ColH {0}ColS {0}ColL {0}Status".format(field).split()
        color_vals = [raw_row[key] for key in color_ids]
        color = Color(*color_vals[:3])
        color_idx = self.click_to_idx[color_vals[-1]]
        return (color_idx, color)
            
    def parse_colors(self, raw_row):
        """
        Parses the colors into Color Objects from the raw_row.

        Args:
            raw_row - the Pandas Dataframe row containing the colors.
        """
        colors = [None, None, None]
        for field in ['click', 'alt1', 'alt2']:
            color_idx, color = self._parse_color(raw_row, field)
            colors[color_idx] = color
        return colors
    
    def __repr__(self):
        return self.caption

    def __eq__(self, other):
        return hash(other) == hash(self)

    def __hash__(self):
        hash_str = "{}".format(self.caption)
        for color in colors:
            hash_str += "{}".format(hash(color))
        return int(hash_str)

class Color():
    """
    Class used for storing various color formats that are used in the experiment data (HSL),
    and for plotting colors (RGB). Also stored are normalized versions of the colors with 
    coordinates 0 to 1 instead of their natural scales.

    Available color formats:
        1. RGB / RGB_NORM
        2. HSL / HSL_NORM
        3. HSV / HSV_NORM
    """

    def __init__(self, coordinate_1, coordinate_2, coordinate_3, space='hsl'):
        """
        Initializes Color object.
        
        Args:
            coordinate_x - the xth coordinate of the color
            space - whether the passed coordinates are in hsl or rgb

        """
        if space.lower() == 'hsl':
            self.hsl = [coordinate_1, coordinate_2, coordinate_3]
            self.rgb = self.hsl_to_rgb(self.hsl)
            
        elif space.lower() == "rgb":
            self.rgb = [coordinate_1, coordinate_2, coordinate_3]
            self.hsl = self.rgb_to_hsl(self.rgb)

        else:
            print("Unknown color space: {}. Please use HSL or RGB.".format(space))
            return
        
        self.hsv = self.hsl_to_hsv(self.hsl)
        
        self.hsl_norm = self.normalize_hsl(self.hsl)
        self.rgb_norm = self.normalize_rgb(self.rgb)
        self.hsv_norm = self.normalize_hsv(self.hsv)
    
    def normalize_hsl(self, hsl):
        """
        Normalizes hsl coordinates to lie between 0 and 1.
        """
        norm_circle = cols.Normalize(0, 360)
        norm_hundred = cols.Normalize(0, 100)
        return [norm_circle(hsl[0]), norm_hundred(hsl[1]), norm_hundred(hsl[2])]
    
    def normalize_rgb(self, rgb):
        """
        Normalizes rgb coordinates to lie between 0 and 1.
        """
        norm_byte = cols.Normalize(0, 256)
        norm_color = [norm_byte(c) for c in rgb]
        return norm_color
    
    def normalize_hsv(self, hsv):
        """
        Normalizes hsv coordinates to lie between 0 and 1.
        """
        norm_circle = cols.Normalize(0, 360)
        norm_hundred = cols.Normalize(0, 100)
        return [norm_circle(hsv[0]), norm_hundred(hsv[1]), norm_hundred(hsv[2])]
    
    def rgb_to_hsl(self, rgb):
        """
        Converts rgb list to hsl.
        """
        norm_color = self.normalize_rgb(rgb)
        hls = cs.rgb_to_hls(*norm_color)
        hsl = [int(round(360*hls[0])), round(100*hls[2]), round(100*hls[1])]
        return hsl
    
    def hsl_to_rgb(self, hsl):
        """
        Converts hsl list to rgb.
        """
        # colorsys package takes colors in hls not hsl, so we swap the last two columns here.
        norm_color = self.normalize_hsl(hsl)
        norm_color[1], norm_color[2] = norm_color[2], norm_color[1]
        rgb = cs.hls_to_rgb(norm_color[0], norm_color[1], norm_color[2])
        rgb = [int(round(c*256)) for c in rgb]
        return rgb
    
    def hsl_to_hsv(self, hsl):
        """
        Converts hsl list to hsv.
        """
        # shamelessly taken from Will Monroes code and he based his code off of
        #   http://ariya.ofilabs.com/2008/07/converting-between-hsl-and-hsv.html
        h_in, s_in, l_in = hsl
        h_out = h_in
        s_in *= (l_in/100.0) if l_in <= 50.0 else (1.0 - (l_in/100.0))
        v_out = l_in + s_in
        s_out = (200.0 * s_in / v_out) if v_out else 0.0
        return [h_out, s_out, v_out]
    
    def __repr__(self):
        return 'hsl: {}, rgb {}, hsv {}'.format(self.hsl, self.rgb, self.hsv)
    
    def __eq__(self, other):
        # return np.allclose(other.hsl, self.hsl) and np.allclose(other.rgb, self.rgb)
        return hash(other) == hash(self)
    
    def __hash__(self):
       return int(''.join(['{:02X}'.format(int(coord)) for coord in rgb]), 16)

        
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="filteredCorpus.csv", help="csv file with color data for dataframe. Download from http://cocolab.stanford.edu/datasets/colors.html")
    parser.add_argument("--entries_file", type=str, default=None, help="path to pickled list of MonroeDataEntries for faster loading")
    parser.add_argument("--only_speaker", type=bool, default=True, help="Set to true for data to have only last speaker utterance and no listener utterances per round. False to include all utterances")
    args = parser.parse_args()

    # load in color dataset and adjust figure size so it's not huge
    # (the plot will really only work on iterm2 with this package installed: https://github.com/daleroberts/itermplot)
    monroe_dataset = MonroeData(args.data_file, args.entries_file, args.only_speaker)
    plt.figure(figsize=(8,3))
    monroe_dataset.display_game(1)
