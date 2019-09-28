import torch
import torch.nn as nn
import numpy as np

import time
import math
from queue import PriorityQueue # for beam search in Literal Speaker


##########################################################
# torch.nn.Module subclasses
##########################################################

# FOR LISTENER
class CaptionEncoder(nn.Module):
    """
    Implements the Monroe et al., 2017 color caption encoder.

    The CaptionEncoder takes a list of possible colors and a
    caption and converts the caption into a distribution over
    the colors. The color with the most probability mass is 
    most likely to be refered to by the caption.
    
    To do this, the network first embeds the caption into a
    vector space, then it runs an LSTM over the vectors.
    From the last hidden state it produces a mean vector and
    covariance matrix giving a gaussian distribution over
    the color space. This vector and matrix are used to
    score each candidate color and these scores are softmaxed.
    For more details on the scoring see Monroe et al., 2017.
    """
    
    def __init__(self, embed_dim, hidden_dim, vocab_size, color_dim, weight_matrix=None):
        """
        Initializes CaptionEncoder.

        This initialization is based on the released code from Monroe et al., 2017.
        Further inline comments explain how this code differs.

        Args:
            embed_dim -  the output dimension for embeddings. Should be 100.
            hidden_dim - dimension used for the hidden state of the LSTM. Should be 100.
            vocab_size - the input dimension to the embedding layer.
            color_dim - number of dimensions of the input color vectors, the mean
                vector and the covariance matrix. Usually will be 54 (if using
                color_phi_fourier with resolution 3.)
        
        """
        super(CaptionEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if weight_matrix is not None:
            self.embed.load_state_dict({'weight': weight_matrix})
        
        # Various notes based on Will Monroe's code
        # should initialize bias to 0: https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L383
        # √ he also DOESN'T use dropout for the base listener 
        # also non-linearity is "leaky_rectify" - I can't implement this without rewriting lstm :(, so I'm just going
        # to hope this isn't a problem
        # √ also LSTM is bidirectional (https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/listener.py#L713)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        self.mean = nn.Linear(2*hidden_dim, color_dim)
        # covariance matrix is square, so we initialize it with color_dim^2 dimensions
        # we also initialize the bias to be the identity bc that's what Will does
        covar_dim = color_dim*color_dim
        self.covariance = nn.Linear(2*hidden_dim, covar_dim)
        self.covariance.bias.data = torch.tensor(np.eye(color_dim), dtype=torch.float).flatten()
        self.logsoftmax = nn.LogSoftmax(dim=0)

        self.color_dim = color_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, colors, caption):
        """Turns caption into distribution over colors."""
        embeddings = self.embed(caption)
        output, (hn, cn) = self.lstm(embeddings)
        
        # we only care about last output (first dim is batch size) 
        # here we are concatenating the the last output vector of the forward direction (at index -1)
        # and the last output vector of the first direction (at index 0)
        output = torch.cat((output[:, -1, :self.hidden_dim],
                            output[:, 0, self.hidden_dim:]), 1) 

        output_mean = self.mean(output)[0]
        output_covariance = self.covariance(output)[0]
        covar_matrix = output_covariance.reshape(-1, self.color_dim) # make it a square matrix again
        
        
        # now compute score: -(f-mu)^T Sigma (f-mu)
        output_mean = output_mean.repeat(3,1)
        diff_from_mean = colors[0] - output_mean
        scores = torch.matmul(diff_from_mean, covar_matrix)
        scores = torch.matmul(scores, diff_from_mean.transpose(0,1))
        scores = -torch.diag(scores)
        distribution = self.logsoftmax(scores)
        return distribution 
    
    def init_hidden_and_context(self):
        # first 2 because one vector for each direction.
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))
       

# FOR SPEAKER
class ColorEncoder(nn.Module):
    """
    Encodes the context colors with an LSTM for the Literal Speaker.
    """
    
    def __init__(self, color_dim, hidden_dim):
        """
        Initialize ColorEncoder.

        Args:
            color_dim - dimensions of the color vectors to be encoded.
            hidden_dim - the dimension of the encoding LSTM hidden state.
        """
        super(ColorEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.color_dim = color_dim
        self.color_lstm = nn.LSTM(color_dim, hidden_dim, batch_first=True)

    def forward(self, colors):
        """
        Encodes colors.

        Colors should be any order with the target LAST.
        """
        color_states = self.init_hidden_and_context()
        color_output, (hn, cn) = self.color_lstm(colors, color_states)
        # target is last - return hidden representation, why not context i have no idea
        color_output = color_output[:, -1, :]
        return color_output 

    def init_hidden_and_context(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

class CaptionGenerator(nn.Module):
    """
    Generates/gives the likelihood of a caption from a list of colors.

    The CaptionGenerator is a conditional language model
    that gives a probability of a caption given a list of
    colors. It can also be used as a caption generator.

    First it encodes the colors using a ColorEncoder module. The target
    color is last. Then it uses the embedding that is generated to
    calculate probabilities of subsequent tokens using an LSTM.
    """
    def __init__(self, color_in_dim, color_dim, vocab_size, embed_dim, speaker_hidden_dim):
        """
        Initialize CaptionGenerator object.

        Args:
            color_in_dim - dimension of each color vector. Usually 54.
            color_dim - dimension of the color embedding. Usually 100 so is
                the same as the vocab embeddings.
            vocab_size - the input dimension for the embedding layer. Number
                of vocab items.
            embed_dim - size of each token embedding. Usually 100.
            speaker_hidden_dim - dimension of the hidden state of the
                generator's LSTM.
        """
        super(CaptionGenerator, self).__init__()
        
        self.color_encoder = ColorEncoder(color_in_dim, color_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.speaker_lstm = nn.LSTM(embed_dim + color_dim, speaker_hidden_dim, batch_first=True)
        self.linear = nn.Linear(speaker_hidden_dim, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.hidden_dim = speaker_hidden_dim
        
    def forward(self, colors, captions):
        """Gets probabilities of captions given colors."""
        # first get color context encoding
        color_features = self.color_encoder(colors)

        # all teacher forcing during training in this function (i.e. don't feed output of network
        # back in as next input)
        embeds = self.embed(captions)
        #print("Embed Shape:", embeds.shape)
        
        color_features = color_features.repeat(1, captions.shape[1], 1) # repeat for number of tokens
        
        inputs = torch.cat((embeds, color_features), dim=2) # cat along the innermost dimension
        #print("Input Shape:", inputs.shape)
        hiddens, _ = self.speaker_lstm(inputs) # hidden and context default to 0
        #print("Hiddens Shape:", hiddens.shape)
        outputs = self.linear(hiddens)
        #print("Outputs Shape:", outputs.shape)
        output_norm = self.logsoftmax(outputs)
        return output_norm
        
            
            
# FOR IMAGINATIVE LISTENER
class ColorGenerator(nn.Module):
    """
    Generates colors from a caption.

    This network treats the reference game as a regression task.
    It takes in a caption, encodes it with an LSTM and uses the
    encoding to generate the RGB values of the predicted target color.

    Only used in the 224U project.
    """
    def __init__(self, embed_dim, hidden_dim, vocab_size, color_in_dim, color_hidden_dim, weight_matrix=None):
        """
        Initializes ColorGenerator object.

        Args:
            embed_dim - size of token embedding output. Usually 100.
            hidden_dim - size of LSTM hiddem vector. Usually 100.
            vocab_size - size of token embedding input. Number of tokens in vocabulary.
            color_in_dim - dimensions of color vectors. Usually 54.
            color_hidden_dim - dimension of intermediate fully connected layer.
            weight_matrix - optional GLoVe embedding matrix to initialize embeddings with.
        """
        super(ColorGenerator, self).__init__()
        # Embedding/LSTM for words
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if weight_matrix is not None:
            self.embed.load_state_dict({'weight': weight_matrix})

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Linear layers for colors
        #self.color_rnn = nn.RNN(color_in_dim, color_hidden_dim, bidirectional=True, batch_first=True)
        self.color_encode = nn.Linear(2*color_in_dim, color_hidden_dim)

        # now generate color from embedding dim:
        # two linear layers to allow for some non-linear function of the hidden state elements
        # if this leads to overfitting I'll take it out
        self.linear1 = nn.Linear(2*hidden_dim + color_hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 3) # 3 for rgb

        self.hidden_dim = hidden_dim
        self.color_hidden_dim = color_hidden_dim

    def forward(self, colors, caption):
        # get caption encodings
        embeddings = self.embed(caption)
        output, _ = self.lstm(embeddings)

        # get color encodings
        colors = colors.reshape(1, 1, -1)
        color_encodings = nn.functional.relu(self.color_encode(colors))
        color_encodings = color_encodings.squeeze(0)

        # only care about vector of last sequence
        output = torch.cat((output[:, -1, :self.hidden_dim],
                            output[:, 0, self.hidden_dim:]), 1)
        # combine colors and caption
        combined_output = torch.cat((output, color_encodings), 1)
        output = self.linear1(combined_output)
        output = nn.functional.relu(output)
        output = nn.functional.softmax(self.linear2(output), dim=1)
        return output

# FOR COLOR-ONLY BASELINE
class ColorSelector(nn.Module):
    """
    Attempts to select the target color given only the colors and no caption.

    Rather than use any fancy RNNs or LSTMs, the prediction is made using two
    fully connected layers.

    This is used as a baseline in the 224U project.
    """
    def __init__(self, color_dim):
        """
        Initializes ColorSelector object.

        Args:
            color_dim - the dimension of the color vectors.
        """
        super(ColorSelector, self).__init__()
        self.linear1 = nn.Linear(3*color_dim, color_dim)
        self.linear2 = nn.Linear(color_dim, 3)
        self.nll = nn.LogSoftmax(dim=2)
    
    def forward(self, colors):
        colors = colors.reshape(1, 1, -1)
        output = self.linear1(colors)
        output = nn.functional.relu(output)
        output = self.linear2(output)
        output = self.nll(output)
        return output        

# FOR Single Color with Linear-layer as Encoder
class SingleColorCaptionGenerator(nn.Module):
    """
    Generates a caption of a single color.

    As opposed to the color reference game where the task
    is to caption a color out of a list of colors, this
    model captions a single color.
    """

    def __init__(self, color_in_dim, color_dim, vocab_size, embed_dim, speaker_hidden_dim):
        super(SingleColorCaptionGenerator, self).__init__()

        # for colors
        self.color_encoder = nn.Linear(color_in_dim, color_dim)

        # for text
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.speaker_lstm = nn.LSTM(embed_dim + color_dim, speaker_hidden_dim, batch_first=True)
        self.linear = nn.Linear(speaker_hidden_dim, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, color, captions):
        color_features = self.color_encoder(color)
        caption_features = self.embed(captions)

        # concatenate color features to the embedding of each token
        color_features = color_features.repeat(1, captions.shape[1], 1) # repeat for number of tokens
        inputs = torch.cat((caption_features, color_features), dim=2) # concatenate along the innermost dimension

        # just use teacher forcing here for now (i.e. don't feed predictions back into network some percentage of the time
        #). Not doing this leads to greater instability but is cleaner to implement
        hiddens, _ = self.speaker_lstm(inputs)
        outputs = self.linear(hiddens)
        output_norm = self.logsoftmax(outputs)
        return output_norm


###########################################################
# Wrappers for torch models to handle training/evaluating # 
###########################################################
class PytorchModel():
    """

    """
    def __init__(self,  model, optimizer=torch.optim.Adadelta,
                 criterion=nn.NLLLoss, lr=0.2, num_epochs=30):
        """
        Right now this is kind of ugly because you have to pass literally all of the experiment arguments
        to this constructor. 
        """
        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion

        # misc args:
        self.lr = lr
        self.num_epochs = num_epochs

        # for reproducibility, store training pairs
        self.features  = None
        self.target = None

        # also make sure the model has been initialized before we do anything
        self.initialized = False

    def fit(self, X, y, validation_size = 3000):
        """
        The main sklearn-like function that takes input features X (in
        a list or array - we use color features and caption features
        for pytorch models) and targets, y, also a np array. The function
        just stores them and calls its internal train model function
        """

        # set up validation data if necessary
        if validation_size > 0:
            self.features = X[:-validation_size]
            self.targets = y[:-validation_size]
            self.val_features= X[-validation_size:]
            self.val_targets = y[-validation_size:]
        else:
            self.features = X
            self.targets = y 
            self.val_features = None
            self.val_targets = None

        self.train_model()

    # from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def init_model(self, **model_params):
        """
        Interesting quirk - in most cases, the model takes the vocabulary size
        as an input, but there's no way for the user to know what the vocab
        size is before calling the init_model method. Even though it's kinda ugly,
        what we can do instead is pass all of the model params (named) minus
        the vocab size to this function, and it will create the model. I don't 
        really like this but it works for now
        """
        if self.initialized:
            return

        self.model = self.model(**model_params)
        self.initialized = True

    def train_iter(self, caption, colors, target, criterion):
        """
        Generate predictions based on the caption and color. Caculate loss against
        the target using the criterion and optimize using the optimizer. All subclasses
        will do this a little bit differently, so do implement this in the subclasses
        """
        pass


    def train_model(self):
        if not self.initialized:
            print("Make sure you initialize the model with the parameters you want")
            return


        optimizer = self.optimizer(lr=self.lr, params=self.model.parameters())
        criterion = self.criterion()

        start_time = time.time()
        store_losses_every = 100
        print_losses_every = 1000
        self.stored_losses = [] # theoretically we store losses so we can plot them later - 
                                # I don't think this part of the code works though. What we
                                # can do instead is take a few thousand or so training examples
                                # out and use them for "evaluation" every 1 epoch or so
        for epoch in range(self.num_epochs):
            print("---EPOCH {}---".format(epoch))
            stored_loss_total = 0
            print_loss_total = 0

            for i, pair in enumerate(self.features):
                caption, colors = pair
                caption = torch.tensor([caption], dtype=torch.long)
                colors = torch.tensor([colors], dtype=torch.float)
                target = torch.tensor([self.targets[i]]) 


                optimizer.zero_grad()
                loss = self.train_iter(caption, colors, target, criterion)
                loss.backward()
                optimizer.step()
                stored_loss_total += loss.item()
                print_loss_total += loss.item()

                if i % print_losses_every == 0:
                    print_loss_avg = print_loss_total / print_losses_every
                    print("{} ({}:{} {:.2f}%) {:.4f}".format(self.asMinutes(time.time() - start_time),
                                                      epoch, i, i/len(self.features)*100,
                                                      print_loss_avg))
                    print_loss_total = 0

                if i % store_losses_every == 0:
                    stored_loss_avg = stored_loss_total / store_losses_every
                    self.stored_losses.append(stored_loss_avg)
                    stored_loss_total = 0

            # run validation data after every epoch
            self.validate_model(epoch, criterion)

    def validate_model(self, epoch_num, criterion):
        if self.val_features is None:
            return

        # run validation
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, pair in enumerate(self.val_features):
                caption, colors = pair
                caption = torch.tensor([caption], dtype=torch.long)
                colors = torch.tensor([colors], dtype=torch.float)
                target = torch.tensor([self.val_targets[i]]) 
                total_loss += self.train_iter(caption, colors, target, criterion).item()
            print("="*25)
            print("AFTER EPOCH {} - AVERAGE VALIDATION LOSS: {}".format(epoch_num, total_loss / len(self.val_features)))
            print("="*25)
        self.model.train()


    def load_model(self, filename):
        """
        Load model from saved file at filename
        """
        if not self.initialized:
            self.init_model
        self.model.load_state_dict(torch.load(filename))

    def save_model(self, filename):
        """
        Save model to file at filename
        """
        torch.save(self.model.state_dict(), filename)

class LiteralListener(PytorchModel):

    def predict(self, X):
        """
        Produces and tracks model outputs
        """
        self.model.eval()
        model_outputs = [] # np.empty([len(X), len(X[0][1])]) # (num_entries, num_colors)

        for i, feature in enumerate(X):
            caption, colors = feature
            caption = torch.tensor([caption], dtype=torch.long)
            colors = torch.tensor([colors], dtype=torch.float)
            model_output_np = self.evaluate_iter((caption, colors)).view(-1).numpy()
            model_outputs.append(model_output_np)

        return np.array(model_outputs)


    def train_iter(self, caption_tensor, color_tensor, target, criterion):
        """
        Iterates through a single training pair, querying the model, getting a loss and
        updating the parameters. (TODO: addd some kind of batching to this).

        Very much inspired by the torch NMT example/tutorial thingy
        """
        input_length = caption_tensor.size(0)
        loss = 0

        model_output = self.model(color_tensor, caption_tensor)
        model_output = model_output.view(1, -1)

        loss += criterion(model_output, target)

        return loss


    def evaluate_iter(self, pair):
        """
        Same as train_iter except don't use an optimizer and gradients or anything
        like that
        """
        with torch.no_grad():
            caption_tensor, color_tensor = pair
            model_output = self.model(color_tensor, caption_tensor)

            model_output = model_output.view(1, -1)
            return model_output

# the ConditionPredictor uses the exact same structure as the literal speaker, so
# let's make it use the same class
ConditionPredictor = LiteralListener

# Used for literal speaker

class BeamNode():
    def __init__(self, log_prob, tokens, ended):
        self.log_prob = log_prob
        self.tokens = tokens
        self.ended = ended
    
    def score(self):
        return -self.log_prob
    
    def __eq__(self, other):
        return self.score() == other.score()

    def __lt__(self, other):
        return self.score() < other.score()

class LiteralSpeaker(PytorchModel):
    
    def __init__(self, model, max_gen_len=20, **kwargs):
        super(LiteralSpeaker, self).__init__(model, **kwargs)
        self.max_gen_len = max_gen_len

    def train_iter(self, caption_tensor, color_tensor, target, criterion):
        """
        Iterates through a single training pair, querying the model, getting a loss and
        updating the parameters. (TODO: addd some kind of batching to this).

        Very much inspired by the torch NMT example/tutorial thingy
        """
        loss = 0

        # target color is FIRST in the tensor, so flip it so it's LAST
        color_tensor = np.flip(color_tensor.numpy(), axis=1).copy()
        color_tensor = torch.from_numpy(color_tensor);
        model_output = self.model(color_tensor, caption_tensor)

        # we don't care about the last prediction, because nothing follows the final </s> token
        model_output = model_output[:,:-1,:].squeeze(0) # go from 1 x seq_len x vocab_size => seq_len x vocab_size
                                                        # for calculating loss function:
                                                        # see here for details when implementing batching
                                                        # https://discuss.pytorch.org/t/calculating-loss-for-entire-batch-using-nllloss-in-0-4-0/17142/7

        # targets should be caption without start index: i.e. [the blue one </s>] so we can predict
        # next tokens from input like [<s> the blue one]

        target = target.squeeze()
        loss += criterion(model_output, target)

        return loss
        
    def predict(self, X, sample=1, beam_width=5):
        """
        There are a bunch of choices for what we might want to do here:
        1. X contains just color contexts and we incrementatlly sample 
           using beam search to generate a caption and return that.
        2. X contains color contexts and captions and we return the
           softmax probabilities assigned to each token in the caption
           (for calculating perplexity or some notion of how probable
           our model believes the caption is given the color context)

        Right now 1 is implemented, but I think 2 is better in the long
        run. We are greedily taking the most likely token
        """
        # Create a tensor with just the starting token
        all_tokens = []
        max_gen_len = self.max_gen_len

        self.model.eval()
        with torch.no_grad():
            for i, feature in enumerate(X):
                caption, colors_before_roll = feature
                caption = torch.tensor([caption], dtype=torch.long)
                predictions_all_targets =  []
                for r in range(len(colors_before_roll)): # Roll through the different possible targets being last in the same order the colors are presented
                    colors = np.roll(colors_before_roll, shift=-r, axis=0) # hence the -r
                    colors = torch.tensor([colors], dtype=torch.float)
                    colors = np.flip(colors.numpy(), axis=1).copy()
                    colors = torch.from_numpy(colors)

                    beam_nodes = PriorityQueue()
                    ended_list = []

                    tokens = caption[:, 0].view(-1, 1) # begin at start token
                    start = BeamNode(0, tokens, False)
                    beam_nodes.put(start)
                
                    for i in range(max_gen_len + 1):
                        node = beam_nodes.get()
                        if node.ended:
                            ended_list.append(np.array(node.tokens[0].numpy()))
                            if len(ended_list) == sample:
                                break
                        else:
                            tokens = node.tokens
                            vocab_preds = self.model(colors, tokens)[:,-1:,:] # just distribution over last token
                            log_probs, prediction_indices = vocab_preds.topk(beam_width, dim=2)  # taking the topk predictions
                            for j in range(beam_width):
                                prediction_index = prediction_indices[:,-1,j:j+1] # a single prediction
                                log_prob = log_probs[0][0][j].item()
                                updated_tokens = tokens.clone()
                                updated_tokens = torch.cat((updated_tokens, prediction_index), dim=1)
                                updated_log_prob = node.log_prob + log_prob
                                ended = ((i == max_gen_len - 1) or (prediction_index.item() == caption[:, -1].view(-1, 1)))
                                new_node = BeamNode(updated_log_prob, updated_tokens, ended)
                                beam_nodes.put(new_node)
                    predictions_all_targets += ended_list
                if sample == 1: # for backwards compatability
                    all_tokens.append(np.array(predictions_all_targets))
                else:
                    all_tokens.append(predictions_all_targets)
        return all_tokens

class LiteralSpeakerScorer(LiteralSpeaker):
    """
    Only difference between this and above Literal Speaker is the predict function. Above we care about generating the next tokens with beam
    search but here we want to use the model to get the log probability of the passed feature
    """

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            all_color_outputs = []
            for i, feature in enumerate(X):
                caption, colors_before_roll = feature
                caption_tensor = torch.tensor([caption])
                
                each_color_outputs = []
                for r in range(len(colors_before_roll)):
                    # we want to get probability that each color is target.
                    # r is negative so indices of color output line up with indices of colors passed
                    colors = np.roll(colors_before_roll, shift=-r, axis=0)
                    # flip colors so target is last - consistent with target being last
                    colors = np.flip(colors, axis=0).copy()
                    color_tensor = torch.tensor([colors])

                    results = self.model(color_tensor, caption_tensor)
                    each_color_outputs.append(results.squeeze()[:-1])
                
                all_color_outputs.append(each_color_outputs)
            return all_color_outputs

class ImaginativeListener(PytorchModel):
    def __init__(self, model, use_color=True, **kwargs):
        super(ImaginativeListener, self).__init__(model, **kwargs)
        # the Imaginative Listener might not use the colors at all, but this could be difficult
        self.use_color = use_color

    def train_iter(self, caption_tensor, color_tensor, target_tensor, criterion):
        loss = 0

        if self.use_color:
            color_tensor = color_tensor[:, 1:3, :] # don't include the target at index 0
            model_output = self.model(color_tensor, caption_tensor)
        else:
            model_output = self.model(caption_tensor)

        if isinstance(criterion, nn.MSELoss):
            loss += criterion(model_output, target_tensor.type(torch.FloatTensor))
        else:
            model_output = model_output.type(torch.DoubleTensor)
            label = torch.tensor(1, dtype=torch.double)
            loss += criterion(model_output, target_tensor.detach(), label)

        return loss

    def predict(self, X):
        model_outputs = np.empty([len(X), 3])
        self.model.eval()
        with torch.no_grad():
            for i, feature in enumerate(X):
                caption, colors = feature
                caption_tensor = torch.tensor([caption], dtype=torch.long)
                color_tensor = torch.tensor([colors], dtype=torch.float)
                if self.use_color:
                    color_tensor = color_tensor[:, 1:3, :] # don't include the target
                    model_output = self.model(color_tensor, caption_tensor)
                else:
                    model_output = self.model(caption_tensor)

                model_output_np = model_output.view(-1).numpy()
                model_outputs[i] = model_output_np
        return np.array(model_outputs)
    
    
class PragmaticListener():
    def __init__(self, literal_listener, literal_speaker, alpha=0.544, sample=5, beam_width=5):
        self.literal_listener = literal_listener
        self.literal_speaker = literal_speaker
        self.alpha = alpha
        self.sample = sample
        self.beam_width = beam_width
        self.prior = 1/3. # uniform prior over colors
    
    def get_utterance_universe(self, feature):
        # sample from literal speaker model
        caption, _ = feature
        U = self.literal_speaker.predict([feature], sample=self.sample, beam_width=self.beam_width)[0]
        U = [caption] + U
        return U
    
    def calculate_l0_log(self, feature, utterances):
        _, colors = feature
        feature_modified = []
        for u in utterances:
            feature_modified.append([u, colors])
        return self.literal_listener.predict(feature_modified)
    
    def calculate_s1(self, l0):
        s1 = l0.T * self.alpha
        # could subtract costs here
        s1 = np.exp(s1)
        s1 = self.row_norm(s1)
        return s1
    
    def calculate_l2(self, s1):
        l2 = s1.T * self.prior
        l2 = self.row_norm(l2)
        return l2
            
    def row_norm(self, a):
        row_sum = a.sum(axis=1, keepdims=True)
        return a / row_sum
    
    def safelog(self, vals):
        with np.errstate(divide='ignore'):
            return np.log(vals)
    
    def predict(self, X):
        model_outputs = np.empty([len(X), 3])
        for i, feature_concat in enumerate(X): 
            if len(feature_concat) != 2:
                print("Pragmatic listener requires both literal listener and literal speaker features")
                return
            feature_listener, feature_speaker = feature_concat
            utterances = self.get_utterance_universe(feature_speaker)            
            l0 = self.calculate_l0_log(feature_listener, utterances)
            s1 = self.calculate_s1(l0)
            l2 = self.calculate_l2(s1)
            model_outputs[i] = self.safelog(l2[0])
                        
        return np.array(model_outputs)


class ColorOnlyBaseline(PytorchModel):
    
    def train_iter(self, caption_tensor, color_tensor, target_tensor, criterion):
        model_output = self.model(color_tensor)
        loss = criterion(model_output.view(1, -1), target_tensor)
        return loss
        
    
    def predict(self, X):
        model_outputs = np.empty([len(X), 3])
        self.model.eval()
        with torch.no_grad():
            for i, feature in enumerate(X):
                caption, colors = feature
                color_tensor = torch.tensor([colors], dtype=torch.float)
                model_output = self.model(color_tensor)
                            
                model_output_np = model_output.view(-1).numpy()
                model_outputs[i] = model_output_np
        return model_outputs
                
