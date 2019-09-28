import torch
import numpy as np


def color_phi_id(color_list, space):
    """
    Function for turning a list of colors in the given space
    (with normalization marked by "_norm") into a feature function.
    
    This is just the identity feature function, so it's kind of boring,
    but we can imagine doing more complext things too (like fourier
    transform). We pass space as well in case you want to do different
    operations based on the space or only have a feature function work
    for HSL for example
    
    ex:
    color_list = [256, 0, 0] space = 'rgb'
    color_list = [1, 0, 0] space = 'rgb_norm'
    """
    return color_list

def color_phi_fourier(color_list, space, resolution=3):
    """
    This is lifted straight from https://github.com/futurulus/colors-in-context/blob/2e7b830668cd039830154e7e8f211c6d4415d30f/vectorizers.py#L650
    Haven't figured out how it works yet. but it seems to be the only feature function to get somewhat decent results so far
    """
    if space not in ("rgb_norm", "hsv_norm"):
        print("Space must be rgb_norm or hsv_norm to use fourier transform")
        return None

    resolution = [resolution for _ in color_list]
    colors = np.array([color_list])
    # I'm pretty sure ranges aren't actually used anywhere...
    if space == "rgb_norm":
        ranges = np.array([256, 256, 256])
    else:
        ranges = np.array([361, 101, 101])

    
    # Using a Fourier representation causes colors at the boundary of the
    # space to behave as if the space is toroidal: red = 255 would be
    # about the same as red = 0. We don't want this... so we divide
    # all of the rgb values by 2. (If we were doing this with hsv
    # we wouldn't divide the h value by two becaus it actually is 
    # polar, so 360 is close to 0 (both are red)
    if space == "rgb_norm":
        xyz = colors / 2
    else:
        xyz = colors / 2
        xyz[:, 0] *= 2 # this is the 'h' of the 'hsv'

    ax, ay, az = [np.arange(0, g) for g, r in zip(resolution, ranges)]
    gx, gy, gz = np.meshgrid(ax, ay, az)

    arg = (np.multiply.outer(xyz[:, 0], gx) +
           np.multiply.outer(xyz[:, 1], gy) +
           np.multiply.outer(xyz[:, 2], gz))

    repr_complex = np.exp(-2j * np.pi * (arg % 1.0)).swapaxes(1, 2).reshape((xyz.shape[0], -1))
    result = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
    return result[0]

class ColorFeaturizer:
    def __init__(self, featurizer, space, normalized=True, **kwargs):
        """
        Params:
        - featurizer is a function to apply to each color to extract the feature representation
        of that color
        - space is hsv, rgb, or hsl
        - normalized is whether all values used should be between 0 and 1 (normalized = True) or 
        take on their full values 0-360 for hue, 0-100 for value/lightness/saturation, 0-255 for rgb
        """
        self.featurizer = featurizer
        # by default, the target is the 0th entry of the original featurizer
        self.get_target = lambda permutation, entry: torch.argmin(permutation).view(-1)
        self.space = space
        self.normalized = normalized
        # these are other arguments that the featurizer function could need. They should in theory be
        # passed to the call to self.featurizer in the `to_tensor` method, but they aren't right now
        self.featurizer_kwargs = kwargs
    

    def to_color_lists(self, colors, normalized):
        """
        Convert from a list of color objects to a list of color values
        in the color space set by self.space
        """
        # non-standard, but use the space as the variable name
        # to access the color attribute directly from the color objects
        class_var_name = self.space
        if normalized:
            class_var_name = "{}_norm".format(class_var_name)
        return [color.__dict__[class_var_name] for color in colors], class_var_name
    
    def to_color_features(self, colors):
        """
        Given a list of colors, compute use the feature function to get
        a each color's features
        """
        color_lists, space = self.to_color_lists(colors, self.normalized) 
        color_lists = [self.featurizer(color_list, space) for color_list in color_lists]
        return np.array(color_lists)

    def to_tensor(self, colors):
        """
        Convert colors to tensors where the vectors are the given by applying
        the feature function self.featurizer to the colors 

        returns all colors as |colors| x |phi| torch tensor
        """
        color_features = self.to_color_features(colors)
        color_tensor = torch.tensor(color_features, dtype=torch.float)
        return color_tensor
    

    def shuffle_colors(self, color_features):
        """
        Randomly permute colors. Keep track of where each color ends up
        for training and error analysis. If targets is none, assume the target
        is the speaker's target i.e. the 0th element of the original tensor
        """
        permutation = np.random.permutation(len(color_features))
        color_features = np.array(color_features)
        return color_features[permutation], permutation

