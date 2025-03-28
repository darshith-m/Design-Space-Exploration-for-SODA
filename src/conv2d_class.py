"""Python file for 2D convolution layer class"""

from utilities import get_factors

class Conv2D:
    """Class to store 2D convolution layer information"""

    def __init__(self, args, input_tensor_shape, kernel_tensor_shape,
                 output_tensor_shape, dilations, strides):

        # Initialize parameters from arguments
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll

        # Store file path of the current layer
        self.file_path = None

        # Store dilations and strides
        self.dilations = dilations
        self.strides = strides
        print("---------------------------------------------------")
        print(f"Dilations: {self.dilations}")
        print(f"Strides: {self.strides}")

        # Store input tensor shape details
        self.input_batch = input_tensor_shape[0]
        self.input_width = input_tensor_shape[1]
        self.input_height = input_tensor_shape[2]
        self.input_channel = input_tensor_shape[3]
        print(f"Input batch size: {self.input_batch}")
        print(f"Input width: {self.input_width}")
        print(f"Input height: {self.input_height}")
        print(f"Input channels: {self.input_channel}")

        # Store output tensor shape details
        self.output_batch = output_tensor_shape[0]
        self.output_width = output_tensor_shape[1]
        self.output_height = output_tensor_shape[2]
        self.output_channel = output_tensor_shape[3]
        print(f"Output batch size: {self.output_batch}")
        print(f"Output width: {self.output_width}")
        print(f"Output height: {self.output_height}")
        print(f"Output channels: {self.output_channel}")

        # Store kernel tensor shape details
        self.kernel_width = kernel_tensor_shape[0]
        self.kernel_height = kernel_tensor_shape[1]
        self.kernel_input_channel = kernel_tensor_shape[2]
        self.kernel_output_channel = kernel_tensor_shape[3]
        print(f"Kernel width: {self.kernel_width}")
        print(f"Kernel height: {self.kernel_height}")
        print(f"Kernel input channels: {self.kernel_input_channel}")
        print(f"Kernel output channels: {self.kernel_output_channel}")

        self.folded_input_channel = None
        self.folded_output_channel = None

        # Calculate FLOP count for the convolution operation
        self.flop_count = self.output_batch * \
            self.output_width * self.output_height * self.output_channel * \
            self.kernel_width * self.kernel_height * self.input_channel * 2

        # Initialize number of tiles to 1
        self.no_of_tiles = 1

        # Adjust output channels if greater than 1
        if self.output_channel > 1:
            self.folded_output_channel = 1
            self.no_of_tiles *= self.output_channel / self.folded_output_channel
            print("Folded output channel to 1 due to output_channel > 1")

        # Adjust input channels based on tiling or permutation
        if self.tile or self.permute:
            if self.input_height <= 16 and self.input_channel > 128:
                factors = get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 128]
                self.folded_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel / self.folded_input_channel
            elif self.input_height > 16 and self.input_height <= 32 and self.input_channel > 32:
                factors = get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 32]
                self.folded_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel / self.folded_input_channel
            elif self.input_height > 32 and self.input_height <= 64 and self.input_channel > 8:
                factors = get_factors(self.input_channel)
                factors_list = [factor for factor in factors if factor <= 8]
                self.folded_input_channel = factors_list[-1]
                self.no_of_tiles *= self.input_channel / self.folded_input_channel
            elif self.input_height > 64 and self.input_channel > 1:
                self.folded_input_channel = 1
                self.no_of_tiles *= self.input_channel / self.folded_input_channel
        # Adjust input channels based on unrolling
        elif self.unroll:
            if self.input_height <= 64:
                if self.kernel_height <= 3 and self.input_channel > 32:
                    factors = get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 32]
                    self.folded_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel / self.folded_input_channel
                elif self.kernel_height > 3 and self.kernel_height <= 5 and self.input_channel > 8:
                    factors = get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 8]
                    self.folded_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel / self.folded_input_channel
                elif self.kernel_height > 5 and self.kernel_height <= 7 and self.input_channel > 4:
                    factors = get_factors(self.input_channel)
                    factors_list = [factor for factor in factors if factor <= 4]
                    self.folded_input_channel = factors_list[-1]
                    self.no_of_tiles *= self.input_channel / self.folded_input_channel
                elif self.kernel_height > 7 and self.kernel_height <= 11 and self.input_channel > 1:
                    self.folded_input_channel = 1
                    self.no_of_tiles *= self.input_channel / self.folded_input_channel
            elif self.input_height > 64 and self.input_channel > 1:
                self.folded_input_channel = 1
                self.no_of_tiles *= self.input_channel / self.folded_input_channel

        print(f"Folded input channel to {self.folded_input_channel}")
