"""Python file for fully connected layer class"""

from utilities import get_factors

class FullyConnected:
    """Class to store fully connected layer information"""

    def __init__(self, args, input_tensor_shape, kernel_tensor_shape, output_tensor_shape):

        # Initialize parameters from args
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll

        # Initialize file path
        self.file_path = None

        # Extract input tensor dimensions
        self.input_batch = input_tensor_shape[0]
        self.input_width = input_tensor_shape[1]
        self.input_height = input_tensor_shape[2]
        print("---------------------------------------------------")
        print(f"FullyConnected: Input batch size: {self.input_batch}")
        print(f"FullyConnected: Input width: {self.input_width}")
        print(f"FullyConnected: Input height: {self.input_height}")

        # Extract output tensor dimensions
        self.output_batch = output_tensor_shape[0]
        self.output_width = output_tensor_shape[1]
        self.output_height = output_tensor_shape[2]
        print(f"FullyConnected: Output batch size: {self.output_batch}")
        print(f"FullyConnected: Output width: {self.output_width}")
        print(f"FullyConnected: Output height: {self.output_height}")

        # Extract kernel tensor dimensions
        self.kernel_batch = kernel_tensor_shape[0]
        self.kernel_width = kernel_tensor_shape[1]
        self.kernel_height = kernel_tensor_shape[2]
        print(f"FullyConnected: Kernel batch size: {self.kernel_batch}")
        print(f"FullyConnected: Kernel width: {self.kernel_width}")
        print(f"FullyConnected: Kernel height: {self.kernel_height}")

        # Calculate FLOP count
        self.flop_count = self.output_batch * self.output_width * \
            self.output_height * self.kernel_width * 2

        # Initialize folded dimensions and tile count
        self.folded_output_width = None
        self.folded_output_height = None
        self.folded_input_width = None
        self.folded_input_height = None
        self.folded_kernel_width = None
        self.folded_kernel_height = None

        #Initialize number of tiles to 1
        self.no_of_tiles = 1

        # Adjust dimensions if tiling or permutation is enabled
        if self.tile or self.permute:
            if self.output_width > 128:
                factors = get_factors(self.output_width)
                factors_list = [factor for factor in factors if factor <= 128]
                self.folded_output_width = factors_list[-1]
                self.folded_input_width = factors_list[-1]
                print(f"FullyConnected: folded output width to {self.folded_output_width} due to output_width > 128")
                print(f"FullyConnected: folded input width to {self.folded_input_width} due to input_width > 128")
                self.no_of_tiles *= self.output_width / self.folded_output_width
            if self.output_height > 128:
                factors = get_factors(self.output_height)
                factors_list = [factor for factor in factors if factor <= 128]
                self.folded_output_height = factors_list[-1]
                self.folded_kernel_height = factors_list[-1]
                print(f"FullyConnected: folded output height to {self.folded_output_height} due to output_height > 128")
                print(f"FullyConnected: folded kernel height to {self.folded_kernel_height} due to kernel_height > 128")
                self.no_of_tiles *= self.output_height / self.folded_output_height
            if self.kernel_width > 128 and self.input_height > 128:
                factors = get_factors(self.kernel_width)
                factors_list = [factor for factor in factors if factor <= 128]
                self.folded_kernel_width = factors_list[-1]
                self.folded_input_height = factors_list[-1]
                print(f"FullyConnected: folded kernel width to {self.folded_kernel_width} due to kernel_width > 128")
                print(f"FullyConnected: folded input height to {self.folded_input_height} due to input_height > 128")
                self.no_of_tiles *= self.kernel_width / self.folded_kernel_width

        # Adjust dimensions if unrolling is enabled
        elif self.unroll:
            if self.output_width > 32:
                factors = get_factors(self.output_width)
                factors_list = [factor for factor in factors if factor <= 32]
                self.folded_output_width = factors_list[-1]
                self.folded_input_width = factors_list[-1]
                print(f"FullyConnected: folded output width to {self.folded_output_width} due to output_width > 32")
                print(f"FullyConnected: folded input width to {self.folded_input_width} due to input_width > 32")
                self.no_of_tiles *= self.output_width / self.folded_output_width
            if self.output_height > 32:
                factors = get_factors(self.output_height)
                factors_list = [factor for factor in factors if factor <= 32]
                self.folded_output_height = factors_list[-1]
                self.folded_kernel_height = factors_list[-1]
                print(f"FullyConnected: folded output height to {self.folded_output_height} due to output_height > 32")
                print(f"FullyConnected: folded kernel height to {self.folded_kernel_height} due to kernel_height > 32")
                self.no_of_tiles *= self.output_height / self.folded_output_height
            if self.kernel_width > 32 and self.input_height > 32:
                factors = get_factors(self.kernel_width)
                factors_list = [factor for factor in factors if factor <= 32]
                self.folded_kernel_width = factors_list[-1]
                self.folded_input_height = factors_list[-1]
                print(f"FullyConnected: folded kernel width to {self.folded_kernel_width} due to kernel_width > 32")
                print(f"FullyConnected: folded input height to {self.folded_input_height} due to input_height > 32")
                self.no_of_tiles *= self.kernel_width / self.folded_kernel_width
