"""Python file to create temporary MLIR files"""

import os
from utilities import get_folded_layer_info

class MlirFiles():
    """Class to create temporary MLIR file"""
    def __init__(self, args, layers):
        self.layers = layers
        self.model_name, _ = os.path.splitext(os.path.basename(args.read_mlir))
        self.permute = args.permute
        self.tile = args.tile
        self.unroll = args.unroll
        if self.permute:
            self.loop_optimizer = 'permute'
        elif self.tile:
            self.loop_optimizer = 'tile'
        elif self.unroll:
            self.loop_optimizer = 'unroll'

    def create_mlir_function(self, layer_name, current_layer):
        """Function to create MLIR files"""
        linalg_line, input_shape, kernel_shape, output_shape = None, None, None, None
        if layer_name.startswith("conv2d"):
            input_channel, output_channel = get_folded_layer_info(current_layer,
                                                                  layer_name, return_all=False)
            # Construct input, kernel, and output shapes
            input_shape = "x".join([str(current_layer.input_batch),
                        str(current_layer.input_width),
                        str(current_layer.input_height),
                        str(input_channel)])
            kernel_shape = "x".join([str(current_layer.kernel_width),
                         str(current_layer.kernel_height),
                         str(input_channel),
                         str(output_channel)])
            output_shape = "x".join([str(current_layer.output_batch),
                         str(current_layer.output_width),
                         str(current_layer.output_height),
                         str(output_channel)])
            # Get dilations and strides
            dilations = current_layer.dilations
            strides = current_layer.strides
            # Define linalg operation name and line
            linalg_name = "conv_2d_nhwc_hwcf"
            linalg_line = f"    linalg.{linalg_name} {{dilations = dense<{dilations}> : tensor<2xi64>, strides = dense<{strides}> : tensor<2xi64>}}"
        elif layer_name.startswith("depthwise_conv2d"):
            # Determine input channel, considering clipped value if available
            input_channel = get_folded_layer_info(current_layer, layer_name, return_all=False)
            # Construct input, kernel, and output shapes
            input_shape = "x".join([str(current_layer.input_batch),
                        str(current_layer.input_width),
                        str(current_layer.input_height),
                        str(input_channel)])
            kernel_shape = "x".join([str(current_layer.kernel_width),
                         str(current_layer.kernel_height),
                         str(input_channel)])
            output_shape = "x".join([str(current_layer.output_batch),
                         str(current_layer.output_width),
                         str(current_layer.output_height),
                         str(input_channel)])
            # Get dilations and strides
            dilations = current_layer.dilations
            strides = current_layer.strides
            # Define linalg operation name and line
            linalg_name = "depthwise_conv_2d_nhwc_hwc"
            linalg_line = f"    linalg.{linalg_name} {{dilations = dense<{dilations}> : tensor<2xi64>, strides = dense<{strides}> : tensor<2xi64>}}"
        elif layer_name.startswith("matmul"):
            output_width, output_height, kernel_width, kernel_height, input_width, input_height = \
                get_folded_layer_info(current_layer, layer_name, return_all=False)
            # Construct input, kernel, and output shapes
            input_shape = "x".join([str(current_layer.input_batch),
                        str(input_width),
                        str(input_height)])
            kernel_shape = "x".join([str(current_layer.kernel_batch),
                         str(kernel_width),
                         str(kernel_height)])
            output_shape = "x".join([str(current_layer.output_batch),
                         str(output_width),
                         str(output_height)])
            # Define linalg operation name and line
            linalg_name = "batch_matmul"
            linalg_line = f"    linalg.{linalg_name}"
        # Write the MLIR file with the constructed parameters
        self.write_mlir_file(layer_name, linalg_line, input_shape, kernel_shape, output_shape)

    def write_mlir_file(self, layer_name, linalg_line, input_shape, kernel_shape, output_shape):
        """Function to write MLIR files"""
        # Assuming self.layers[layer_name].file_name is already defined
        file_path = f"output/layers_{self.model_name}_{self.loop_optimizer}/{layer_name}.mlir"
        self.layers[layer_name].file_path = file_path
        # Get the directory name
        directory = os.path.dirname(file_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w', encoding='UTF-8') as file:
            file.write(f"func.func @main(%arg0: memref<{input_shape}xf32>, ")
            file.write(f"%arg1: memref<{kernel_shape}xf32>, ")
            file.write(f"%arg2: memref<{output_shape}xf32>) {{\n")
            file.write("  cf.br ^bb1\n")
            file.write("^bb1:  // pred: ^bb0\n")
            file.write("  soda.launch {\n")
            # Modify dilations and strides line to use direct variable substitution
            file.write(f"{linalg_line}")
            file.write(f" ins(%arg0, %arg1 : memref<{input_shape}xf32>, ")
            file.write(f"memref<{kernel_shape}xf32>) ")
            file.write(f"outs(%arg2 : memref<{output_shape}xf32>)\n")
            file.write("    soda.terminator\n")
            file.write("  }\n")
            file.write("  return\n")
            file.write("}\n")

    def execute(self):
        """Function to execute creation of MLIR files"""
        for layer_name, layer in self.layers.items():
            self.create_mlir_function(layer_name, layer)
        return self.layers
