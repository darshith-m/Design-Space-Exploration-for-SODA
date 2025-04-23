"""Python file for utility functions"""

def get_factors(n):
    """Return the factors of n."""
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

def get_layer_info(layer, layer_name, configuration):
    """Function to get the actual layer information"""
    # Set the actual convolution layer's data
    if layer_name.startswith("conv2d"):
        return [
            configuration,
            layer.strides, layer.dilations,
            layer.input_batch,
            layer.input_width, layer.input_height, layer.input_channel,
            layer.kernel_width, layer.kernel_height,
            layer.kernel_input_channel, layer.kernel_output_channel,
            layer.output_batch,
            layer.output_width, layer.output_height,
            layer.output_channel
        ]
    # Set the actual depthwise convolution layer's data
    elif layer_name.startswith("depthwise_conv2d"):
        return [
            configuration,
            layer.strides, layer.dilations,
            layer.input_batch,
            layer.input_width, layer.input_height, layer.input_channel,
            layer.kernel_width, layer.kernel_height,
            layer.kernel_input_channel, layer.kernel_multiplier,
            layer.output_batch,
            layer.output_width, layer.output_height,
            layer.output_channel, layer.output_multiplier
        ]
    # Set the actual fully connection layer's data
    elif layer_name.startswith("matmul"):
        return [
            configuration,
            layer.input_batch, layer.input_width, layer.input_height,
            layer.kernel_batch, layer.kernel_width, layer.kernel_height,
            layer.output_batch, layer.output_width, layer.output_height
        ]

def get_folded_layer_info(layer, layer_name, return_all):
    """Function to get the folded layer information"""
    # Set the folded convolution layer's row data
    if layer_name.startswith("conv2d"):
        input_channel = layer.folded_input_channel or layer.input_channel
        output_channel = layer.folded_output_channel or layer.output_channel
        if return_all:
            return [
                layer.input_batch,
                layer.input_width, layer.input_height, input_channel,
                layer.kernel_width, layer.kernel_height,
                input_channel, output_channel,
                layer.output_batch,
                layer.output_width, layer.output_height, output_channel,
                layer.no_of_tiles
            ]
        else:
            return input_channel, output_channel
    # Set the folded depthwise convolution layer's data
    elif layer_name.startswith("depthwise_conv2d"):
        input_channel = layer.folded_input_channel or layer.input_channel
        if return_all:
            return [
                layer.input_batch,
                layer.input_width, layer.input_height, input_channel,
                layer.kernel_width, layer.kernel_height, input_channel,
                layer.output_batch,
                layer.output_width, layer.output_height, input_channel,
                layer.no_of_tiles
            ]
        else:
            return input_channel
    # Set the folded fully connected layer's data
    elif layer_name.startswith("matmul"):
        output_width = layer.folded_output_width or layer.output_width
        output_height = layer.folded_output_height or layer.output_height
        kernel_width = layer.folded_kernel_width or layer.kernel_width
        kernel_height = layer.folded_kernel_height or layer.kernel_height
        input_width = layer.folded_input_width or layer.input_width
        input_height = layer.folded_input_height or layer.input_height
        if return_all:
            return [
                layer.input_batch, input_width, input_height,
                layer.kernel_batch, kernel_width, kernel_height,
                layer.output_batch, output_width, output_height,
                layer.no_of_tiles
            ]
        else:
            return output_width, output_height, \
                    kernel_width, kernel_height, \
                    input_width, input_height
