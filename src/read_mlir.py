"""Python file to read MLIR files"""

import re
from conv2d_class import Conv2D
from depthwise_conv2d_class import DepthConv2D
from fully_connected_class import FullyConnected

# Regex expression to identify neural network layers in MLIR file
regex_patterns = {
    "conv2d": r'linalg\.conv_2d_nhwc_hwcf\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "depthwise_conv2d_multiplier": r'linalg\.depthwise_conv_2d_nhwc_hwcm\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "depthwise_conv2d": r'linalg\.depthwise_conv_2d_nhwc_hwc\s*\{\s*dilations\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*,\s*strides\s*=\s*dense<(\d+)>\s*:\s*tensor<\d+xi64>\s*[^}]*\}\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)',
    "matmul": r'linalg\.batch_matmul\s*ins\(\s*([^:]+):\s*memref<([^>]+)>\s*,\s*memref<([^>]+)>s*\)\s*outs\(\s*([^:]+):\s*memref<([^>]+)>\s*\)'
}

def process_tensor_shape(tensor_shape):
    """Function for reading dimensions string"""
    dimensions, _ = tensor_shape.strip().rsplit('x', 1)
    return list(map(int, dimensions.split('x')))

def read_file(args):
    """Function to read MLIR file"""
    layers = {}
    with open(args.read_mlir, 'r', encoding="UTF-8") as file:
        conv2d_count = 0
        depthwise_conv2d_count = 0
        matmul_count = 0
        for line in file:
            # Loop through each regex pattern in the dictionary
            for name, pattern in regex_patterns.items():
                match = re.findall(pattern, line)  # Find all matches for the current line
                if match and name == "conv2d" and args.conv2d:
                    # Increment conv2d count
                    conv2d_count += 1
                    print("====================")
                    print(f"Convolution layer - {conv2d_count}")
                    print("---------------------")
                    # If there are matches for conv2d and conv2d flag is set
                    dilations = int(match[0][0])
                    strides = int(match[0][1])
                    input_tensor_shape = process_tensor_shape(match[0][3])
                    kernel_tensor_shape = process_tensor_shape(match[0][4])
                    output_tensor_shape = process_tensor_shape(match[0][6])
                    # Create Conv2D layer
                    layer = Conv2D(args, input_tensor_shape,
                                   kernel_tensor_shape, output_tensor_shape, dilations, strides)
                    # Add layer to layers dictionary
                    layers[f"{name}_{conv2d_count}"] = layer
                elif match and (name == "depthwise_conv2d" or name == "depthwise_conv2d_multiplier") and args.depthwise_conv2d:
                    # Increment depthwise_conv2d count
                    depthwise_conv2d_count += 1
                    print("====================")
                    print(f"Depth-wise Convolution layer - {depthwise_conv2d_count}")
                    print("---------------------")
                    # If there are matches for depthwise_conv2d and depthwise_conv2d flag is set
                    dilations = int(match[0][0])
                    strides = int(match[0][1])
                    input_tensor_shape = process_tensor_shape(match[0][3])
                    kernel_tensor_shape = process_tensor_shape(match[0][4])
                    output_tensor_shape = process_tensor_shape(match[0][6])
                    # Create DepthwiseDepthConv2D layer
                    layer = DepthConv2D(args, input_tensor_shape,
                                        kernel_tensor_shape, output_tensor_shape,
                                        dilations, strides)

                    # Add layer to layers dictionary
                    layers[f"{name}_{depthwise_conv2d_count}"] = layer
                elif match and name == "matmul" and args.matmul:
                    # Increment matmul count
                    matmul_count += 1
                    print("====================")
                    print(f"FC layer - {matmul_count}")
                    print("---------------------")
                    # If there are matches for matmul and matmul flag is set
                    input_tensor_shape = process_tensor_shape(match[0][1])
                    kernel_tensor_shape = process_tensor_shape(match[0][2])
                    output_tensor_shape = process_tensor_shape(match[0][4])
                    # Create FullyConnected layer
                    layer = FullyConnected(args, input_tensor_shape,
                                           kernel_tensor_shape, output_tensor_shape)

                    # Add layer to layers dictionary
                    layers[f"{name}_{matmul_count}"] = layer

    print("\n\n=========================\n")
    print(f"Convolution layer count: {conv2d_count}")  # Print conv2d count
    print(f"Depthwise Convolution layer count: {depthwise_conv2d_count}")  # Print depthwise_conv2d count
    print(f"FC layer count: {matmul_count}")  # Print matmul count
    print("\n=========================\n")
    return layers  # Return layers dictionary
