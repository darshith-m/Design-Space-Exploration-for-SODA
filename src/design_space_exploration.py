"""Python file to perform Design Space Exploration on SODA framework"""

import os
import csv
import subprocess
from itertools import permutations
from utilities import get_layer_info, get_folded_layer_info
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DSE:
    """"Design Space Exploration class"""

    def __init__(self, args, layers):

        self.model_name, _ = os.path.splitext(os.path.basename(args.read_mlir)) # NN Architecture
        self.layers = layers    # Information of architecture's layers
        self.permute = args.permute # Perform loop permutation if True
        self.tile = args.tile   # Perform loop tiling if True
        self.unroll = args.unroll   # Perform loop permutation if True
        self.performance_oriented_exploration = 1 if args.performance else 0 # (1 = performence oriented; 0 = energy oriented)
        self.commands = None    # Current set of docker commands to execute
        self.current_configuration = None # Current configuration of loop optimization executed
        self.current_layer_name = None  # Current layer being explored

        if self.permute:
            self.permutations_list = [] # List of permutations to explore
            self.permutation_mapping = {} # List to correct permutation mapping
            self.current_permutation = None # Current Loop permutation

        if self.tile:
            self.tiling_combinations = []   # List of tiling combinations to explore
            self.current_tiling_combination = None  # Current Loop tiling configuration

        if self.unroll:
            self.unrolling_combinations = []    # List of unrolling combinations to explore
            self.current_unroll_combination = None  # Current Loop unrolling configuration

    def create_docker_commands(self):
        """Function to create docker commands"""
        # Basee command to initiate docker
        base_command = "docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda soda-opt "
        # SODA pipeline for Bambu
        soda_opt_bambu_pipeline = [
            "-affine-scalrep",
            "-cse",
            "-affine-data-copy-generate='generate-dma=false fast-mem-space=0'",
            "-erase-buffer-deallocation",
            "-promote-buffers-to-stack='max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096'",
            "-lower-affine",
            "-convert-scf-to-cf",
            "-convert-memref-to-llvm",
            "-convert-math-to-llvm",
            "-convert-math-to-libm",
            "-arith-expand",
            "-memref-expand",
            "-convert-arith-to-llvm",
            "-convert-func-to-llvm='use-bare-ptr-memref-call-conv'",
            "-reconcile-unrealized-casts",
            "--mlir-print-ir-after-all",
            f"output/04b{self.current_configuration}.mlir",
            f"-o output/04c{self.current_configuration}.mlir",
            f"2>&1 | cat > output/05cintermediate-{self.current_configuration}.mlir"
        ]
        # Generate different combinations of commands for loop optimizations
        if self.permute:
            # For Loop permutation, use a custom pass
            soda_opt_bambu_pipeline.insert(0,
                f"-test-loop-permutation='permutation-map={self.current_permutation}'")
        elif self.tile:
            # Generate the tiling combination by joining the contents of the list
            tiling_combination_string = ",".join(str(i) for i in self.current_tiling_combination)
            # Remove '-promote-buffers-to-stack' pass
            soda_opt_bambu_pipeline.pop(4)
            # Check if the current tiling combination is not baseline configuration
            if any(x != 0 for x in self.current_tiling_combination):
                # For Loop tiling, use a custom pass
                soda_opt_bambu_pipeline.insert(0,
                    f"-affine-loop-tile='tile-sizes={tiling_combination_string}'")
        elif self.unroll:
            # Remove 'affine-data-copy-generate' pass
            soda_opt_bambu_pipeline.pop(2)
            # Remove 'erase-buffer-deallocation' pass
            soda_opt_bambu_pipeline.pop(2)
            # String for unroll full command
            loop_unroll_full_string = "-affine-loop-unroll='unroll-full'"
            # String for partial unroll command
            loop_unroll_factor_string = f"-affine-loop-unroll='unroll-factor={self.current_unroll_combination[1]}'"
            # Check if the loop unrolling is ful-unroll
            if self.current_unroll_combination[1] == 0:
                # Repeat full-unrolls
                loop_unroll_string_list = [loop_unroll_full_string] * self.current_unroll_combination[0]
            else:
                # Command for partial-unrolls
                loop_unroll_string_list = [loop_unroll_full_string] * (self.current_unroll_combination[0] - 1) +  [loop_unroll_factor_string]
            # Join all the unroll commands to a string
            loop_unroll_string = " ".join(loop_unroll_string_list)
            # For loop unroll, use custom passes
            soda_opt_bambu_pipeline.insert(2, loop_unroll_string)
        # Concatenate all commands to one string
        soda_command = base_command + " ".join(soda_opt_bambu_pipeline)
        # Print the SODA docker command
        #print(f"SODA command: {soda_command}")
        # Dictionary of commands to execute from MLIR to ASIC
        self.commands = {
            "1a-soda": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            soda-opt \
            -soda-outline-bambu-code \
            -soda-extract-arguments-to-xml='using-bare-ptr' \
            -soda-generate-bambu-accelcode \
            -convert-linalg-to-affine-loops \
            --mlir-print-ir-after-all \
            {self.layers[self.current_layer_name].file_path} \
            -o output/04a{self.current_configuration}.mlir \
            2>&1 | cat > output/05aintermediate-{self.current_configuration}.mlir",

            "1b-soda":
            f"for file in *.xml; do mv \"$file\" \"output/${{file%.xml}}_{self.current_configuration}.xml\"; done",

            "1c-mlir": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            mlir-opt \
            -expand-strided-metadata \
            --mlir-print-ir-after-all \
            output/04a{self.current_configuration}.mlir \
            -o output/04b{self.current_configuration}.mlir \
            2>&1 | cat > output/05bintermediate-{self.current_configuration}.mlir",

            "1d-soda": soda_command,

            "1e-mlir-opt": 
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            mlir-opt \
            -symbol-dce \
            --mlir-print-ir-after-all \
            output/04c{self.current_configuration}.mlir \
            -o output/04d{self.current_configuration}.mlir \
            2>&1 | cat > output/05dintermediate-{self.current_configuration}.mlir",

            "1f-soda":
            f"docker run -u $(id -u) -v $(pwd):/working_dir --rm agostini01/soda \
            mlir-translate -opaque-pointers=0  \
            --mlir-to-llvmir \
            output/04d{self.current_configuration}.mlir \
            -o output/05{self.current_configuration}.ll",

            "2-bambu":
            f"scripts/run-bambu.sh {self.current_configuration} 2>&1 \
            | tee output/bambu-{self.current_configuration}.log",

            "3-openroad":
            f"scripts/run-openroad.sh {self.current_configuration} 2>&1 \
            | tee output/openroad-{self.current_configuration}.log"
        }

    def create_or_append_to_csv(self, file_path, headers, data):
        """Create a CSV file with headers if it doesn't exist, or append data to it if it does."""
        # Check if the file exists in file path
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write headers only if the file does not exist
            if not file_exists:
                writer.writerow(headers)
            # Append the data
            writer.writerow(data)

    def evaluate_asic(self, simulation_cycles, total_power, available_area, frequency):
        """Function to calculate ASIC's PPA, efficiency and energy consumed"""
        # Constant to calculate performance in GFLOPS
        giga_multiplier = 1e9
        # Set the current layer
        current_layer = self.layers[self.current_layer_name]
        # Evaluate Clock cycles, runtime, performance, efficiency and energy consumed
        actual_simulation_cycles = simulation_cycles * current_layer.no_of_tiles
        runtime_in_s = round(actual_simulation_cycles / frequency, 6)
        gflops = round(current_layer.flop_count / runtime_in_s / giga_multiplier, 6)
        gflops_per_watt = round(gflops / total_power, 6)
        energy_consumed = round(total_power * runtime_in_s, 12)
        # Store results in a dictionary
        results = {
                "Simulation Cycles": actual_simulation_cycles,
                "Total Power (W)": total_power,
                "Available Area (um²)": available_area,
                "Runtime (s)": runtime_in_s,
                "GFLOPS": gflops,
                "GFLOPS/Watt": gflops_per_watt,
                "Energy Consumed (J)": energy_consumed,
                "FLOP Count": current_layer.flop_count
        }
        # Print the results
        print("Simulation Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
        return results

    def record_results(self, simulation_cycles, total_power, available_area, frequency=100e6):
        """Function to record results"""
        # Set current layer
        current_layer_name = self.current_layer_name
        current_layer = self.layers[self.current_layer_name]
        # Set results directory
        results_directory = "./results"
        # Check if the results directory exits, else create it
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        # Get actual layer info
        layer_info = get_layer_info(current_layer, current_layer_name, self.current_configuration)
        # Get folded layer info
        folded_layer_info = get_folded_layer_info(current_layer,
                                                  current_layer_name, return_all=True)
        # Get PPA, efficiency and energy consumed metrics of ASIC
        results = self.evaluate_asic(simulation_cycles, total_power, available_area, frequency)
        # Set file path, row header and row to None initially
        file_path, row_header, row = None, None, None
        # Check if the layer evaluated is convolution layer
        if self.current_layer_name.startswith("conv2d"):
            # Set CSV row headers
            layer_info_header = ["configuration", "strides", "dilations",
                                 "input_batch", "input_width", "input_height", "input_channel", 
                                 "kernel_width", "kernel_height", "kernel_input_channels", 
                                 "kernel_output_channels", 
                                 "output_batch", "output_width", "output_height", "output_channel"]
            folded_layer_info_header = ["actual_input_batch",
                                "actual_input_width", "actual_input_height", 
                                "actual_input_channel", 
                                "actual_kernel_width", "actual_kernel_height", 
                                "actual_kernel_input_channels", "actual_kernel_output_channels", 
                                "actual_output_batch", 
                                "actual_output_width", "actual_output_height", 
                                "actual_output_channel", "number_of_tiles"]
            results_header = ["simulation_cycles", "total_power", "area",
                              "runtime_in_s", "gflops", "gflops_per_watt", 
                              "energy_consumed", "flop_count"]
            # Check if the current optimization is permutation
            if self.permute:
                file_path = f"./results/{self.model_name}_conv2d_permute.csv"
                # Convert the current permutation string to a list of integers
                permutation_order = list(map(int, self.current_permutation.split(',')))
                # Create headers for the permutation order
                permutation_order_header = \
                    [f"permuation_order_{i}" for i in range(1, len(permutation_order) + 1)]
                # Combine all headers
                row_header = layer_info_header + permutation_order_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + permutation_order + \
                    folded_layer_info + list(results.values())
            # Check if the current optimization is tiling
            elif self.tile:
                file_path = f"./results/{self.model_name}_conv2d_tile.csv"
                # Get the current tiling combination
                tiles = self.current_tiling_combination
                # Create headers for the tiling combination
                tiles_header = ["tiled_output_batch",
                        "tiled_output_width", "tiled_output_height", "tiled_output_channel", 
                        "tiled_kernel_width", "tiled_kernel_height", "tiled_input_channel"]
                # Combine all headers
                row_header = layer_info_header + tiles_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + tiles + folded_layer_info + list(results.values())
            # Check if the current optimization is unrolling
            elif self.unroll:
                file_path = f"./results/{self.model_name}_conv2d_unroll.csv"
                # Get the current unroll combination
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                # Create headers for the unroll combination
                unroll_header = ["unroll_full", "unroll_factor"]
                # Combine all headers
                row_header = layer_info_header + unroll_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + unrolls + folded_layer_info + list(results.values())
        # Check if the layer evaluated is depth-wise convolution layer
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            # Set CSV row headers
            layer_info_header = ["configuration", "strides", "dilation",
                                 "input_batch",
                                 "input_width", "input_height", "input_channel",
                                 "kernel_width", "kernel_height",
                                 "kernel_input_channel", "kernel_multiplier",
                                 "output_batch",
                                 "output_width", "output_height", "output_channel",
                                 "output_multiplier"]
            folded_layer_info_header = ["actual_input_batch",
                                "actual_input_width", "actual_input_height", "actual_input_channel",
                                "actual_kernel_width", "actual_kernel_height",
                                "actual_kernel_input_channels",
                                "actual_output_batch",
                                "actual_output_width", "actual_output_height",
                                "actual_output_channel",
                                "number_of_tiles"]
            results_header = ["simulation_cycles", "total_power", "area",
                        "runtime_in_s", "gflops", "gflops_per_watt",
                        "energy_consumed", "flop_count"]
            # Check if the current optimization is permutation
            if self.permute:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_permute.csv"
                # Convert the current permutation string to a list of integers
                permutation_order = list(map(int, self.current_permutation.split(',')))
                # Create headers for the permutation order
                permutation_order_header = \
                    [f"permuation_order_{i}" for i in range(1, len(permutation_order) + 1)]
                # Combine all headers
                row_header = layer_info_header + permutation_order_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + permutation_order + \
                    folded_layer_info + list(results.values())
            # Check if the current optimization is tiling
            elif self.tile:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_tile.csv"
                # Get the current tiling combination
                tiles = self.current_tiling_combination
                # Create headers for the tiling combination
                tiles_header = ["tiled_output_batch",
                                "tiled_output_width", "tiled_output_height",
                                "tiled_input_channel",
                                "tiled_kernel_width", "tiled_kernel_height"]
                # Combine all headers
                row_header = layer_info_header + tiles_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + tiles + folded_layer_info + list(results.values())
            # Check if the current optimization is unrolling
            elif self.unroll:
                file_path = f"./results/{self.model_name}_depthwise_conv2d_unroll.csv"
                # Get the current unroll combination
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                # Create headers for the unroll combination
                unroll_header = ["unroll_full", "unroll_factor"]
                # Combine all headers
                row_header = layer_info_header + unroll_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + unrolls + folded_layer_info + list(results.values())
        # Check if the layer evaluated is fully connected layer
        elif self.current_layer_name.startswith("matmul"):
            # Set CSV row headers
            layer_info_header = ["configuration",
                                 "input_batch", "input_width", "input_height",
                                 "weight_batch", "weight_width", "weight_height",
                                 "output_batch", "output_width", "output_height"]
            folded_layer_info_header = ["actual_input_batch",
                                "actual_input_width", "actual_input_height",
                                "actual_weight_batch",
                                "actual_weight_width", "actual_weight_height",
                                "actual_output_batch",
                                "actual_output_width", "actual_output_height",
                                "number_of_tiles"]
            results_header = ["simulation_cycles", "total_power", "area",
                              "runtime_in_s", "gflops", "gflops_per_watt",
                              "energy_consumed", "flop_count"]
            # Check if the current optimization is permutation
            if self.permute:
                file_path = f"./results/{self.model_name}_matmul_permute.csv"
                # Convert the current permutation string to a list of integers
                permutation_order = list(map(int, self.current_permutation.split(',')))
                # Create headers for the permutation order
                permutation_order_header = \
                    [f"permutation_order_{i}" for i in range(1, len(permutation_order) + 1)]
                # Combine all headers
                row_header = layer_info_header + permutation_order_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + permutation_order + \
                    folded_layer_info + list(results.values())
            # Check if the current optimization is tiling
            elif self.tile:
                file_path = f"./results/{self.model_name}_matmul_tile.csv"
                # Get the current tiling combination
                tiles = self.current_tiling_combination
                # Create headers for the tiling combination
                tiles_header = ["tiled_output_batch",
                                "tiled_output_width", "tiled_output_height",
                                "tiled_kernel_width"]
                # Combine all headers
                row_header = layer_info_header + tiles_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + tiles + folded_layer_info + list(results.values())
            # Check if the current optimization is unrolling
            elif self.unroll:
                file_path = f"./results/{self.model_name}_matmul_unroll.csv"
                # Get the current unroll combination
                unroll_full, unrolling_factor = self.current_unroll_combination
                unrolls = [unroll_full, unrolling_factor]
                # Create headers for the unroll combination
                unroll_header = ["unroll_full", "unroll_factor"]
                # Combine all headers
                row_header = layer_info_header + unroll_header + \
                    folded_layer_info_header + results_header
                # Combine all row data
                row = layer_info + unrolls + folded_layer_info + list(results.values())
        # Call the function to add row to CSV File
        self.create_or_append_to_csv(file_path, row_header, row)

    def execute_commands(self):
        """Function to execute docker commands"""
        # Define the paths
        txt_file_path = f"output/progress-{self.current_configuration}.txt"
        # Get the directory name
        directory = os.path.dirname(txt_file_path)
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Open the output text file in write mode
        with open(txt_file_path, "w", encoding="utf-8") as output_file:
            # Initialize values for CSV
            simulation_cycles = None
            total_power = None
            available_area = None
            utilization_area = None
            for key, command in self.commands.items():
                # Execute the command and redirect stdout and stderr to the output file
                subprocess.run(command, shell=True, stdout=output_file,
                               stderr=output_file, check=True)
                # Check specific conditions after certain commands
                if key == "2-bambu":
                    # Initialize cycles variable
                    cycles = ""
                    # Read the Bambu log file to extract the average execution cycles
                    for runtime in open(f'output/{self.current_configuration}/bambu-log',
                                            encoding='utf-8').readlines():
                        if "Average execution" in runtime:
                            # Extract the first integer from the line
                            cycles = [int(s) for s in runtime.split() if s.isdigit()][0]
                    # Write the average execution cycles to the output file
                    output_file.write(f"Average execution in cycles: {cycles}\n")
                    # Store the simulation cycles for later use
                    simulation_cycles = int(cycles)
                elif key == "3-openroad":
                    # Define the path to the OpenROAD log file
                    log_path_suffix = 'HLS_output/Synthesis/bash_flow/openroad/logs/nangate45/main_kernel/base/6_report.log'
                    log_file = f'output/{self.current_configuration}/' + log_path_suffix
                    # Initialize power multiplier
                    power_multiplier = 1
                    # Read the OpenROAD log file to extract power and area information
                    for l in open(log_file, 'r', encoding='utf-8').readlines():
                        if ("Total" in l and "Group" not in l):
                            # Extract total power consumption
                            total_power = float(l.split()[4]) * power_multiplier
                        if ("Design area" in l):
                            available_area = float(l.split()[2])
                            utilization_area = float(l.split()[4].strip('%'))
                            # Extract available area and utilization area
                            available_area = float(l.split()[2])
                            utilization_area = float(l.split()[4].strip('%'))
                    # Write the extracted information to the output file
                    output_file.write('Optimized accelerator:\n')
                    output_file.write(f'  total power consumption: {total_power} W\n')
                    output_file.write(f'  available chip area: {available_area} um^2\n')
                    output_file.write(f'  utilized chip area: {utilization_area} %\n')
                    # Record the results in a CSV file
                    self.record_results(simulation_cycles, total_power, available_area)
                    # Path to the output folder
                    output_folder = './output'
                    # Command to delete files and folders of current configuration
                    command = f"find {output_folder} -name \
                        '*{self.current_configuration}*' -exec rm -rf {{}} +"
                    # Execute the command to clean up the output folder
                    subprocess.run(command, shell=True, check=True)

    def docker_commands(self):
        """Function to set and execute the docker commands for current 
        loop optimization configuration"""
        if self.permute:
            # Iterate over each permutation in the permutations list
            for permutation in self.permutations_list:
                # Set the current configuration string
                self.current_configuration = f"{self.model_name}_permute_{self.current_layer_name}_{''.join(map(str, permutation))}"
                print("--------------------------------")
                print(f"Configuration: {self.current_configuration}")
                # Convert permutation list to a comma-separated string for Docker command
                docker_perm_string = ','.join(map(str, permutation))
                # Get actual permutation string from mapping or use Docker string
                actual_perm_string = \
                    self.permutation_mapping.get(docker_perm_string, docker_perm_string)
                # Set the current permutation
                self.current_permutation = actual_perm_string
                print(f"Current permutation: {self.current_permutation}")
                            # Create Docker commands for the current configuration
                self.create_docker_commands()
                # Execute the Docker commands
                self.execute_commands()
                # Reset the current permutation
                self.current_permutation = None
        elif self.tile:
            # Iterate over each tiling combination
            for _, tile in enumerate(self.tiling_combinations):
                # Set the current configuration string
                self.current_configuration = \
                    f"{self.model_name}_tile_{self.current_layer_name}_{''.join(map(str, tile))}"
                print("--------------------------------")
                print(f"Configuration: {self.current_configuration}")
                # Set the current tiling combination
                self.current_tiling_combination = tile
                print(f"Current tiling combination: {self.current_tiling_combination}")
                # Create Docker commands for the current configuration
                self.create_docker_commands()
                # Execute the Docker commands
                self.execute_commands()
                # Reset the current tiling combination
                self.current_tiling_combination = None
        elif self.unroll:
            # Iterate over each unrolling combination
            for _, unroll in enumerate(self.unrolling_combinations):
                # Set the current configuration string
                self.current_configuration = \
                    f"{self.model_name}_unroll_{self.current_layer_name}_unroll_{unroll[0]}_factor_{unroll[1]}"
                print("--------------------------------")
                print(f"Configuration: {self.current_configuration}")
                # Set the current unroll combination
                self.current_unroll_combination = unroll
                print(f"Current unroll combination: {self.current_unroll_combination}")
                # Create Docker commands for the current configuration
                self.create_docker_commands()
                # Execute the Docker commands
                self.execute_commands()
                # Set current unrolling combination to None
                self.current_unroll_combination = None   

    def get_permutations(self):
        """Function to generate loop permutations"""
        # Initialize the mapping CSV path to None
        mapping_csv_path = None
        # Check if the current layer is a convolution layer
        if self.current_layer_name.startswith("conv2d"):
            # Set the mapping CSV path for convolution layers
            mapping_csv_path = './scripts/conv2d_mapping.csv'
            # Define groups for permutations
            group1 = [1, 2, 3]
            group2 = [4, 6, 5]
            # Generate permutations for output-first configuration
            for perm1 in permutations(group1):
                for perm2 in permutations(group2):
                    permutation = [0] + list(perm1) + list(perm2)
                    self.permutations_list.append(permutation)
            # Generate permutations for kernel-first configuration
            for perm2 in permutations(group2):
                for perm1 in permutations(group1):
                    permutation = [0] + list(perm2) + list(perm1)
                    self.permutations_list.append(permutation)
        # Check if the current layer is a depthwise convolution layer
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            # Set the mapping CSV path for depthwise convolution layers
            mapping_csv_path = './scripts/depthwise_conv2d_mapping.csv'
            # Generate permutations for output-first configuration
            group1 = [1, 2, 3]
            group2 = [4, 5]
            for perm1 in permutations(group1):
                for perm2 in permutations(group2):
                    permutation = [0] + list(perm1) + list(perm2)
                    self.permutations_list.append(permutation)
            # Generate permutations for kernel-first configuration
            group1 = [3, 4, 5]
            group2 = [1, 2]
            for perm1 in permutations(group1):
                for perm2 in permutations(group2):
                    permutation = [0] + list(perm1) + list(perm2)
                    self.permutations_list.append(permutation)
        # Check if the current layer is a fully connected layer
        elif self.current_layer_name.startswith("matmul"):
            # Set the mapping CSV path for fully connected layers
            mapping_csv_path = './scripts/matmul_mapping.csv'
            # Define group for permutations
            group1 = [1, 2, 3]
            # Generate permutations for group1 and append to permutations list
            for perm in permutations(group1):
                permutation = [0] + list(perm)  # Add 0 at the beginning
                self.permutations_list.append(permutation)
        # Return the mapping CSV path
        return mapping_csv_path

    def perform_permutation(self):
        """Function to perform loop permutation optimization"""
        # Get the path to the mapping CSV file based on the current layer type
        mapping_csv_path = self.get_permutations()
        # Read the mapping CSV file into a dictionary
        with open(mapping_csv_path, mode='r', encoding='UTF-8') as mapping_file:
            reader = csv.reader(mapping_file)
            for row in reader:
                docker_perm = row[0]
                actual_perm = row[1]
                if actual_perm:  # Only add if the actual permutation is not empty
                    self.permutation_mapping[docker_perm] = actual_perm
        # Execute the Docker commands for the current permutation configuration
        self.docker_commands()

    def generate_tiling_combinations(self, tiling_dimensions_1,
                                    tiling_dimensions_2, tiling_dimensions_3):
        """Function to generate loop tiling configurations"""
        current_layer = self.layers[self.current_layer_name]
        if self.current_layer_name.startswith("conv2d"):
            # Set tiling dimensions for convolution layers
            output_tiles = tiling_dimensions_1
            kernel_tiles = tiling_dimensions_2
            input_channel_tiles = tiling_dimensions_3
            # Initialize tiling combinations with a baseline configuration
            self.tiling_combinations.append([0, 0, 0, 0, 0, 0, 0])
            # Generate tiling combinations
            for output in output_tiles:
                for kernel in kernel_tiles:
                    if output >= kernel:
                        for input_channel in input_channel_tiles:
                            tiling_combination_string = [
                                current_layer.output_batch,
                                output, output,
                                current_layer.folded_output_channel,
                                kernel, kernel, input_channel
                            ]
                            self.tiling_combinations.append(tiling_combination_string)
            # Remove the baseline configuration
            self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            # Set tiling dimensions for depthwise convolution layers
            output_tiles = tiling_dimensions_1
            kernel_tiles = tiling_dimensions_2
            input_channel_tiles = tiling_dimensions_3
            # Initialize tiling combinations with a baseline configuration
            self.tiling_combinations.append([0, 0, 0, 0, 0, 0])
            # Generate tiling combinations
            for output in output_tiles:
                for kernel in kernel_tiles:
                    if output >= kernel:
                        for input_channel in input_channel_tiles:
                            tiling_combination = [
                                current_layer.output_batch,
                                output, output,
                                input_channel,
                                kernel, kernel
                            ]
                            self.tiling_combinations.append(tiling_combination)
            # Remove the baseline configuration
            self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")
        elif self.current_layer_name.startswith("matmul"):
            # Set tiling dimensions for fully connected layers
            output_width_tiles = tiling_dimensions_1
            output_height_tiles = tiling_dimensions_2
            kernel_width_tiles = tiling_dimensions_3
            # Initialize tiling combinations with a baseline configuration
            self.tiling_combinations.append([0, 0, 0, 0])
            # Generate tiling combinations
            for output_width in output_width_tiles:
                for output_height in output_height_tiles:
                    for kernel_width in kernel_width_tiles:
                        tiling_combination = [
                            current_layer.output_batch,
                            output_width, output_height, kernel_width
                        ]
                        self.tiling_combinations.append(tiling_combination)
            # Remove the baseline configuration
            self.tiling_combinations.pop(-1)
            print(f"Tiling combinations: {self.tiling_combinations}")

    def get_tile_sizes(self, max_tile_size, is_kernel = False, min_power = 2):
        """Function to generate tile sizes"""
        # Check if tile size is for kernel
        if is_kernel:
            # Initialize an empty list for kernel tile sizes
            kernel_tile_sizes = []
            # Generate odd tile sizes from 3 to max_tile_size (inclusive)
            kernel_tile_sizes = list(range(3, max_tile_size + 1, 2))
            # Ensure max_tile_size is included in the list
            if max_tile_size not in kernel_tile_sizes:
                kernel_tile_sizes.append(max_tile_size)
            return kernel_tile_sizes
        # Tile sizes for output feature map
        else:
            # Generate powers of two from 2^min_power to 2^log2(max_tile_size) (inclusive)
            power_of_two = [2**i for i in range(min_power, int(np.log2(max_tile_size))+1)]
            # Initialize an empty list for output tile sizes
            output_tile_sizes = []
            # Filter power_of_two to include only values less than max_tile_size
            output_tile_sizes = [i for i in power_of_two if i < max_tile_size]
            # Ensure max_tile_size is included in the list
            if max_tile_size not in output_tile_sizes:
                output_tile_sizes.append(max_tile_size)
            return output_tile_sizes

    def perform_tiling(self):
        """Function to perform loop tiling configurations"""
        # Get the current layer being processed
        current_layer = self.layers[self.current_layer_name]
        current_layer_name = self.current_layer_name
        # Check if the current layer is a convolution layer
        if self.current_layer_name.startswith("conv2d"):
            # Get tile sizes for output height
            output_tiles = self.get_tile_sizes(current_layer.output_height)
            # Get tile sizes for kernel height, specifying it's a kernel
            kernel_tiles = self.get_tile_sizes(current_layer.kernel_height, is_kernel=True)
            # Get tile sizes for input channel, considering folded input channel if available
            input_channel = current_layer.folded_input_channel or current_layer.input_channel
            input_channel_tiles = self.get_tile_sizes(input_channel)
                # Generate tiling combinations for convolution layer
            self.generate_tiling_combinations(output_tiles, kernel_tiles, input_channel_tiles)
        # Check if the current layer is a depthwise convolution layer
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            # Get tile sizes for output height
            output_tiles = self.get_tile_sizes(current_layer.output_height)
            # Get tile sizes for kernel height, specifying it's a kernel
            kernel_tiles = self.get_tile_sizes(current_layer.kernel_height, is_kernel=True)
            # Get tile sizes for input channel, considering folded input channel if available
            input_channel = current_layer.folded_input_channel or current_layer.input_channel
            input_channel_tiles = self.get_tile_sizes(input_channel)
                # Generate tiling combinations for depthwise convolution layer
            self.generate_tiling_combinations(output_tiles, kernel_tiles, input_channel_tiles)
        # Check if the current layer is a fully connected layer
        elif self.current_layer_name.startswith("matmul"):
            output_width, output_height, kernel_width, _, _, _ = \
                get_folded_layer_info(current_layer, current_layer_name, return_all=False)
            # Get tile sizes for output width, output height, and kernel width
            output_width_tiles = self.get_tile_sizes(output_width)
            output_height_tiles = self.get_tile_sizes(output_height)
            kernel_width_tiles = self.get_tile_sizes(kernel_width)
            # Generate tiling combinations for fully connected layer
            self.generate_tiling_combinations(output_width_tiles,
                                              output_height_tiles, kernel_width_tiles)
        # Execute Docker commands for the current tiling configuration
        self.docker_commands()

    def generate_unrolling_combinations(self, unrolls_1, unrolls_2, unrolls_3 = None):
        """Function to generate loop unrolling combinations"""
        # Append baseline configuration
        self.unrolling_combinations.append((0,0))
        # For partial unroll-1
        for i in unrolls_1:
            self.unrolling_combinations.append((1,i))
        # Full-unroll once
        self.unrolling_combinations.append((1,0))
        # For partial unroll-2
        for i in unrolls_2:
            self.unrolling_combinations.append((2,i))
        # Full-unroll twice
        self.unrolling_combinations.append((2,0))
        # Fully connected layer does not require 3 unrolls
        if unrolls_3 is not None:
            # For partial unroll-3
            for i in unrolls_3:
                self.unrolling_combinations.append((3,i))
            # Full-unroll thrice
            self.unrolling_combinations.append((3,0))
        print(f"Unrolling combinations: {self.unrolling_combinations}")

    def get_unroll_sizes(self, max_unroll_size, min_power = 2):
        """Function to generate partial unroll sizes"""
        powers_of_two = [2**i for i in range(min_power, int(np.log2(max_unroll_size)) + 1)]
        return [p for p in powers_of_two if p < max_unroll_size]

    def perform_unrolling(self):
        """Function to perform loop unrolling optimization"""
        # Get the current layer being processed
        current_layer = self.layers[self.current_layer_name]
        current_layer_name = self.current_layer_name
        # Check if the current layer is a convolution layer
        if self.current_layer_name.startswith("conv2d"):
            # Get unroll sizes for kernel height
            kernel_unrolls = self.get_unroll_sizes(current_layer.kernel_height)
            # Get unroll sizes for input channel, considering folded input channel if available
            input_channel = current_layer.folded_input_channel or current_layer.input_channel
            input_channel_unrolls = self.get_unroll_sizes(input_channel)
            # Generate unrolling combinations for convolution layer
            self.generate_unrolling_combinations(input_channel_unrolls,
                                                 kernel_unrolls, kernel_unrolls)
        # Check if the current layer is a depthwise convolution layer
        elif self.current_layer_name.startswith("depthwise_conv2d"):
            # Get unroll sizes for kernel height
            kernel_unrolls = self.get_unroll_sizes(current_layer.kernel_height)
            # Get unroll sizes for input channel, considering folded input channel if available
            input_channel = current_layer.folded_input_channel or current_layer.input_channel
            input_channel_unrolls = self.get_unroll_sizes(input_channel)
            # Generate unrolling combinations for depthwise convolution layer
            self.generate_unrolling_combinations(kernel_unrolls,
                                                 kernel_unrolls, input_channel_unrolls)
        # Check if the current layer is a fully connected layer
        elif self.current_layer_name.startswith("matmul"):
            _, output_height, kernel_width, _, _, _ = \
                get_folded_layer_info(current_layer, current_layer_name, return_all=False)
            # Get unroll sizes for output height and kernel width
            output_height_unrolls = self.get_unroll_sizes(output_height)
            kernel_width_unrolls = self.get_unroll_sizes(kernel_width)
            # Generate unrolling combinations for fully connected layer
            self.generate_unrolling_combinations(kernel_width_unrolls,
                                                 output_height_unrolls)
        # Execute Docker commands for the current unrolling configuration
        self.docker_commands()

    def execute_dse(self):
        """Function to perform Design Space Exploration"""
        for layer_name in self.layers.keys():
            print("===========================")
            print(f"Layer name: {layer_name}")
            # Set the current layer name
            self.current_layer_name = layer_name
            # Perform permutation optimization if required
            if self.permute:
                self.perform_permutation()
            # Perform tiling optimization if required
            if self.tile:
                self.perform_tiling()
            # Perform unrolling optimization if required
            if self.unroll:
                self.perform_unrolling()
    
    def perform_opt_permutation(self):
        """Function to perform loop permutation optimization using decision trees"""
        # Import necessary modules
        import pickle
        import os
        
        print("Performing decision tree-based permutation optimization...")
        
        # Load the permutation optimizer model based on layer type
        model_dir = 'models'
        
        # Determine model paths based on layer type
        if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
            # Use conv2d models from ./decision_tree_models directory
            permutation_model_path = os.path.join('./decision_tree_models', "conv2d_permutation_model.pkl")
            permutation_scaler_path = os.path.join('./decision_tree_models', "conv2d_permutation_scaler.pkl")
            perm_encoder_path = os.path.join('./decision_tree_models', "conv2d_perm_encoder.pkl")
        elif self.current_layer_name.startswith("matmul"):
            # Use fc (fully connected) models from models directory
            permutation_model_path = os.path.join('./decision_tree_models', "fc_permutation_model.pkl")
            permutation_scaler_path = os.path.join('./decision_tree_models', "fc_permutation_scaler.pkl")
            perm_encoder_path = os.path.join('./decision_tree_models', "fc_perm_encoder.pkl")
        else:
            print(f"Unsupported layer type for optimization: {self.current_layer_name}")
            return
        
        try:
            # Load model, scaler and encoder
            with open(permutation_model_path, 'rb') as f:
                permutation_model = pickle.load(f)
            
            with open(permutation_scaler_path, 'rb') as f:
                permutation_scaler = pickle.load(f)
                
            with open(perm_encoder_path, 'rb') as f:
                perm_encoder = pickle.load(f)
                
            # Get the current layer being processed
            current_layer = self.layers[self.current_layer_name]
            
            # Extract features for prediction (handle both conv2d and matmul layers)
            if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
                features = {
                    'I': current_layer.input_height,
                    'K': current_layer.kernel_height,
                    'IC': current_layer.input_channel,
                    'O': current_layer.output_channel,
                    'S': current_layer.strides,
                    'D': current_layer.dilations,
                    'P': self.performance_oriented_explore,
                    'DW': 1 if self.current_layer_name.startswith("depthwise_conv2d") else 0
                }
                
                # Prepare input for prediction
                X = [[features['I'], features['K'], features['IC'], features['O'], 
                    features['S'], features['D'], features['P'], features['DW']]]
            elif self.current_layer_name.startswith("matmul"):
                features = {
                    'B': current_layer.input_batch,
                    'M': current_layer.input_height,
                    'K': current_layer.input_width,
                    'N': current_layer.weight_width,
                    'P': self.performance_oriented_explore,
                }
                
                # Prepare input for prediction (adjust as needed based on fc model features)
                X = [[features['B'], features['M'], features['K'], features['N']]]
            
            # Scale the features
            X_scaled = permutation_scaler.transform(X)
            
            # Make prediction
            perm_class = permutation_model.predict(X_scaled)[0]
            perm_sequence = perm_encoder.inverse_transform([perm_class])[0]
            
            # Handle potential different formats for permutation sequences
            if isinstance(perm_sequence, str) and '_' in perm_sequence:
                perm_list = [int(x) for x in perm_sequence.split('_')]
            elif isinstance(perm_sequence, str):
                perm_list = [int(x) for x in perm_sequence.split(',')]
            else:
                perm_list = [int(x) for x in perm_sequence]
            
            print(f"Predicted optimal permutation: {perm_list}")
            
            # Set the current permutation for execution
            self.current_permutation = ','.join(map(str, perm_list))
            
            # Set the configuration name
            self.current_configuration = f"{self.model_name}_dt_permute_{self.current_layer_name}_{''.join(map(str, perm_list))}"
            
            print(f"Configuration: {self.current_configuration}")
            print(f"Current permutation: {self.current_permutation}")
            
            # Create and execute Docker commands
            self.create_docker_commands()
            self.execute_commands()
            
        except FileNotFoundError as e:
            print(f"Error: Required model files not found - {e}")
            print("Make sure the decision tree models have been trained and saved.")
        except Exception as e:
            print(f"Error during permutation optimization: {e}")

    def perform_opt_tiling(self):
        """Function to perform loop tiling optimization using decision trees"""
        # Import necessary modules
        import pickle
        import os
        
        print("Performing decision tree-based tiling optimization...")
        
        # Determine model paths based on layer type
        if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
            # Use conv2d models from ./decision_tree_models directory
            tiling_ot_model_path = os.path.join('./decision_tree_models', "conv2d_tiling_O_T_model.pkl")
            tiling_ict_model_path = os.path.join('./decision_tree_models', "conv2d_tiling_IC_T_model.pkl")
            tiling_scaler_path = os.path.join('./decision_tree_models', "conv2d_tiling_scaler.pkl")
        elif self.current_layer_name.startswith("matmul"):
            # Use fc models from models directory
            tiling_k_model_path = os.path.join('./decision_tree_models', "fc_tile_k_model.pkl")
            tiling_n_model_path = os.path.join('./decision_tree_models', "fc_tile_n_model.pkl")
            tiling_scaler_path = os.path.join('./decision_tree_models', "fc_tile_scaler.pkl")
        else:
            print(f"Unsupported layer type for optimization: {self.current_layer_name}")
            return
        
        try:
            # Get the current layer being processed
            current_layer = self.layers[self.current_layer_name]
            
            if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
                # Load models and scaler for conv2d
                with open(tiling_ot_model_path, 'rb') as f:
                    tiling_ot_model = pickle.load(f)
                
                with open(tiling_ict_model_path, 'rb') as f:
                    tiling_ict_model = pickle.load(f)
                    
                with open(tiling_scaler_path, 'rb') as f:
                    tiling_scaler = pickle.load(f)
                
                # Extract features for prediction
                features = {
                    'I': current_layer.input_height,
                    'K': current_layer.kernel_height,
                    'IC': current_layer.input_channel,
                    'O': current_layer.output_channel,
                    'S': current_layer.strides,
                    'D': current_layer.dilations,
                    'P': self.performance_oriented_explore,
                    'DW': 1 if self.current_layer_name.startswith("depthwise_conv2d") else 0
                }
                
                # Prepare input for prediction
                X = [[features['I'], features['K'], features['IC'], features['O'], 
                    features['S'], features['D'], features['P'], features['DW']]]
                
                # Scale the features
                X_scaled = tiling_scaler.transform(X)
                
                # Make predictions
                o_t_value = tiling_ot_model.predict(X_scaled)[0]
                ic_t_value = tiling_ict_model.predict(X_scaled)[0]
                
                # Convert numpy integers to Python integers
                o_t_value = int(o_t_value)
                ic_t_value = int(ic_t_value)

                if o_t_value == 0:
                    o_t_value = current_layer.output_height
                if ic_t_value == 0:
                    ic_t_value = current_layer.input_channel
                
                print(f"Predicted optimal O_T value: {o_t_value}")
                print(f"Predicted optimal IC_T value: {ic_t_value}")
                
                # Build tiling combination
                # [output_batch, output_width, output_height, output_channel, kernel_width, kernel_height, input_channel]
                tiling_combination = [
                    int(current_layer.output_batch),
                    min(o_t_value, int(current_layer.output_width)), 
                    min(o_t_value, int(current_layer.output_height)),
                    int(current_layer.folded_output_channel),
                    int(current_layer.kernel_height),
                    int(current_layer.kernel_width),
                    min(ic_t_value, int(current_layer.input_channel))
                ]
                
            elif self.current_layer_name.startswith("matmul"):
                # Load models and scaler for matmul
                with open(tiling_k_model_path, 'rb') as f:
                    tiling_k_model = pickle.load(f)
                
                with open(tiling_n_model_path, 'rb') as f:
                    tiling_n_model = pickle.load(f)
                    
                with open(tiling_scaler_path, 'rb') as f:
                    tiling_scaler = pickle.load(f)
                
                # Extract features for prediction
                features = {
                    'B': current_layer.input_batch,
                    'M': current_layer.input_height,
                    'K': current_layer.input_width,
                    'N': current_layer.weight_width,
                    'P': self.performance_oriented_explore
                }
                
                # Prepare input for prediction
                X = [[features['B'], features['M'], features['K'], features['N']]]
                
                # Scale the features
                X_scaled = tiling_scaler.transform(X)
                
                # Make predictions and convert to Python integers
                k_value = int(tiling_k_model.predict(X_scaled)[0])
                n_value = int(tiling_n_model.predict(X_scaled)[0])
                
                print(f"Predicted optimal K tile value: {k_value}")
                print(f"Predicted optimal N tile value: {n_value}")
                
                # Build tiling combination for matmul
                # [output_batch, output_width, output_height, kernel_width]
                tiling_combination = [
                    int(current_layer.output_batch),
                    n_value, 
                    int(current_layer.output_height),
                    k_value
                ]
            
            # Set the current tiling combination for execution
            self.current_tiling_combination = tiling_combination
            
            # Set the configuration name
            self.current_configuration = f"{self.model_name}_dt_tile_{self.current_layer_name}_{''.join(map(str, tiling_combination))}"
            
            print(f"Configuration: {self.current_configuration}")
            print(f"Current tiling combination: {self.current_tiling_combination}")
            
            # Create and execute Docker commands
            self.create_docker_commands()
            self.execute_commands()
        
        except FileNotFoundError as e:
            print(f"Error: Required model files not found - {e}")
            print("Make sure the decision tree models have been trained and saved.")
        except Exception as e:
            print(f"Error during tiling optimization: {e}")

    def perform_opt_unrolling(self):
        """Function to perform loop unrolling optimization using decision trees"""
        # Import necessary modules
        import pickle
        import os
        
        print("Performing decision tree-based unrolling optimization...")
        
        # Determine model paths based on layer type
        if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
            # Use conv2d models from ./decision_tree_models directory
            unrolling_u_model_path = os.path.join('./decision_tree_models', "conv2d_unrolling_U_model.pkl")
            unrolling_f_model_path = os.path.join('./decision_tree_models', "conv2d_unrolling_F_model.pkl")
            unrolling_scaler_path = os.path.join('./decision_tree_models', "conv2d_unrolling_scaler.pkl")
        elif self.current_layer_name.startswith("matmul"):
            # Use fc models from models directory
            unrolling_u_model_path = os.path.join('./decision_tree_models', "fc_unroll_model.pkl")
            unrolling_scaler_path = os.path.join('./decision_tree_models', "fc_unroll_scaler.pkl")
        else:
            print(f"Unsupported layer type for optimization: {self.current_layer_name}")
            return
        
        try:
            # Get the current layer being processed
            current_layer = self.layers[self.current_layer_name]
            
            if self.current_layer_name.startswith("conv2d") or self.current_layer_name.startswith("depthwise_conv2d"):
                # Load models and scaler for conv2d
                with open(unrolling_u_model_path, 'rb') as f:
                    unrolling_u_model = pickle.load(f)
                
                with open(unrolling_f_model_path, 'rb') as f:
                    unrolling_f_model = pickle.load(f)
                    
                with open(unrolling_scaler_path, 'rb') as f:
                    unrolling_scaler = pickle.load(f)

                # Extract features for prediction
                features = {
                    'I': current_layer.input_height,
                    'K': current_layer.kernel_height,
                    'IC_U': current_layer.folded_input_channel,  # Use folded or estimate
                    'O': current_layer.output_channel,
                    'S': current_layer.strides,
                    'D': current_layer.dilations,
                    'P': self.performance_oriented_explore,
                    'DW': 1 if self.current_layer_name.startswith("depthwise_conv2d") else 0
                }
                
                # Prepare input for prediction
                X = [[features['I'], features['K'], features['IC_U'], features['O'], 
                    features['S'], features['D'], features['P'], features['DW']]]
                
                # Scale the features
                X_scaled = unrolling_scaler.transform(X)
                
                # Make predictions
                u_value = unrolling_u_model.predict(X_scaled)[0]  # Number of full unrolls
                f_value = unrolling_f_model.predict(X_scaled)[0]  # Unroll factor (for partial unroll)

                # Make predictions
                u_prediction = unrolling_u_model.predict(X_scaled)
                u_value = u_prediction[0] if isinstance(u_prediction, (list, np.ndarray)) else u_prediction

                f_prediction = unrolling_f_model.predict(X_scaled)
                f_value = f_prediction[0] if isinstance(f_prediction, (list, np.ndarray)) else f_prediction

                print(f"Prediction type: {type(u_value)}")
                print(f"Prediction type: {type(f_value)}")
                print(f"Predicted optimal unroll count (U): {u_value}")
                print(f"Predicted optimal unroll factor (F): {f_value}")
                
                # Build unrolling combination [unroll_full, unroll_factor]
                unroll_combination = [u_value, f_value]
                
            elif self.current_layer_name.startswith("matmul"):
                # Load models and scaler for matmul
                with open(unrolling_u_model_path, 'rb') as f:
                    unrolling_u_model = pickle.load(f)
                    
                with open(unrolling_scaler_path, 'rb') as f:
                    unrolling_scaler = pickle.load(f)
                
                # Extract features for prediction
                features = {
                    'B': current_layer.input_batch,
                    'M': current_layer.input_height,
                    'K': current_layer.input_width,
                    'N': current_layer.weight_width,
                    'P': self.performance_oriented_explore
                }
                
                # Prepare input for prediction
                X = [[features['B'], features['M'], features['K'], features['N']]]
                
                # Scale the features
                X_scaled = unrolling_scaler.transform(X)
                
                # Make predictions - for FC layers, the model may predict only unroll count
                u_value = unrolling_u_model.predict(X_scaled)[0]  # Unroll factor or unroll strategy
                
                print(f"Prediction type: {type(u_value)}")
                print(f"Predicted optimal unroll value: {u_value}")
                
                # For FC/matmul layers, we might use a different format for unrolling
                # Assuming the model predicts an unroll factor, with full unroll = 1
                unroll_combination = [1, u_value]  # [unroll_full=1, unroll_factor=u_value]
                
                # If the model predicts 0, it means full unroll
                if u_value == 0:
                    unroll_combination = [1, 0]  # Full unroll once
            
            # Set the current unrolling combination for execution
            self.current_unroll_combination = unroll_combination
            
            # Set the configuration name
            self.current_configuration = f"{self.model_name}_dt_unroll_{self.current_layer_name}_unroll_{unroll_combination[0]}_factor_{unroll_combination[1]}"
            
            print(f"Configuration: {self.current_configuration}")
            print(f"Current unroll combination: {self.current_unroll_combination}")
            
            # Create and execute Docker commands
            self.create_docker_commands()
            self.execute_commands()
            
        except FileNotFoundError as e:
            print(f"Error: Required model files not found - {e}")
            print("Make sure the decision tree models have been trained and saved.")
        except Exception as e:
            print(f"Error during unrolling optimization: {e}")

    def check_available_models(self):
        """Check which model files are available and report to the user"""
        import os
        
        model_status = {
            'conv2d': {
                'permutation': False,
                'tiling': False,
                'unrolling': False
            },
            'fc': {
                'permutation': False,
                'tiling': False,
                'unrolling': False
            }
        }
        
        # Check conv2d models (in ./decision_tree_models directory)
        if os.path.exists(os.path.join('./decision_tree_models', "conv2d_permutation_model.pkl")):
            model_status['conv2d']['permutation'] = True
        
        if (os.path.exists(os.path.join('./decision_tree_models', "conv2d_tiling_O_T_model.pkl")) and 
            os.path.exists(os.path.join('./decision_tree_models', "conv2d_tiling_IC_T_model.pkl"))):
            model_status['conv2d']['tiling'] = True
        
        if (os.path.exists(os.path.join('./decision_tree_models', "conv2d_unrolling_U_model.pkl")) and 
            os.path.exists(os.path.join('./decision_tree_models', "conv2d_unrolling_F_model.pkl"))):
            model_status['conv2d']['unrolling'] = True
        
        # Check fc/matmul models (in models directory)
        if os.path.exists(os.path.join('./decision_tree_models', "fc_permutation_model.pkl")):
            model_status['fc']['permutation'] = True
        
        if (os.path.exists(os.path.join('./decision_tree_models', "fc_tile_k_model.pkl")) and 
            os.path.exists(os.path.join('./decision_tree_models', "fc_tile_n_model.pkl"))):
            model_status['fc']['tiling'] = True
        
        if os.path.exists(os.path.join('./decision_tree_models', "fc_unroll_model.pkl")):
            model_status['fc']['unrolling'] = True
        
        # Print model availability
        print("\n=== Available Models for Optimization ===")
        print("Conv2D Models:")
        print(f"  - Permutation: {'Available' if model_status['conv2d']['permutation'] else 'Not available'}")
        print(f"  - Tiling: {'Available' if model_status['conv2d']['tiling'] else 'Not available'}")
        print(f"  - Unrolling: {'Available' if model_status['conv2d']['unrolling'] else 'Not available'}")
        
        print("\nFully Connected (MatMul) Models:")
        print(f"  - Permutation: {'Available' if model_status['fc']['permutation'] else 'Not available'}")
        print(f"  - Tiling: {'Available' if model_status['fc']['tiling'] else 'Not available'}")
        print(f"  - Unrolling: {'Available' if model_status['fc']['unrolling'] else 'Not available'}")
        print("=========================================\n")
        
        return model_status

    def execute_decision_tree(self):
        """Function to perform Decision Tree Optimization"""
        # Check available models
        model_status = self.check_available_models()
        
        # Process each layer
        for layer_name in self.layers.keys():
            print("===========================")
            print(f"Layer name: {layer_name}")
            # Set the current layer name
            self.current_layer_name = layer_name
            
            # Determine layer type
            layer_type = None
            if layer_name.startswith("conv2d") or layer_name.startswith("depthwise_conv2d"):
                layer_type = 'conv2d'
            elif layer_name.startswith("matmul"):
                layer_type = 'fc'
            else:
                print(f"Skipping layer {layer_name}: unsupported layer type for decision tree optimization")
                continue
            
            # Perform optimizations based on available models and optimization flags
            if self.permute:
                if model_status[layer_type]['permutation']:
                    print(f"Applying decision tree permutation optimization for {layer_name}...")
                    self.perform_opt_permutation()
                else:
                    print(f"Skipping permutation optimization for {layer_name}: models not available")
            
            if self.tile:
                if model_status[layer_type]['tiling']:
                    print(f"Applying decision tree tiling optimization for {layer_name}...")
                    self.perform_opt_tiling()
                else:
                    print(f"Skipping tiling optimization for {layer_name}: models not available")
            
            if self.unroll:
                if model_status[layer_type]['unrolling']:
                    print(f"Applying decision tree unrolling optimization for {layer_name}...")
                    self.perform_opt_unrolling()
                else:
                    print(f"Skipping unrolling optimization for {layer_name}: models not available")
            
            print(f"Completed optimization for layer: {layer_name}")
