"""Python file to perform Design Space Exploration on SODA framework"""

import os
import csv
import subprocess
from utilities import get_layer_info, get_folded_layer_info
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
        self.part1 = args.part1 # Execute only part1 of the flow if True
        self.part2 = args.part2 # Execute only part2 of the flow if True
        self.commands = None    # Current set of docker commands to execute
        self.current_configuration = None # Current configuration of loop optimization executed
        self.current_layer_name = None  # Current layer being explored
        self.trial_number = args.trial # Trial number for loop optimization

    def create_docker_commands(self):
        """Function to create docker commands"""
        # Base command to initiate docker
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
        # Generate different commands for loop optimizations
        if self.permute:
            pass
        elif self.tile:
            # Remove '-promote-buffers-to-stack' pass
            soda_opt_bambu_pipeline.pop(4)
        elif self.unroll:
            # Remove '-affine-scalrep' pass 
            soda_opt_bambu_pipeline.pop(0)
            # Remove 'affine-data-copy-generate' pass
            soda_opt_bambu_pipeline.pop(1)
            # Remove 'erase-buffer-deallocation' pass
            soda_opt_bambu_pipeline.pop(1)
            # Remove 'promote-buffers-to-stack' pass
            soda_opt_bambu_pipeline.pop(1)
        # Concatenate all commands to one string
        soda_command = base_command + " ".join(soda_opt_bambu_pipeline)
        
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
            
            # If part1 is set, only execute commands 1a to 1c
            if self.part1:
                for key in ["1a-soda", "1b-soda", "1c-mlir"]:
                    print(f"Executing: {key}")
                    subprocess.run(self.commands[key], shell=True, stdout=output_file,
                                stderr=output_file, check=True)
                print(f"Part 1 execution completed for configuration: {self.current_configuration}")
                return
                
            # If part2 is set, execute commands 1d to 3-openroad
            if self.part2:
                for key in ["1d-soda", "1e-mlir-opt", "1f-soda", "2-bambu", "3-openroad"]:
                    print(f"Executing: {key}")
                    subprocess.run(self.commands[key], shell=True, stdout=output_file,
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
                        output_folder = './output_llm'
                        # Command to delete files and folders of current configuration
                        command = f"find {output_folder} -name \
                            '*{self.current_configuration}*' -exec rm -rf {{}} +"
                        # Execute the command to clean up the output folder
                        subprocess.run(command, shell=True, check=True)
                
                print(f"Part 2 execution completed for configuration: {self.current_configuration}")
                return
                
            # If neither part1 nor part2 is set, execute all commands
            for key, command in self.commands.items():
                print(f"Executing: {key}")
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
                    output_folder = './output_llm'
                    # Command to delete files and folders of current configuration
                    command = f"find {output_folder} -name \
                        '*{self.current_configuration}*' -exec rm -rf {{}} +"
                    # Execute the command to clean up the output folder
                    subprocess.run(command, shell=True, check=True)

    def execute(self):
        """Function to execute a single optimization for each layer"""
        if self.permute:
            loop_optimization = '_permute_'
        elif self.tile:
            loop_optimization = '_tile_'
        elif self.unroll:
            loop_optimization = '_unroll_'
        for layer_name in self.layers.keys():
            self.current_configuration = self.model_name + "_" + layer_name + loop_optimization + "_trial_" + self.trial_number
            self.current_layer_name = layer_name
            self.create_docker_commands()
            self.execute_commands()