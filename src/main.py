"""Python main file to run Design Space Exploration for SODA framework"""

from pathlib import Path
from parse_arguments import parse_arguments
from read_mlir import read_file
from create_mlir_files import MlirFiles
from design_space_exploration import DSE

def main():
    """Main function to perform Design Space Exploration"""

    print("\n\n**************************************************\n")
    print("1. Parsing arguments")

    # Parse command-line arguments
    args = parse_arguments()
    print(args)

    # Check if an MLIR file is provided
    if args.read_mlir is not None:
        path = Path(args.read_mlir)

        # Check if the provided MLIR file exists
        if path.exists():

            print("\n\n**************************************************\n")
            print("2. Reading MLIR files")

            layers = read_file(args)

            # Check if a start layer is specified
            if args.start_layer is not None:
                start_layer = int(args.start_layer)

                # Check if an end layer is specified
                if args.end_layer is not None:
                    end_layer = int(args.end_layer)
                else:
                    end_layer = len(layers)

                # Validate the range of start and end layers
                if start_layer > 0 and end_layer <= len(layers):
                    # Filter layers within the specified range
                    filtered_layers = {}
                    for key, value in layers.items():
                        number = int(key.split("_")[-1])
                        if number >= start_layer and number <= end_layer:
                            filtered_layers[key] = value
                    layers = filtered_layers
                else:
                    print("Start layer out of range!")

            # Check if a specific layer is selected
            elif args.select_layer is not None:
                select_layer = int(args.select_layer)

                # Validate the selected layer
                if select_layer > 0 and select_layer <= len(layers):
                    # Filter the specific selected layer
                    filtered_layers = {}
                    for key, value in layers.items():
                        number = int(key.split("_")[-1])
                        if number == select_layer:
                            filtered_layers[key] = value
                            break
                    layers = filtered_layers
                else:
                    print("Select layer out of range!")

            print("\n\n**************************************************\n")
            print("3. Creating temporary MLIR files")

            # Create MLIR files from the filtered layers
            layers = MlirFiles(args, layers).execute()

            print("\n\n**************************************************\n")
            print("4. Design Space Exploration")

            # Perform design space exploration on the layers
            layers = DSE(args, layers).execute()

        else:
            print("MLIR file doesn't exist!")
    else:
        print("No input MLIR file given!")

if __name__ == "__main__":
    main()
