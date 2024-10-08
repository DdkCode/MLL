import subprocess
import os


def run_setup(directory):
    setup_script = os.path.join(directory, 'setup.py')
    print(setup_script)

    # Check if setup.py exists in the directory
    if os.path.exists(setup_script):
        print(f"Running setup.py in {directory}...")
        try:
            # Run the setup script and capture the output
            result = subprocess.call(
                ['python', 'setup.py', 'build_ext', '--inplace'],
                cwd=directory
                # check=True,
                # capture_output=True,
                # text=True
            )

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running setup.py in {directory}")
            print("Output: ")
            print(e.output)
            print(e.stdout)  # Print the standard output
            print("Error: ")
            print(e.stderr)  # Print the standard error
            raise  # Re-raise the exception after printing for visibility

    else:
        print(f"No setup_main.py found in {directory}")


def main():
    # Directories containing setup.py files
    directories = [
        'package//Models//Neural_Network',
        'package//Models//Classifier//Rotation_Forest',
        'package//Models//Classifier//DecisionTrees',
        'package//Models//Classifier//Random_Forest',
    ]

    # Iterate through each directory and run setup.py
    for directory in directories:
        run_setup(directory)


if __name__ == "__main__":
    main()
