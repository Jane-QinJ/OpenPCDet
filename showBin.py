def display_bin_file_content(file_path):
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            print("Hexadecimal representation of the file content:")
            print(content.hex())
            print("\nByte-by-byte representation of the file content:")
            print(list(content))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = 'data/custom/gt_database/000000_Pedestrian_0.bin'  # Replace with the path to your binary file
display_bin_file_content(file_path)
