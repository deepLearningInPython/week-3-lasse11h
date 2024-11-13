import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    output_length = input_length - kernel_length + 1
    return output_length

# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------
def compute_output_size_1d(input_array, kernel_array):
    # Calculate the output length for 1D convolution
    input_length = len(input_array)
    kernel_length = len(kernel_array)
    return input_length - kernel_length + 1

def convolve_1d(input_array, kernel_array):
    # Calculate the output length
    output_length = compute_output_size_1d(input_array, kernel_array)
    # Initialize an empty output array of the calculated length
    output_array = np.zeros(output_length)
    
    # Perform convolution without flipping the kernel
    for i in range(output_length):
        # Element-wise multiplication and sum for the current segment
        output_array[i] = np.sum(input_array[i:i + len(kernel_array)] * kernel_array)
    
    return output_array
 # Test case
input_array = np.array([1, 2, 3, 4, 6])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    # Get dimensions of input and kernel matrices
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    
    # Compute output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    # Return the dimensions as a tuple
    return (output_height, output_width)
# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------
def compute_output_size_2d(input_matrix, kernel_matrix):
    # Get dimensions of input and kernel matrices
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    
    # Compute output dimensions
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    return (output_height, output_width)

def convolute_2d(input_matrix, kernel_matrix):
    # Calculate output dimensions
    output_height, output_width = compute_output_size_2d(input_matrix, kernel_matrix)
    # Initialize the output matrix with zeros
    output_matrix = np.zeros((output_height, output_width))
    
    # Perform 2D convolution without flipping the kernel
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current segment of the input matrix
            current_segment = input_matrix[i:i + kernel_matrix.shape[0], j:j + kernel_matrix.shape[1]]
            # Perform element-wise multiplication and sum the result
            output_matrix[i, j] = np.sum(current_segment * kernel_matrix)
    
    return output_matrix

# Test case
input_matrix = np.array([[2, 2, 3], [4, 5, 6], [7, 8, 10]])
kernel_matrix = np.array([[1, 0], [0, -1]])
print(convolute_2d(input_matrix, kernel_matrix)) 

# -----------------------------------------------
