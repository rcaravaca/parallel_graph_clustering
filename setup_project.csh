#!/bin/bash

# Define the root of the project
project_root="./"


# Create subdirectories
mkdir -p $project_root/src/kernels
mkdir -p $project_root/include
mkdir -p $project_root/data
mkdir -p $project_root/tests
mkdir -p $project_root/build
mkdir -p $project_root/doc

# Create example source files
touch $project_root/src/main.cu
touch $project_root/src/graph.cu
touch $project_root/src/graph.h
touch $project_root/src/utils.cu
touch $project_root/src/utils.h
touch $project_root/src/kernels/bfs_kernel.cu
touch $project_root/src/kernels/bfs_kernel.h

# Create include files
touch $project_root/include/cuda_utils.h
touch $project_root/include/config.h

# Create an example data file
touch $project_root/data/input_data.txt

# Create test files
touch $project_root/tests/test_graph.cu
touch $project_root/tests/test_bfs.cu


# Create a README file in the doc directory
cat << EOF > $project_root/doc/README.md
# Parallel Graph Clustering

This is a sample CUDA project. It contains:

- **src/**: Main source code
- **include/**: Common header files
- **data/**: Input data files
- **tests/**: Unit and integration tests
- **build/**: Build artifacts
- **doc/**: Project documentation

## Build

To build the project, navigate to the root directory and run:

\`\`\`
make
\`\`\`

To clean the build artifacts, run:

\`\`\`
make clean
\`\`\`
EOF

echo "Project structure successfully created in $project_root"

