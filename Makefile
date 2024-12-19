# Compiler and flags
NVCC = nvcc
CFLAGS = -Iinclude -arch=sm_60
CXX = g++
CXXFLAGS = -Iinclude

# Project directories
SRCDIR = src
KERNELDIR = $(SRCDIR)/kernels
BUILDDIR = build
BINDIR = bin

# Project files
EXECUTABLE = parallel_graph_clustering
TARGET = $(BINDIR)/$(EXECUTABLE)

# Source files
CUDA_SRC_FILES = $(SRCDIR)/graph.cu $(SRCDIR)/main.cu $(SRCDIR)/utils.cu $(KERNELDIR)/insertNodesAndEdges.cu
CPP_SRC_FILES = $(SRCDIR)/GraphClusteringCPU.cpp

# Object files
CUDA_OBJ_FILES = $(patsubst $(KERNELDIR)/%.cu, $(BUILDDIR)/kernels/%.o, $(wildcard $(KERNELDIR)/*.cu)) \
                 $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(wildcard $(SRCDIR)/*.cu))
CPP_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))


# Print variables for debugging
$(info CUDA_SRC_FILES = $(CUDA_SRC_FILES))
$(info CPP_SRC_FILES = $(CPP_SRC_FILES))
$(info CUDA_OBJ_FILES = $(CUDA_OBJ_FILES))
$(info CPP_OBJ_FILES = $(CPP_OBJ_FILES))

# Build rules
all: $(TARGET)

$(TARGET): $(CUDA_OBJ_FILES) $(CPP_OBJ_FILES)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) $(CUDA_OBJ_FILES) $(CPP_OBJ_FILES) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

$(BUILDDIR)/kernels/%.o: $(KERNELDIR)/%.cu
	@mkdir -p $(BUILDDIR)/kernels
	$(NVCC) $(CFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
	@echo "Clean complete"

.PHONY: all clean




