# Compiler and flags
NVCC = nvcc
CFLAGS = -Iinclude -arch=sm_60

# Project directories
SRCDIR = src
KERNELDIR = $(SRCDIR)/kernels
BUILDDIR = build
BINDIR = bin

# Project files
EXECUTABLE = parallel_graph_clustering
TARGET = $(BINDIR)/$(EXECUTABLE)

# Source files
SRC_FILES = $(SRCDIR)/graph.cu $(SRCDIR)/main.cu $(SRCDIR)/utils.cu $(KERNELDIR)/insertNodesAndEdges.cu

# Object files
OBJ_FILES = $(patsubst $(KERNELDIR)/%.cu, $(BUILDDIR)/kernels/%.o, $(wildcard $(KERNELDIR)/*.cu)) \
			$(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(wildcard $(SRCDIR)/*.cu))

# Print variables for debugging
$(info SRC_FILES = $(SRC_FILES))
$(info OBJ_FILES = $(OBJ_FILES))

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CFLAGS) $(OBJ_FILES) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

$(BUILDDIR)/kernels/%.o: $(KERNELDIR)/%.cu
	@mkdir -p $(BUILDDIR)/kernels
	$(NVCC) $(CFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
	@echo "Clean complete"

.PHONY: all clean




