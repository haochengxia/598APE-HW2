#######################################
# negative optimization
#######################################
USE_ICPX ?= 0
#######################################

ifeq ($(USE_ICPX), 1)
    CXX := icpx
else
    CXX := g++
endif

CXXFLAGS := -std=c++17 -Wall -Wextra -I include -I. 
#######################

ENABLE_BETTER_OPT ?= 0

# for common optimization, like allocation
ENABLE_COMMON ?= 0

# for stack operation, use array to simulate stack
ENABLE_STACK ?= 0

# for remove insertion sort
ENABLE_REMOVE_INSERTION_SORT ?= 0

# for kernel execution, enable one of them
ENABLE_OMP ?= 0
ENABLE_OMP_BATCHED ?= 0
ENABLE_OMP_PRECOMPILE_BATCHED ?= 0

ifeq ($(ENABLE_BETTER_OPT), 1)
    EXTRA_FLAGS += -O3
	ifeq ($(USE_ICPX), 1)
		EXTRA_FLAGS += -xCORE-AVX2
	else
		EXTRA_FLAGS += -march=native
	endif
endif

ifeq ($(ENABLE_COMMON), 1)
    EXTRA_FLAGS += -DENABLE_COMMON -fopenmp
endif

ifeq ($(ENABLE_STACK), 1)
    EXTRA_FLAGS += -DENABLE_STACK
endif

ifeq ($(ENABLE_REMOVE_INSERTION_SORT), 1)
    EXTRA_FLAGS += -DENABLE_REMOVE_INSERTION_SORT
endif

ifeq ($(ENABLE_OMP), 1)
    EXTRA_FLAGS += -DENABLE_OMP -fopenmp
endif

ifeq ($(ENABLE_OMP_BATCHED), 1)
    EXTRA_FLAGS += -DENABLE_OMP_BATCHED -fopenmp
endif

ifeq ($(ENABLE_OMP_PRECOMPILE_BATCHED), 1)
    EXTRA_FLAGS += -DENABLE_OMP_PRECOMPILE_BATCHED -fopenmp
endif

#############################################
# if use icpx, shut down the following flags: ENABLE_OMP, ENABLE_OMP_BATCHED, ENABLE_OMP_PRECOMPILE_BATCHED
ifeq ($(USE_ICPX), 1)
    EXTRA_FLAGS += -DENABLE_OMP=0 -DENABLE_OMP_BATCHED=0 -DENABLE_OMP_PRECOMPILE_BATCHED=0
endif

#############################################


# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BENCH_DIR := benchmark

# Find all cpp files in src directory
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# Generate object file names
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Benchmark program
BENCH_SRC := $(BENCH_DIR)/genetic_benchmark.cpp
BENCH_BIN := genetic_benchmark

# Default target
all: directories $(BENCH_BIN) 

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BENCH_DIR)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $< -o $@

# build benchmark lib 
$(BENCH_BIN): $(BENCH_SRC) $(OBJS)
	@echo "Building benchmark program..."
	@$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) $< $(OBJS) -o $@ 

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJ_DIR) $(BENCH_BIN)

# Clean and rebuild
rebuild: clean all

.PHONY: all clean rebuild directories 
