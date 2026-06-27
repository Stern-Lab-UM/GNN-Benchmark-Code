# Purpose: build helper targets for this repository component.
# Inputs: make variables and source files in this directory.
# Outputs: compiled binaries or cleaned build artifacts.

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wextra

# SRC: make target for the data_generation/vertex_model/Makefile build workflow.
SRC := src/main.c
# BIN: make target for the data_generation/vertex_model/Makefile build workflow.
BIN := build/vertex_model_generator

# .PHONY: make target for the data_generation/vertex_model/Makefile build workflow.
.PHONY: all clean

# all: make target for the data_generation/vertex_model/Makefile build workflow.
all: $(BIN)

$(BIN): $(SRC) src/*.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) -lm

# clean: make target for the data_generation/vertex_model/Makefile build workflow.
clean:
	rm -rf build
