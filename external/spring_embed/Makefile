# Purpose: build helper targets for this repository component.
# Inputs: make variables and source files in this directory.
# Outputs: compiled binaries or cleaned build artifacts.

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wextra
LDFLAGS ?= -lm

# SRC: make target for the external/spring_embed/Makefile build workflow.
SRC := src/main.c
# TARGET: make target for the external/spring_embed/Makefile build workflow.
TARGET := build/spring_embed

# .PHONY: make target for the external/spring_embed/Makefile build workflow.
.PHONY: all clean

# all: make target for the external/spring_embed/Makefile build workflow.
all: $(TARGET)

$(TARGET): $(SRC) src/*.h | build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# build: make target for the external/spring_embed/Makefile build workflow.
build:
	mkdir -p build

# clean: make target for the external/spring_embed/Makefile build workflow.
clean:
	rm -rf build
