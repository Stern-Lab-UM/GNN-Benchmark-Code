CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wextra

SRC := src/main.c
BIN := build/vertex_model_generator

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) src/*.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) -lm

clean:
	rm -rf build
