CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11 -Wall -Wextra
LDFLAGS ?= -lm

SRC := src/main.c
TARGET := build/spring_embed

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) src/*.h | build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

build:
	mkdir -p build

clean:
	rm -rf build
