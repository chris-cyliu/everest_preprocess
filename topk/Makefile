export CXX = g++

export CXXFLAGS = -std=c++11 -Wall -O3
export LDFLAGS = -lpthread -lcurl -fopenmp

BIN = topk

.PHONY: clean all

all: $(BIN)

$(BIN): $(shell ls *.c) $(shell ls *.cpp) $(shell ls *.h)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(shell ls *.c) $(shell ls *.cpp) -o topk

clean:
	rm -rf $(BIN)
