CXX = g++
CC = gcc
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
CFLAGS = -O3 -Wall -std=c99
LDFLAGS = $(shell pkg-config --libs opencv4) -lm

app: main.cpp neural_network.c neural_network.h
	$(CXX) $(CXXFLAGS) main.cpp neural_network.c -o $@ $(LDFLAGS)

clean:
	rm -f app