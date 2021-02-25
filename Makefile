CXX = g++
CPPFLAGS = -g -O3 -Wall -I/opt/homebrew/include
CXXFLAGS = -std=c++11
LDFLAGS = -L/opt/homebrew/lib
LDLIBS = -lbenchmark -lgtest -lpthread

sum:

clean:
	rm -rf sum sum.dSYM
