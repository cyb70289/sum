CXX = g++
CPPFLAGS = -g -O3 -Wall
CXXFLAGS = -std=c++11
LDLIBS = -lbenchmark -lgtest -lpthread

SYSTEM := $(shell uname -s)
ifeq ($(SYSTEM),Linux)
  CPPFLAGS += -march=native
endif
ifeq ($(SYSTEM),Darwin)
  PROCESSOR := $(shell uname -p)
  ifeq ($(PROCESSOR),arm)
    CPPFLAGS += -I/opt/homebrew/include
    LDFLAGS += -L/opt/homebrew/lib
  endif
endif

sum:

clean:
	rm -rf sum sum.dSYM
