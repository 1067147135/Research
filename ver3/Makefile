CXX = g++
CXXFLAGS = -g -Wall -Wextra -O3


all: bin bin/benchmark 
#bin/construct_index bin/query_distance 
bin:
	mkdir -p bin

# bin/construct_index: samples/construct_index_main.cc src/pruned_landmark_labeling.h samples/io.h
# 	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^ -fopenmp

# bin/query_distance: samples/query_distance_main.cc src/pruned_landmark_labeling.h samples/io.h
# 	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^ -fopenmp

bin/benchmark: samples/benchmark_main.cc src/pruned_landmark_labeling.h samples/io.h
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $^ -fopenmp -fsanitize=thread




.PHONY: clean


clean:
	rm -rf bin