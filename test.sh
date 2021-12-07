#!/bin/bash
mkdir build
cd build
cmake ..
make
# cd toolset
# ./EdgeList2DirectedCSR.out ../../dataset/soc-Epinions1/ ../../dataset/soc-Epinions1/for_demo/ # 
# ./GenerateVertexPairs.out ../../dataset/soc-Epinions1/for_demo/ ../../dataset/soc-Epinions1/for_demo/ 1000 0.1 > test.txt 2> log0.txt
cd cycle
./CycleEnumerator.out ../../dataset/soc-Epinions1/for_demo/ ../../dataset/soc-Epinions1/for_demo/hot2hot_pairs.bin ./ "IDX_DFS" 7 120 > h2h.txt 2> log1.txt
# ./CycleEnumerator.out ../../dataset/soc-Epinions1/for_demo/ ../../dataset/soc-Epinions1/for_demo/hot2unhot_pairs.bin ./ "IDX_DFS" 7 30 > h2u.txt 2> log2.txt
# ./CycleEnumerator.out ../../dataset/soc-Epinions1/for_demo/ ../../dataset/soc-Epinions1/for_demo/unhot2hot_pairs.bin ./ "IDX_DFS" 7 30 > u2h.txt 2> log3.txt
# ./CycleEnumerator.out ../../dataset/soc-Epinions1/for_demo/ ../../dataset/soc-Epinions1/for_demo/unhot2unhot_pairs.bin ./ "IDX_DFS" 7 30 > u2u.txt 2> log4.txt
# cd ../cycle/script
# python3 test_cycle_enumerator.py ../../build/cycle/CycleEnumerator.out

# /bin/sh test.sh