#!/bin/bash
make
bin/benchmark samples/b_edge_list.bin > output.txt 2> log.txt
# /bin/sh test.sh