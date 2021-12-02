# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shiwenlan/test/parallel3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shiwenlan/test/parallel3/build

# Include any dependencies generated for this target.
include util/CMakeFiles/graph.dir/depend.make

# Include the progress variables for this target.
include util/CMakeFiles/graph.dir/progress.make

# Include the compile flags for this target's objects.
include util/CMakeFiles/graph.dir/flags.make

util/CMakeFiles/graph.dir/graph/directed_graph.cpp.o: util/CMakeFiles/graph.dir/flags.make
util/CMakeFiles/graph.dir/graph/directed_graph.cpp.o: ../util/graph/directed_graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object util/CMakeFiles/graph.dir/graph/directed_graph.cpp.o"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph.dir/graph/directed_graph.cpp.o -c /home/shiwenlan/test/parallel3/util/graph/directed_graph.cpp

util/CMakeFiles/graph.dir/graph/directed_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph.dir/graph/directed_graph.cpp.i"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shiwenlan/test/parallel3/util/graph/directed_graph.cpp > CMakeFiles/graph.dir/graph/directed_graph.cpp.i

util/CMakeFiles/graph.dir/graph/directed_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph.dir/graph/directed_graph.cpp.s"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shiwenlan/test/parallel3/util/graph/directed_graph.cpp -o CMakeFiles/graph.dir/graph/directed_graph.cpp.s

util/CMakeFiles/graph.dir/graph/graph_operation.cpp.o: util/CMakeFiles/graph.dir/flags.make
util/CMakeFiles/graph.dir/graph/graph_operation.cpp.o: ../util/graph/graph_operation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object util/CMakeFiles/graph.dir/graph/graph_operation.cpp.o"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph.dir/graph/graph_operation.cpp.o -c /home/shiwenlan/test/parallel3/util/graph/graph_operation.cpp

util/CMakeFiles/graph.dir/graph/graph_operation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph.dir/graph/graph_operation.cpp.i"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shiwenlan/test/parallel3/util/graph/graph_operation.cpp > CMakeFiles/graph.dir/graph/graph_operation.cpp.i

util/CMakeFiles/graph.dir/graph/graph_operation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph.dir/graph/graph_operation.cpp.s"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shiwenlan/test/parallel3/util/graph/graph_operation.cpp -o CMakeFiles/graph.dir/graph/graph_operation.cpp.s

# Object files for target graph
graph_OBJECTS = \
"CMakeFiles/graph.dir/graph/directed_graph.cpp.o" \
"CMakeFiles/graph.dir/graph/graph_operation.cpp.o"

# External object files for target graph
graph_EXTERNAL_OBJECTS =

util/libgraph.so: util/CMakeFiles/graph.dir/graph/directed_graph.cpp.o
util/libgraph.so: util/CMakeFiles/graph.dir/graph/graph_operation.cpp.o
util/libgraph.so: util/CMakeFiles/graph.dir/build.make
util/libgraph.so: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
util/libgraph.so: /usr/lib/x86_64-linux-gnu/libpthread.so
util/libgraph.so: util/CMakeFiles/graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libgraph.so"
	cd /home/shiwenlan/test/parallel3/build/util && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
util/CMakeFiles/graph.dir/build: util/libgraph.so

.PHONY : util/CMakeFiles/graph.dir/build

util/CMakeFiles/graph.dir/clean:
	cd /home/shiwenlan/test/parallel3/build/util && $(CMAKE_COMMAND) -P CMakeFiles/graph.dir/cmake_clean.cmake
.PHONY : util/CMakeFiles/graph.dir/clean

util/CMakeFiles/graph.dir/depend:
	cd /home/shiwenlan/test/parallel3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shiwenlan/test/parallel3 /home/shiwenlan/test/parallel3/util /home/shiwenlan/test/parallel3/build /home/shiwenlan/test/parallel3/build/util /home/shiwenlan/test/parallel3/build/util/CMakeFiles/graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : util/CMakeFiles/graph.dir/depend

