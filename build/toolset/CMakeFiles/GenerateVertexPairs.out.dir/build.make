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
include toolset/CMakeFiles/GenerateVertexPairs.out.dir/depend.make

# Include the progress variables for this target.
include toolset/CMakeFiles/GenerateVertexPairs.out.dir/progress.make

# Include the compile flags for this target's objects.
include toolset/CMakeFiles/GenerateVertexPairs.out.dir/flags.make

toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o: toolset/CMakeFiles/GenerateVertexPairs.out.dir/flags.make
toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o: ../toolset/generate_vertex_pairs.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o"
	cd /home/shiwenlan/test/parallel3/build/toolset && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o -c /home/shiwenlan/test/parallel3/toolset/generate_vertex_pairs.cpp

toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.i"
	cd /home/shiwenlan/test/parallel3/build/toolset && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shiwenlan/test/parallel3/toolset/generate_vertex_pairs.cpp > CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.i

toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.s"
	cd /home/shiwenlan/test/parallel3/build/toolset && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shiwenlan/test/parallel3/toolset/generate_vertex_pairs.cpp -o CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.s

# Object files for target GenerateVertexPairs.out
GenerateVertexPairs_out_OBJECTS = \
"CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o"

# External object files for target GenerateVertexPairs.out
GenerateVertexPairs_out_EXTERNAL_OBJECTS =

toolset/GenerateVertexPairs.out: toolset/CMakeFiles/GenerateVertexPairs.out.dir/generate_vertex_pairs.cpp.o
toolset/GenerateVertexPairs.out: toolset/CMakeFiles/GenerateVertexPairs.out.dir/build.make
toolset/GenerateVertexPairs.out: util/libgraph.so
toolset/GenerateVertexPairs.out: util/liblog.so
toolset/GenerateVertexPairs.out: toolset/CMakeFiles/GenerateVertexPairs.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GenerateVertexPairs.out"
	cd /home/shiwenlan/test/parallel3/build/toolset && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GenerateVertexPairs.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
toolset/CMakeFiles/GenerateVertexPairs.out.dir/build: toolset/GenerateVertexPairs.out

.PHONY : toolset/CMakeFiles/GenerateVertexPairs.out.dir/build

toolset/CMakeFiles/GenerateVertexPairs.out.dir/clean:
	cd /home/shiwenlan/test/parallel3/build/toolset && $(CMAKE_COMMAND) -P CMakeFiles/GenerateVertexPairs.out.dir/cmake_clean.cmake
.PHONY : toolset/CMakeFiles/GenerateVertexPairs.out.dir/clean

toolset/CMakeFiles/GenerateVertexPairs.out.dir/depend:
	cd /home/shiwenlan/test/parallel3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shiwenlan/test/parallel3 /home/shiwenlan/test/parallel3/toolset /home/shiwenlan/test/parallel3/build /home/shiwenlan/test/parallel3/build/toolset /home/shiwenlan/test/parallel3/build/toolset/CMakeFiles/GenerateVertexPairs.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : toolset/CMakeFiles/GenerateVertexPairs.out.dir/depend

