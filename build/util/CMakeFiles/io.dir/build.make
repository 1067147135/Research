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
include util/CMakeFiles/io.dir/depend.make

# Include the progress variables for this target.
include util/CMakeFiles/io.dir/progress.make

# Include the compile flags for this target's objects.
include util/CMakeFiles/io.dir/flags.make

util/CMakeFiles/io.dir/io/io.cpp.o: util/CMakeFiles/io.dir/flags.make
util/CMakeFiles/io.dir/io/io.cpp.o: ../util/io/io.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object util/CMakeFiles/io.dir/io/io.cpp.o"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/io.dir/io/io.cpp.o -c /home/shiwenlan/test/parallel3/util/io/io.cpp

util/CMakeFiles/io.dir/io/io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/io.dir/io/io.cpp.i"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shiwenlan/test/parallel3/util/io/io.cpp > CMakeFiles/io.dir/io/io.cpp.i

util/CMakeFiles/io.dir/io/io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/io.dir/io/io.cpp.s"
	cd /home/shiwenlan/test/parallel3/build/util && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shiwenlan/test/parallel3/util/io/io.cpp -o CMakeFiles/io.dir/io/io.cpp.s

# Object files for target io
io_OBJECTS = \
"CMakeFiles/io.dir/io/io.cpp.o"

# External object files for target io
io_EXTERNAL_OBJECTS =

util/libio.so: util/CMakeFiles/io.dir/io/io.cpp.o
util/libio.so: util/CMakeFiles/io.dir/build.make
util/libio.so: util/CMakeFiles/io.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shiwenlan/test/parallel3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libio.so"
	cd /home/shiwenlan/test/parallel3/build/util && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/io.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
util/CMakeFiles/io.dir/build: util/libio.so

.PHONY : util/CMakeFiles/io.dir/build

util/CMakeFiles/io.dir/clean:
	cd /home/shiwenlan/test/parallel3/build/util && $(CMAKE_COMMAND) -P CMakeFiles/io.dir/cmake_clean.cmake
.PHONY : util/CMakeFiles/io.dir/clean

util/CMakeFiles/io.dir/depend:
	cd /home/shiwenlan/test/parallel3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shiwenlan/test/parallel3 /home/shiwenlan/test/parallel3/util /home/shiwenlan/test/parallel3/build /home/shiwenlan/test/parallel3/build/util /home/shiwenlan/test/parallel3/build/util/CMakeFiles/io.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : util/CMakeFiles/io.dir/depend

