SET(CYCLE_ENUMERATOR_SRC_FILE_LIST main.cpp cycle_enumerator.cpp)
ADD_EXECUTABLE(CycleEnumerator.out ${CYCLE_ENUMERATOR_SRC_FILE_LIST})
TARGET_LINK_LIBRARIES(CycleEnumerator.out graph log io)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fsanitize=thread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()
