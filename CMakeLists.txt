cmake_minimum_required(VERSION 3.17)
project(LU_Paralelo)

set(CMAKE_CXX_STANDARD 14)

add_executable(LU_Paralelo parallelLU.cpp)


#Acesso ao GSL-GNU
target_link_libraries(LU_Paralelo -lgsl -lgslcblas -lm)

#Acesso ao OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(SOURCE_FILES parallelLU.cpp)