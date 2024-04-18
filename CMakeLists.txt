

cmake_minimum_required(VERSION 3.1...3.28)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


project(
    spgemm-bench
    VERSION 1.0
    LANGUAGES CXX
)


add_executable(spgemm-bench 

src/main.cpp

)

# Find and link MPI
find_package(MPI REQUIRED)
target_link_libraries(spgemm-bench PUBLIC MPI::MPI_CXX)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(spgemm-bench PUBLIC OpenMP::OpenMP_CXX)

# Find and link CUDA
target_include_directories(spgemm-bench PRIVATE "/usr/local/cuda/include/")
target_link_libraries(spgemm-bench PUBLIC "/usr/local/cuda/lib64/")


# Add include dir 
target_include_directories(spgemm-bench PRIVATE include)

# Add CombBLAS
target_include_directories(spgemm-bench PRIVATE "/home/jb2695/CombBLAS/install/include/")
target_link_libraries(spgemm-bench PRIVATE "/home/jb2695/CombBLAS/install/lib/libCombBLAS.a")
target_link_libraries(spgemm-bench PRIVATE "/home/jb2695/CombBLAS/install/lib/libGraphGenlib.a")
target_link_libraries(spgemm-bench PRIVATE "/home/jb2695/CombBLAS/install/lib/libUsortlib.a")

# TODO: Install target

