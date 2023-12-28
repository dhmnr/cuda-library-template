
add_executable(
  run_tests
  test_vector_add.cpp
  ../include/acutest.h
)
target_link_libraries(run_tests cuda_starter)
