set(SOURCE_FILES 
    cuda_starter.cu)

add_library(cuda_starter SHARED ${SOURCE_FILES})

# set_target_properties(cuda_starter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_starter ${CUDA_LIBRARIES})


