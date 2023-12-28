set(SOURCE_FILES 
    vector_add.cu
    utils.cu)

add_library(cuda_starter SHARED ${SOURCE_FILES})

# Turn on separable compilation for using device functions across library boundary
# set_target_properties(cuda_starter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_starter ${CUDA_LIBRARIES})


