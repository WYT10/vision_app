
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS
    /usr/include/onnxruntime
    /usr/local/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
  PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)

if (ONNXRUNTIME_FOUND)
  set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
  set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
endif()
