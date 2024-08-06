file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${small_la_SOURCE_DIR}/include/small_la")

add_library(small_la_library small_la.cpp ${HEADER_LIST})

target_include_directories(small_la_library PUBLIC ../include)

target_compile_features(small_la_library PUBLIC cxx_std_17)