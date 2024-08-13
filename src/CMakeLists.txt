add_library(small_la_library INTERFACE)

target_include_directories(small_la_library INTERFACE ../include)

target_compile_features(small_la_library INTERFACE cxx_std_17)