
add_executable(test1 test1.cpp)

target_link_libraries(test1 PRIVATE small_la_library)