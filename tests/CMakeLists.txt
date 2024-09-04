# Testing library
FetchContent_Declare(
        catch
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.6)
FetchContent_MakeAvailable(catch)
# Adds Catch2::Catch2


set(TEST_LIST
"SVD_test"
)

foreach(testVAR IN LISTS TEST_LIST)
    add_executable("${testVAR}" "${testVAR}.cpp")
    target_compile_features("${testVAR}" PRIVATE cxx_std_17)
    target_link_libraries("${testVAR}" PRIVATE small_la_library Catch2::Catch2)
    add_test(NAME "${testVAR}" COMMAND "${testVAR}")
    set_target_properties("${testVAR}" PROPERTIES LINKER_LANGUAGE CXX)
endforeach()
