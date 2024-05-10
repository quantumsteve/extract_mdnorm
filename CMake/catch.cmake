# catch2
set( catch2_url "https://github.com/catchorg/Catch2.git" )

set( catch2_tag "v3.6.0" )

include (FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY ${catch2_url}
    GIT_TAG ${catch2_tag}
)
FetchContent_MakeAvailable(Catch2)

#Mark CATCH variables as advanced.
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    string(FIND "${_variableName}" "CATCH_" out)
    if("${out}" EQUAL 0)
        mark_as_advanced(${_variableName})
    endif()
endforeach()

