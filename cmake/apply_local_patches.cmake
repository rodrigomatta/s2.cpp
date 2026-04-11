if(NOT DEFINED S2_PATCH_ROOT)
    message(FATAL_ERROR "S2_PATCH_ROOT is required")
endif()

if(NOT DEFINED S2_PATCH_EXECUTABLE OR S2_PATCH_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "S2_PATCH_EXECUTABLE is required")
endif()

set(S2_PATCH_DIR "${S2_PATCH_ROOT}/patches")
file(GLOB S2_PATCH_FILES "${S2_PATCH_DIR}/*.patch")

if(NOT S2_PATCH_FILES)
    message(STATUS "No local patches found in ${S2_PATCH_DIR}")
    return()
endif()

foreach(_patch IN LISTS S2_PATCH_FILES)
    execute_process(
        COMMAND "${S2_PATCH_EXECUTABLE}" --batch --dry-run --forward -p1 -i "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _can_apply
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_can_apply EQUAL 0)
        execute_process(
            COMMAND "${S2_PATCH_EXECUTABLE}" --batch --forward -p1 -i "${_patch}"
            WORKING_DIRECTORY "${S2_PATCH_ROOT}"
            RESULT_VARIABLE _apply_result
        )
        if(NOT _apply_result EQUAL 0)
            message(FATAL_ERROR "Failed to apply local patch: ${_patch}")
        endif()
        message(STATUS "Applied local patch: ${_patch}")
        continue()
    endif()

    execute_process(
        COMMAND "${S2_PATCH_EXECUTABLE}" --batch --dry-run -R -p1 -i "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_already_applied EQUAL 0)
        message(STATUS "Local patch already applied: ${_patch}")
    else()
        message(FATAL_ERROR "Local patch is not applicable: ${_patch}")
    endif()
endforeach()
