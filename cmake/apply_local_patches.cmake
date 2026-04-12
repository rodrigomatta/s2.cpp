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
if(WIN32)
    execute_process(
        COMMAND "${S2_PATCH_EXECUTABLE}" apply --check -p1 "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _can_apply
        OUTPUT_QUIET
        ERROR_QUIET
    )
else()
    execute_process(
		COMMAND "${S2_PATCH_EXECUTABLE}" --batch --dry-run --forward -p1 -i "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _can_apply
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()

    if(_can_apply EQUAL 0)
if(WIN32)
        execute_process(
			COMMAND "${S2_PATCH_EXECUTABLE}" apply -p1 "${_patch}"
            WORKING_DIRECTORY "${S2_PATCH_ROOT}"
            RESULT_VARIABLE _apply_result
        )
else()
        execute_process(
            COMMAND "${S2_PATCH_EXECUTABLE}" --batch --forward -p1 -i "${_patch}"
            WORKING_DIRECTORY "${S2_PATCH_ROOT}"
            RESULT_VARIABLE _apply_result
        )
endif()
        if(NOT _apply_result EQUAL 0)
            message(FATAL_ERROR "Failed to apply local patch: ${_patch}")
        endif()
        message(STATUS "Applied local patch: ${_patch}")
        continue()
    endif()

if(WIN32)
    execute_process(
		COMMAND "${S2_PATCH_EXECUTABLE}" apply --check --reverse -p1 "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET
        ERROR_QUIET
    )
else()
    execute_process(
		COMMAND "${S2_PATCH_EXECUTABLE}" --batch --dry-run -R -p1 -i "${_patch}"
        WORKING_DIRECTORY "${S2_PATCH_ROOT}"
        RESULT_VARIABLE _already_applied
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()

    if(_already_applied EQUAL 0)
        message(STATUS "Local patch already applied: ${_patch}")
    else()
        message(FATAL_ERROR "Local patch is not applicable: ${_patch}")
    endif()
endforeach()
