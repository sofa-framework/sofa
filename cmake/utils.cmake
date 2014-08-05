
# This file contains miscellaneous small utils macros

# Intersect two lists
macro(sofa_list_intersection outList inList0 inList1)
    set(${outList})
    set(tmpList)

    foreach(inElem1 ${${inList1}})
        foreach(inElem0 ${${inList0}})
            if(inElem0 STREQUAL ${inElem1})
                set(tmpList ${tmpList} ${inElem0})
            endif()
        endforeach()
    endforeach()

    set(${outList} ${tmpList})
endmacro()

# Substract two lists
macro(sofa_list_subtraction outList inList0 inList1)
    set(${outList})
    set(tmpList)

    foreach(inElem0 ${${inList0}})
        set(add 1)
        foreach(inElem1 ${${inList1}})
            if(inElem0 STREQUAL ${inElem1})
                set(add 0)
                break()
            endif()
        endforeach()
        if(add EQUAL 1)
            set(tmpList ${tmpList} ${inElem0})
        endif()
    endforeach()

    set(${outList} ${tmpList})
endmacro()

# Remove duplicates in list if non-empty
macro(sofa_remove_duplicates list)
    if(${list})
        list(REMOVE_DUPLICATES ${list})
    endif()
endmacro()

# Set 'var' to TRUE if 'value' appears in the remaining arguments, otherwise unset 'var'
macro(sofa_list_contains var value)
  set(${var})
  foreach (value2 ${ARGN})
    if (${value} STREQUAL ${value2})
      set(${var} TRUE)
    endif()
  endforeach()
endmacro()

# Wrapper macro to set boolean value to a variable
macro(setSofaOption name value)
    set(${name} "${value}" CACHE BOOL "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaPath name value)
    set(${name} "${value}" CACHE PATH "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaString name value)
    set(${name} "${value}" CACHE STRING "" FORCE)
    message("${name} ${value}")
endmacro()

macro(setSofaFilePath name value)
    set(${name} "${value}" CACHE FILEPATH "" FORCE)
    message("${name} ${value}")
endmacro()
