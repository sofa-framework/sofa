## DEPRECATED!

## ###########################################################################################
## CMAKE_DOCUMENTATION_START findSofaDefines
## DEPRECATED!
## FIND SOFA DEFINES (QMAKE) AND ADD DEFINITIONS (CMAKE).
## \\li Create a copy of the Sofa.pro in Sofa-temp.pro.
## \\li Add Qmake system command line at the end of the Sofa.pro
##    in order to write all sofa defines in a sofaDefines.cfg file.
## \\li Execute qmake (or qmake-qt4) where sofa is located to execute my appended code.
## \\li Get sofa defines variables from sofaDefines.cfg to add cmake definitions.
## \\li Restore the original Sofa.pro from Sofa-temp.pro.
## \\li Delete the unnecessary Sofa-temp.pro and sofaDefines.cfg files.
## 
## CMAKE_DOCUMENTATION_END
## ###########################################################################################

## DEPRECATED!
if(_FIND_SOFA_DEFINES_INCLUDED_)
  return()
endif()
set(_FIND_SOFA_DEFINES_INCLUDED_ true)

function(FindSofaDefines VERBOSE_SOFA)

    ## check we can run this function
    if(NOT SOFA_FOUND AND NOT QT_QMAKE_EXECUTABLE)
        message("WARNING: in findSofaDefines.cmake, findSofaDefines function can't be run:")
        message("SOFA_FOUND = ${SOFA_FOUND}")
        message("QT_QMAKE_EXECUTABLE = ${QT_QMAKE_EXECUTABLE}")
        return()
    endif()


    ## Save the original file in a temp file
    configure_file(${SOFA_DIR}/Sofa.pro ${SOFA_DIR}/Sofa-temp.pro COPYONLY)
    if(VERBOSE_SOFA)
        message(STATUS "Create a Sofa-temp.pro copy in ${SOFA_DIR}")
    endif()


    ## Insert command line in this file to write all sofa defines in a file
    file(APPEND ${SOFA_DIR}/Sofa.pro
        "# print all SOFA DEFINES into a standard file format"\n
        "message(\"Write temporarily the sofa DEFINES in sofaDefines.cfg to let CMake get them, and then delete this file...\")"\n
        "win32 { system( for %G in ($\${DEFINES}) do echo %G>>sofaDefines.cfg ) }"\n
        "unix  { system( for define in $\${DEFINES}; do echo $define>>sofaDefines.cfg; done ) }"
        )
    if(VERBOSE_SOFA)
        message(STATUS "Customize the ${SOFA_DIR}/Sofa.pro")
    endif()


    ## Force to run qmake to execute my custom code I have inserted above
    execute_process(COMMAND ${QT_QMAKE_EXECUTABLE} Sofa.pro WORKING_DIRECTORY ${SOFA_DIR} RESULT_VARIABLE rv OUTPUT_VARIABLE ov ERROR_VARIABLE er )
    if( rv EQUAL "0" AND EXISTS ${SOFA_DIR}/sofaDefines.cfg)

        if(VERBOSE_SOFA)
            message(STATUS "\nTHE SOFA CONFIGURATION IN ${SOFA_DIR} IS :\n${ov}\n${er}")
            message(STATUS "Let qmake create ${SOFA_DIR}/sofaDefines.cfg")
        endif()

        ## Get Sofa defines file in order to add CMake definitions and then delete the file
        file(READ ${SOFA_DIR}/sofaDefines.cfg SOFA_DEFINES_CONFIG_FILE_CONTENTS)
        string(REGEX REPLACE "\r?\n" ";" SOFA_DEFINES_CONFIG_FILE_CONTENTS "${SOFA_DEFINES_CONFIG_FILE_CONTENTS}") # get a list of lines
        if(VERBOSE_SOFA)
            message(STATUS "Get Sofa defines in ${SOFA_DIR}/sofaDefines.cfg : ")
        endif()

        ## Add "-D" for compatibility with CMake
        foreach( define ${SOFA_DEFINES_CONFIG_FILE_CONTENTS})
            list(APPEND SOFA_DEFINES "-D${define}")
        endforeach( define ${SOFA_DEFINES_CONFIG_FILE_CONTENTS})

        ## HAAAAAAAA! Add_definitions
        add_definitions(${SOFA_DEFINES})

        if(VERBOSE_SOFA)
        message(STATUS "Add SOFA definition into CMake project : ")
            foreach( define ${SOFA_DEFINES})
                message(STATUS "add DEFINITION : ${define}")
            endforeach( define ${SOFA_DEFINES})
        endif(VERBOSE_SOFA)

    else()
        message("WARNING : in findSofaDefines.cmake : Sofa defines will not add to the CMake project.")
    endif()


    # Remove sofaDefines
    if(EXISTS ${SOFA_DIR}/sofaDefines.cfg)
        if(VERBOSE_SOFA)
            message(STATUS "Delete unnecessary ${SOFA_DIR}/sofaDefines.cfg file.")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove sofaDefines.cfg WORKING_DIRECTORY ${SOFA_DIR} )
    endif()

    ## Restore the original file
    if(EXISTS ${SOFA_DIR}/Sofa-temp.pro)
        if(VERBOSE_SOFA)
            message(STATUS "Restore the original ${SOFA_DIR}/Sofa.pro file from ${SOFA_DIR}/Sofa-temp.pro")
        endif()
        configure_file(${SOFA_DIR}/Sofa-temp.pro ${SOFA_DIR}/Sofa.pro COPYONLY)
    else()
        message(WARNING "Problem in FindSofaDefines. We can't restore the original Sofa.pro file from Sofa-temp.pro")
    endif()

    # Remove the temp file
    if(EXISTS ${SOFA_DIR}/Sofa-temp.pro)
        if(VERBOSE_SOFA)
            message(STATUS "Delete unnecessary ${SOFA_DIR}/Sofa-temp.pro file")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove Sofa-temp.pro WORKING_DIRECTORY ${SOFA_DIR} )
    endif()

endfunction()
