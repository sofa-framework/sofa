## ###########################################################################################
## CMAKE_DOCUMENTATION_START findSofaDefines
## 
## FIND SOFA DEFINES (QMAKE) AND ADD DEFINITIONS (CMAKE).
## \\li Create a copy of the sofa-local.prf in sofa-local-temp.prf.
## \\li Add Qmake system command line at the end of the sofa-local.prf
##    in order to write all sofa defines in a sofaDefines.cfg file.
## \\li Execute qmake (or qmake-qt4) where sofa is located to execute my appended code.
## \\li Get sofa defines variables from sofaDefines.cfg to add cmake definitions.
## \\li Restore the original sofa-local.prf from sofa-local-temp.prf.
## \\li Delete the unnecessary sofa-local-temp.prf and sofaDefines.cfg files.
## 
## CMAKE_DOCUMENTATION_END
## ###########################################################################################
function(findSofaDefines VERBOSE_SOFA)

    ## check we can run this function
    if(NOT SOFA_FOUND AND NOT QT_QMAKE_EXECUTABLE)
        message("WARNING: in findSofaDefines.cmake, findSofaDefines function can't be run:")
        message("SOFA_FOUND = ${SOFA_FOUND}")
        message("QT_QMAKE_EXECUTABLE = ${QT_QMAKE_EXECUTABLE}")
        return()
    endif()

    ## If user doesn't yet created the sofa-local.prf in SOFA_DIR, I create it for him.
    if(NOT EXISTS ${SOFA_DIR}/sofa-local.prf)
        configure_file(${SOFA_DIR}/sofa-default.prf ${SOFA_DIR}/sofa-local.prf COPYONLY)
        message("You haven't yet create a ${SOFA_DIR}/sofa-local.prf to personalise your config, it has been created for you!")
    else()
        file(READ ${SOFA_DIR}/sofa-local.prf SOFA_LOCAL_CONTENT)
        if(NOT SOFA_LOCAL_CONTENT)
            message("Your ${SOFA_DIR}/sofa-local.prf is empty. Re create it from sofa-default.prf.")
            configure_file(${SOFA_DIR}/sofa-default.prf ${SOFA_DIR}/sofa-local.prf COPYONLY)
        endif()
    endif()

    ## Save the original file in a temp file
    configure_file(${SOFA_DIR}/sofa-local.prf ${SOFA_DIR}/sofa-local-temp.prf COPYONLY)
    if(VERBOSE_SOFA)
        message(STATUS "Create a sofa-local-temp.prf copy in ${SOFA_DIR}")
    endif()

    ## Add command line in this file to write all sofa defines in a file
    file(APPEND ${SOFA_DIR}/sofa-local.prf
        "# print all SOFA DEFINES into a standard file format"\n
        "message(\"Write temporarily the sofa DEFINES in sofaDefines.cfg to let CMake get them, and then delete it...\")"\n
        "win32 { system( for %G in ($\${DEFINES}) do echo %G>>sofaDefines.cfg ) }"\n
        "unix  { system( for define in $\${DEFINES}; do echo $define>>sofaDefines.cfg; done ) }"
        )
    if(VERBOSE_SOFA)
        message(STATUS "Customize the ${SOFA_DIR}/sofa-local.prf")
    endif()


    ## Force to run qmake to execute my custom code I have inserted above
    execute_process(COMMAND ${QT_QMAKE_EXECUTABLE} WORKING_DIRECTORY ${SOFA_DIR} RESULT_VARIABLE rv OUTPUT_VARIABLE ov ERROR_VARIABLE er )
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
    if(EXISTS ${SOFA_DIR}/sofa-local-temp.prf)
        if(VERBOSE_SOFA)
            message(STATUS "Restore the original ${SOFA_DIR}/sofa-local.prf file from ${SOFA_DIR}/sofa-local-temp.prf")
        endif()
        configure_file(${SOFA_DIR}/sofa-local-temp.prf ${SOFA_DIR}/sofa-local.prf COPYONLY)
    else()
        message(WARNING "Problem in finSofaDefines.We can't restore the original sofa-local.prf file from sofa-local-temp.prf => delete sofa-local.prf")
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove sofa-local.prf WORKING_DIRECTORY ${SOFA_DIR} )
    endif()

    # Remove the temp file
    if(EXISTS ${SOFA_DIR}/sofa-local-temp.prf)
        if(VERBOSE_SOFA)
            message(STATUS "Delete unnecessary ${SOFA_DIR}/sofa-local-temp.prf file")
        endif()
        execute_process(COMMAND ${CMAKE_COMMAND} -E remove sofa-local-temp.prf WORKING_DIRECTORY ${SOFA_DIR} )
    endif()

endfunction()
