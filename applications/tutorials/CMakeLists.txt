cmake_minimum_required(VERSION 2.8.12)

set(SOFATUTORIALS_TARGETS
    # No CMakeLists.txt
    # anatomyModelling
    chainHybrid
    compositeObject
    houseOfCards
    mixedPendulum
    oneParticle
    # Needs SofaTypedefs
    # oneParticleWithSofaTypedefs
    oneTetrahedron
)

foreach(TARGET ${SOFATUTORIALS_TARGETS})
    add_subdirectory(${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaTutorials) # IDE folder
endforeach()
