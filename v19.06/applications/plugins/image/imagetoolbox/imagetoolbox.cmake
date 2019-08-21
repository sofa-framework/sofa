
# these header files need MOCcing
list(APPEND MOC_HEADER_FILES

        ../imagetoolbox/imagetoolboxbasicactionwidget.h
        ../imagetoolbox/imagetoolboxcentralwidget.h
        ../imagetoolbox/imagetoolboxwidget.h
        ../imagetoolbox/imagetoolboxlabelactionwidget.h
        ../imagetoolbox/labelimagetoolboxaction.h
        ../imagetoolbox/labelpoint/labelpointimagetoolboxaction.h
        ../imagetoolbox/contour/contourimagetoolboxaction.h
        ../imagetoolbox/depth/depthimagetoolboxaction.h
        ../imagetoolbox/labelbox/labelboximagetoolboxaction.h
        ../imagetoolbox/labelgrid/labelgridimagetoolboxaction.h
        ../imagetoolbox/labelpointsbysection/labelpointsbysectionimagetoolboxaction.h
        ../imagetoolbox/labelpointsbysection/tablewidget.h
        ../imagetoolbox/zonegenerator/distancezoneimagetoolboxaction.h
        ../imagetoolbox/zonegenerator/zonegeneratorimagetoolboxaction.h

        )


# these header files does not need MOCcing
list(APPEND HEADER_FILES

        ../imagetoolbox/imagetoolboxdata.h
        ../imagetoolbox/depth/depthimagetoolbox.h
        ../imagetoolbox/labelgrid/labelgridimagetoolbox.h
        ../imagetoolbox/labelimagetoolbox.h
        ../imagetoolbox/labelpoint/labelpointimagetoolbox.h
        ../imagetoolbox/contour/contourimagetoolbox.h
        ../imagetoolbox/depth/averagecatchallvector.h
        ../imagetoolbox/depth/catchallvector.h
        ../imagetoolbox/depth/mergedcatchallvector.h
        ../imagetoolbox/depth/meshData.h
        ../imagetoolbox/labelbox/labelboximagetoolbox.h
        ../imagetoolbox/labelpointsbysection/labelpointsbysectionimagetoolbox.h
        ../imagetoolbox/zonegenerator/distancezoneimagetoolbox.h
        ../imagetoolbox/zonegenerator/zonegeneratorimagetoolbox.h

        )

list(APPEND SOURCE_FILES

        ../imagetoolbox/cursorinfoimagetoolbox.cpp
        ../imagetoolbox/imagetoolbox.cpp
        ../imagetoolbox/imagetoolboxwidget.cpp
        ../imagetoolbox/labelimagetoolboxaction.cpp
        ../imagetoolbox/labelimagetoolbox.cpp
        ../imagetoolbox/contour/contourimagetoolboxaction.cpp
        ../imagetoolbox/contour/contourimagetoolbox.cpp
        ../imagetoolbox/labelpoint/labelpointimagetoolboxaction.cpp
        ../imagetoolbox/labelpoint/labelpointimagetoolbox.cpp
        ../imagetoolbox/depth/averagecatchallvector.cpp
        ../imagetoolbox/depth/catchallvector.cpp
        ../imagetoolbox/depth/depthimagetoolbox.cpp
        ../imagetoolbox/depth/depthimagetoolboxaction.cpp
        ../imagetoolbox/depth/mergedcatchallvector.cpp

        ../imagetoolbox/labelbox/labelboximagetoolbox.cpp
        ../imagetoolbox/labelbox/labelboximagetoolboxaction.cpp

        ../imagetoolbox/labelgrid/labelgridimagetoolbox.cpp
        ../imagetoolbox/labelgrid/labelgridimagetoolboxaction.cpp

        ../imagetoolbox/labelpointsbysection/labelpointsbysectionimagetoolbox.cpp
        ../imagetoolbox/labelpointsbysection/labelpointsbysectionimagetoolboxaction.cpp

        ../imagetoolbox/zonegenerator/distancezoneimagetoolbox.cpp
        ../imagetoolbox/zonegenerator/distancezoneimagetoolboxaction.cpp
        ../imagetoolbox/zonegenerator/zonegeneratorimagetoolbox.cpp
        ../imagetoolbox/zonegenerator/zonegeneratorimagetoolboxaction.cpp
        )
