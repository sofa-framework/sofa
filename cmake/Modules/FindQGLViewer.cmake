# Find the QGLViewer headers and libraries
# Behavior is to first look for config files.
# If no config files were found, tries to find
# the library by looking at headers / lib file.
#
# Defines:
#   QGLViewer_FOUND : True if QGLViewer is found
#
# Provides target QGLViewer.

find_package(QGLViewer NO_MODULE QUIET)

if(NOT TARGET QGLViewer)

  if(NOT QGLViewer_INCLUDE_DIR)
    find_path(QGLViewer_INCLUDE_DIR
      NAMES QGLViewer/qglviewer.h
      PATH_SUFFIXES include Headers
    )
  endif()

  if(NOT QGLViewer_LIBRARY)
  find_library(QGLViewer_LIBRARY
    NAMES QGLViewer QGLViewer2 QGLViewer-qt5 QGLViewer-qt6
    PATH_SUFFIXES lib
  )
  endif()

  if(QGLViewer_INCLUDE_DIR AND QGLViewer_LIBRARY)
    set(QGLViewer_FOUND TRUE)
  else()
    if(QGLViewer_FIND_REQUIRED)
      message(FATAL_ERROR "Cannot find QGLViewer")
    endif()
  endif()

  # Same checks as Sofa.GUI.Qt
  # i.e find Qt6, then if not, Qt5, then if not error
  find_package(Qt6 COMPONENTS Core CoreTools QUIET)
  if (NOT Qt6Core_FOUND)
    find_package(Qt5 COMPONENTS Core QUIET)
  endif()

  if (Qt5Core_FOUND)
      find_package(Qt5 COMPONENTS Core Gui Xml OpenGL Widgets REQUIRED)
      set(QT_TARGETS Qt5::Core Qt5::Gui Qt5::Xml Qt5::OpenGL Qt5::Widgets)
  elseif (Qt6Core_FOUND)
      find_package(Qt6 COMPONENTS Gui GuiTools Widgets WidgetsTools OpenGLWidgets Xml REQUIRED)
      set(QT_TARGETS ${QT_TARGETS} Qt::Core Qt::Gui Qt::Widgets Qt::OpenGLWidgets Qt::Xml)
  endif()

  if(QGLViewer_FOUND)
    set(QGLViewer_LIBRARIES ${QGLViewer_LIBRARY})
    set(QGLViewer_INCLUDE_DIRS ${QGLViewer_INCLUDE_DIR})

    if(NOT QGLViewer_FIND_QUIETLY)
      message(STATUS "Found QGLViewer: ${QGLVIEWER_LIBRARIES}")
    endif(NOT QGLViewer_FIND_QUIETLY)

    if(NOT TARGET QGLViewer)
      add_library(QGLViewer INTERFACE IMPORTED)
      set_property(TARGET QGLViewer PROPERTY INTERFACE_LINK_LIBRARIES "${QGLViewer_LIBRARIES}" ${QT_TARGETS})
      set_property(TARGET QGLViewer PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${QGLViewer_INCLUDE_DIR}")
    endif()
  endif()
  mark_as_advanced(QGLViewer_INCLUDE_DIR QGLViewer_LIBRARY)
endif()
