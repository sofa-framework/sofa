find_package(Qt6 COMPONENTS Core REQUIRED)
get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)

include(windeployqt)
