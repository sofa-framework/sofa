Principe des applications dans CGoGN_Apps (english version below)



Faire un repertoire CGoGN_Apps ou on veut. Y mettre des sous répertoires
contenant chacun une application/

Le CMakeLists.txt a la racine de l'application doit commencer par

cmake_minimum_required(VERSION 2.8)
PROJECT(NOM_DE_L_APPLICATION)
SET(CGoGN_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../CGoGN CACHE STRING "CGoGN root dir")
INCLUDE(${CGoGN_ROOT_DIR}/apps_cmake.txt)

Le ../../CGoGN correspond au chemin relatif vers votre version de CGoGN
(ici placée au même niveau que CGoGN_Apps)


Si vous voulez faire des sous répertoires release et debug:

add_subdirectory(${CMAKE_SOURCE_DIR}/Release Release)
IF (NOT WIN32)
	add_subdirectory(${CMAKE_SOURCE_DIR}/Debug Debug)
ENDIF (NOT WIN32)


Le CMakeLists (Release par exemple) doit ensuite contenir:

SET(CMAKE_BUILD_TYPE Release)

IF(WIN32)
	link_directories( ${CGoGN_ROOT_DIR}/lib/$(ConfigurationName) )
ELSE (WIN32)
	link_directories( ${CGoGN_ROOT_DIR}/lib/Release/ )
ENDIF (WIN32)

# Attention pour la version debug, deux solutions:
#   soit mettre Debug et Release si on veut utiliser les ThirdParty en release
#   soit mettre uniquement Debug, mais il faut avoir compiler les ThirdParty en debug

include_directories(
	${CGoGN_ROOT_DIR}/include		# for CGoGN
	${COMMON_INCLUDES}				# for thirdparty & system
	${CMAKE_CURRENT_SOURCE_DIR}		# for application
	${CMAKE_CURRENT_BINARY_DIR}		# for qt (ui,moc,etc.) if necessary
)

# Pour une appli QT
QT4_WRAP_UI( source_ui ../source.ui )
QT4_WRAP_CPP( source_moc ../source.h )

add_executable( mon_exec
	${CMAKE_SOURCE_DIR}/mon_source.cpp
# Pour une appli QT
	${source_moc}
	${source_ui}
)

target_link_libraries( mon_exec ${CGoGN_LIBS_R} ${COMMON_LIBS})

Pour le compiler le plus propre est de faire un repertoire build,
d'aller à l'interieur et de faire ccmake .. (j'ai bien ecrit "..")
ccmake (ou un equivalent type cmake-gui) permet de mettre à jour les variables,
ici WITH_QT, WITH_ASSIMP, WITH_NUMERICAL, WITH_ZINRI.
Ceci ajoutera automatiquement tout ce qu'il faut aux variables COMMON_INCLUDES
et COMMON_LIBS.
On peut aussi utiliser cmake avec l'option -D (moins pratique).



=========================================================================================
                                ENGLISH VERSION
=========================================================================================

Create a directory CGoGN_Apps where you want. Inside create subdirectories, one for each application.

The CMakeLists.txt to the root of the application must begin with

cmake_minimum_required(VERSION 2.8)
PROJECT(NOM_DE_L_APPLICATION)
SET(CGoGN_ROOT_DIR ${CMAKE_SOURCE_DIR}/../../CGoGN CACHE STRING "CGoGN root dir")
INCLUDE(${CGoGN_ROOT_DIR}/apps_cmake.txt)


The .. / .. / CGoGN corresponds to the relative path to your version of CGoGN
(here placed at the same level as CGoGN_Apps)


If you want to do in release and debug directories:

add_subdirectory(${CMAKE_SOURCE_DIR}/Release Release)
IF (NOT WIN32)
	add_subdirectory(${CMAKE_SOURCE_DIR}/Debug Debug)
ENDIF (NOT WIN32)


The CMakeLists (here in Release) must then contain:

SET(CMAKE_BUILD_TYPE Release)

IF(WIN32)
	link_directories( ${CGoGN_ROOT_DIR}/lib/$(ConfigurationName) )
ELSE (WIN32)
	link_directories( ${CGoGN_ROOT_DIR}/lib/Release/ )
ENDIF (WIN32)

# Watch out for the debug version, two solutions:
# Debug and Release to be if you want to use ThirdParty in release version
# or Debug only, but you have to compile ThirdParty in debug

include_directories(
	${CGoGN_ROOT_DIR}/include		# for CGoGN
	${COMMON_INCLUDES}				# for thirdparty & system
	${CMAKE_CURRENT_SOURCE_DIR}		# for application
	${CMAKE_CURRENT_BINARY_DIR}		# for qt (ui,moc,etc.) if necessary
)

# for a QT application
QT4_WRAP_UI( source_ui ../source.ui )
QT4_WRAP_CPP( source_moc ../source.h )

add_executable( mon_exec
	${CMAKE_SOURCE_DIR}/mon_source.cpp
# for a QT application
	${source_moc}
	${source_ui}
)

target_link_libraries( mon_exec ${CGoGN_LIBS_R} ${COMMON_LIBS})




To compile the cleanest way is to build a directory, to go inside and make ccmake .. (I have written "..")
ccmake (or an equivalent type cmake-gui) is used to update the variables, WITH_QT here, WITH_ASSIMP,
WITH_NUMERICAL, WITH_ZINRI.
This will automatically add it all the variables COMMON_INCLUDES and COMMON_LIBS.
You can also use cmake with -D option (less convenient).






