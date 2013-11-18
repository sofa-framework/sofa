 *** english version below ***

Compilation sous VisualExpress C++

Prérequis:
Visual Studio ou C++ Express (testé avec 2010 express)
CMake (au moins 2.8)

Unziper le windows_dependencies.zip (http://iggservis.u-strasbg.fr/Data/) a la racine de CGoGN
Installer boost et Qt (4.8.x pas de 5.0 pour le moment)

Attention installer les binaires de Qt dans une version de compilateur differente (VS2008/VS2010)
permet d'executer mais pas de debugger !
Si vous utiliser BoostPro (http://www.boostpro.com) installer toutes les versions multithread dll
pour eviter les pb de compilation.

Le plus sur mais plus long est de compiler soi-même Qt et Boost (bon courage)

Pour les "ThirdParty libs":
Lancer CMake (gui)
Mettre le chemin de la racine de CGoGN + /ThirdParty pour les sources
Mettre le chemin de la racine de CGoGN + /ThirdParty/build pour le build en release
Mettre le chemin de la racine de CGoGN + /ThirdParty/buildDebug pour le build en debug
Sous windows les 2 versions sont nécessaires.

La suite vaut pour les 2 répertoires:
Cliquer sur Configure
Choisir le bon compilateur (celui de votre version de Visual C++)
Modifier les variables en fonctions de ce que vous voulez compiler (README.TXT de ThirdParty)
Modifier le CMAKE_PREFIX_PATH si nécessaire (cf remarque dépendances)
Recliquer sur Configure
Cliquer sur Generate
Aller sous visual, charger et compiler les 2 solutions 
Ne pas oublier de générer la cible INSTALL (une seule fois)

Pour CGoGN:
Relancer CMake
Mettre le chemin de la racine de CGoGN pour les sources
Mettre le chemin de la racine de CGoGN + /build (par exemple) pour le build
Cliquer sur Configure
Choisir le bon compilateur (celui de votre version de Visual C++)
Modifier les variables comme pour ThirdParty (et CMAKE_PREFIX_PATH si nécessaire)
Recliquer sur Configure
Cliquer sur Generate

Si il ne trouve pas Qt lui remplir la variable QT_QMAKE_EXECUTABLE

Aller dans Visual et double-cliquer sur CGoGN.sln !!

ATTENTION: Problème de link avec QGLViewer en Debug.
Si vous compiler un exemple (tuto1 & viewer) en Debug, il faut modifier
le projet: Proprietés de configuration -> Editeur de liens -> entrée -> Dépendances supplémentaires
et changer le QGLViewer2.lib en QGLViewerd2.lib

Pour que les executables fonctionnent ajouter le chemin vers les dll des dépendances dans la variable
d'environnement path (ou copier les dll dans un répertoire déjà dans le path)


Remarque dépendances:
Si vous voulez utiliser vos propre versions des dépendance (Zlib / glew / Boost / Qt)
changer le CMAKE_PREFIX_PATH (voir ci dessous)
Attention pour Boost le prefix path ne marche pas (utiliser BOOST_ROOT et d'autres si nécessaire)


Compilation 64bits: TODO



=========================================================================================
                                ENGLISH VERSION
=========================================================================================

Prerequisites:
Visual Studio and C ++ Express (2010)
CMake (at least 2.8)

Unzip the windows_dependencies.zip (http://iggservis.u-strasbg.fr/Data/) is the root of CGoGN
Install boost and Qt


Warning installing the binaries of Qt in a version of different compilers (VS2008/VS2010)
allows you to run but not in debug mode!

If you use BoostPro (http://www.boostpro.com) install all versions multithreaded dll
to avoid the pb of compilation.

The safest but longest is to compile Qt yourself and Boost (good luck)


For the "ThirdParty libs"
Run CMake (gui)
Put the path to the root of CGoGN + / ThirdParty sources for
Put the path to the root of CGoGN + / ThirdParty / build to build in release
Put the path to the root of CGoGN + / ThirdParty / buildDebug to build in debug
Under windows the two versions are needed.

The following applies to two directories:
Click on Configure
Choosing the right compiler (your version of Visual C + +)
Change the variables in functions that you want to compile (README.TXT for ThirdParty)
Edit CMAKE_PREFIX_PATH if necessary (see note dependencies)
Click again on Configure
Click on Generate
Go on visual, load and compile the 2 solutions
Do not forgot to generate the target INSTALL (once) 


For CGoGN:
Restart CMake
Put the path to the root of CGoGN for sources
Put the path to the root of CGoGN + / build (for example) to build
Click on Configure
Choosing the right compiler (your version of Visual C + +)
Change the variables as ThirdParty (and CMAKE_PREFIX_PATH if necessary)
Click again on Configure
Click on Generate
Go to Visual and double-click on CGoGN.sln!

WARNING: Problem of linking with QGLViewer in Debug
If you compile an example (tuto1 & viewer) in Debug you have to modifiy
the project: Configuration properties -> Linking Editor -> Entries -> Supplementary dependencies
and change QGLViewer2.lib in QGLViewerd2.lib

For running executables you have to add the path to the dll dependencies in the variable
path environment (or copy the dll in a directory already in the path)


Note Dependencies:
If you want to use your own versions of dependency (Zlib / glew / Boost / Qt)
CMAKE_PREFIX_PATH change (see below)
Attention to Boost the prefix path does not work (use BOOST_ROOT and others if necessary)

Compiling 64-bit: TODO
