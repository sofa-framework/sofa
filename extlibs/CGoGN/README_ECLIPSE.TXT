 *** english version below ***

Pour utiliser Eclipse (testé avec la version Helios, Galileo, Indigo & Juno)
 * Configurer un projet dans Eclipse :
 - Faire un nouveau projet C++
 - Choisir Makefile project -> Empty Project
 - Choisir Comme "Location": la racine de CGoGN
 - Choisir Toolchains: Linux GCC

 * Compilation avec le petit marteau ;) (l'icone)
Dans les propriétés du projet: 
-> C/C++ Build:
	Dans la configuration Default | Active:
	ajouter au chemin de Build Directory /build
	
	on peut ensuite ajouter des configurations pour compiler ce que l'on veut
	en changeant juste le Build directory:
	CGoGNRelease:		build/Release		(compile la lib en release)
	CGoGNDebug:		build/Debug		(compile la lib en debug)
	Apps:			build/Apps		(compile les exemples et les tutos)
	Tuto:			build/Apps/Tuto		(compile les tutos)
	etc..

Ceci ajoute dans le repertoire de CGoGN un .project et un .cproject.

 * Dépendances inter-projets et autocomplétion
Si le projet X dépend d'un autre projet, alors :
Projet X -> Properties -> Project references : cocher le projet dont il dépend.

 * Dépendances de Qt :
Afin d'éviter qu'Eclipse ne rejette les syntaxes Qt (les includes de Qt, les Q_OBJECT, les callbacks) :
 - Télécharger le plugin Qt "eclipse-integration" http://qt.nokia.com/developer/eclipse-integration/
 - Décompresser l'archive et copier son contenu dans le répertoire d'installation d'Eclipse (plugins et features)
 - Relancer Eclipse au moins une fois avec l'option "-clean" (relecture des répertoires locaux)
Normalement, dans Window --> Preferences de Eclipse, Qt apparait dans le menu.
 - Y ajouter le répertoire de Qt (/usr/bin sur Debian/Ubuntu) et le répertoire d'include (/usr/include/qt4).
Ensuite, pour chaque projet, ajouter les includes suivants dans Properties -> C/C++ general -> Paths and Symbols
parmi les includes de C++ :
 - /usr/include/qt4
 - /usr/include/qt4/Qt
 - /usr/include/qt4/QtCore
 - /usr/include/qt4/QtGui
 - /usr/include/qt4/QtOpenGL
REMARQUE : en bas de cette fenêtre, le bouton "Export Settings" permet d'exporter en XML la configuration afin 
de l'importer dans les autres projets.
 - Pour finir, un petit clic-droit sur le projet -> Index -> Freshen all files et le tour est joué.

 * Coloration syntaxique des shaders
 - Télécharger et décompresser EclipseShaders : http://sourceforge.net/projects/glshaders/
 - Copier les répertoires plugins et features dans le répertoire d'installation eclipse
 - Relancer Eclipse au moins une fois avec l'option --clean (relecture des répertoires locaux)
Eclipse reconnaîtra désormais les fichiers .frag et .vert.
Pour associer d'autres fichiers au programme GLSLeditor :
 - Preferences -> General -> Editors -> File Associations
Dans Preferences -> Shaders Preferences, on peut désormais configurer certaines options liées au shaders.

 * Coloration syntaxique des fichiers CMake
 - Installer CMakeEditor pour Eclipse en ajoutant l'URL dans Help -> Install New Software ... : http://cmakeed.sourceforge.net/eclipse/
 - Installer et redémarrer
 - Dans Preferences -> General -> Editors -> File Associations il est possible d'associer les CMakeLists.txt à ce logiciel d'édition
 
 * Problème indexeur
 Si l'indexeur plante avec comme message "Out of memory" (c'est la cas par défaut sur Juno lors de l'indexation totale de CGoGN),
 relancer eclipse avec l'option "/opt/eclipse/eclipse -vmargs -Xms512m -Xmx1024M" avant de refaire une indexation globale.

=========================================================================================
                                ENGLISH VERSION
=========================================================================================

To use Eclipse (tested with version Helios, Galileoi, Indigo & Juno)

 * Setting up a project in Eclipse:
 - Make a new C + + project
 - Select Makefile project -> Empty Project
 - Choose As "Location": root CGoGN
 - Select Toolchain: GCC Linux

 * Compilation with the small hammer (icon)
In the project properties:
	-> C / C + + Build:
	In the configuration Default | Active:
	add to Build Directory path /build

	you can then add configurations to compile what we want by just changing the Build directory:
	CGoGNRelease: build / Release (compile the lib release)
	CGoGNDebug: build / Debug (compiles in debug lib)
	Apps: build / Apps (compile the examples and tutorials)
	Tutorial: Build / Apps / Tutorial (compiles tutorials)
	etc. ..

This adds a .project. and .cproject files to the CGoGN directory.

 * Project inter-dependencies and autocompletion
If project X depends on another project, then:
Project X -> Properties -> Project references: check the project which it depends.

 * Qt dependencies:
To avoid Eclipse rejecting the Qt syntax (includes, Q_OBJECT, callbacks):
 - Download the Qt plugin "eclipse-integration" http://qt.nokia.com/developer/eclipse-integration/
 - Unzip the archive and copy its contents into the installation directory of Eclipse (plugins and features)
 - Restart Eclipse at least once with the "-clean" (proofreading local directories)
Normally in Window --> Preferences of Eclipse, Qt appears in the menu.
 - Add the Qt directory (/ usr / bin on Debian / Ubuntu) and the directory include (/ usr/include/qt4).
Then, for each project, add the following includes in Properties -> C / C + + general -> Paths and Symbols
among the C + + includes:
 - / Usr/include/qt4
 - / Usr/include/qt4/Qt
 - / Usr/include/qt4/QtCore
 - / Usr/include/qt4/QtGui
 - / Usr/include/qt4/QtOpenGL
NOTE: at the bottom of this window, the button "Export Settings" allows you to export the configuration to XML
import it into other projects.
 - Finally, a small right-click the project -> Index -> Freshen all files and go.

 * Syntax of shaders
 - Download and unzip EclipseShaders: http://sourceforge.net/projects/glshaders/
 - Copy the directories features and plugins in the eclipse installation directory
 - Restart Eclipse at least once with the - clean (proofreading local directories)
Eclipse will now recognize the files. And frag. Green.
To assign other files in the program GLSLeditor:
 - Preferences -> General -> Editors -> File Associations
In Preferences -> Shaders Preferences, you can now configure some options related to the shader.

 * Syntax highlighting for CMake files
 - Install CMakeEditor for Eclipse by adding the URL in Help -> Install New Software ... : Http://cmakeed.sourceforge.net/eclipse/
 - Install and restart
 - In Preferences -> General -> Editors -> File Associations You can associate the CMakeLists.txt to editing software
 
 * Indexer problem
 If the indexer fails with an "Out of Memory" error (this is the case when indexing CGoGN fully on eclipse Juno for instance),
 then relaunch eclipse using some options to use a bigger heap "/opt/eclipse/eclipse -vmargs -Xms512m -Xmx1024M" before indexing
 CGoGN globally.
