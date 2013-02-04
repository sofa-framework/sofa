--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                            ____  _   _  ____
                           (_  _)( )_( )( ___)
                             )(   ) _ (  )__)
                            (__) (_) (_)(____)
    ___  ____  __  __  ___     __    ____  ____  ____    __    ____  _  _
   / __)(_  _)(  \/  )/ __)   (  )  (_  _)(  _ \(  _ \  /__\  (  _ \( \/ )
  ( (__  _)(_  )    (( (_-.    )(__  _)(_  ) _ < )   / /(__)\  )   / \  /
   \___)(____)(_/\/\_)\___/   (____)(____)(____/(_)\_)(__)(__)(_)\_) (__)


                    C++ Template Image Processing Toolkit

                       ( http://cimg.sourceforge.net )

                                   1.5.4

--------------------------------------------------------------------------------

# Summary
#---------

  The CImg Library is an open-source C++ toolkit for image processing.
  It consists in a single header file 'CImg.h' providing a minimal set of C++
  classes and methods that can be used in your own sources, to load/save,
  process and display images. Very portable (Unix/X11,Windows, MacOS X, FreeBSD, .. ),
  efficient, easy to use, it's a pleasant library for developping image processing
  algorithms in C++.

# Authors and contributors :
#----------------------------

  - David Tschumperle (project leader) ( http://tschumperle.users.greyc.fr/ )

  - Antonio Albiol
  - Haz-Edine Assemlal
  - Vincent Barra
  - Romain Blei
  - Yohan Bentolila
  - Jerome Boulanger
  - Pierre Buyssens
  - Sebastien Coudert
  - Frederic Devernay
  - Francois-Xavier Dupe
  - Gerd von Egidy
  - Eric Fausett
  - Jean-Marie Favreau
  - Sebastien Fourey
  - Alexandre Fournier
  - Hon-Kwok Fung
  - Vincent Garcia
  - David Grimbichler
  - Jinwei Gu
  - Jean-Daniel Guyot
  - Matt Hanson
  - Sebastien Hanel
  - Michael Holroyd
  - Christoph Hormann
  - Werner Jainek
  - Daniel Kondermann
  - Pierre Kornprobst
  - Orges Leka
  - Francois Lauze
  - Xie Long
  - Thomas Martin
  - Cesar Martinez
  - Jean Martinot
  - Arnold Meijster (Center for High Performance Computing and Visualization, University of Groningen/The Netherlands)
  - Nikita Melnichenko
  - Julien Morat
  - Baptiste Mougel
  - Jovana Milutinovich
  - Guillaume Nee
  - Francisco Oliveira
  - Andrea Onofri
  - Renaud Peteri
  - Martin Petricek
  - Paolo Prete
  - Adrien Reboisson
  - Klaus Schneider
  - Jakob Schluttig
  - Veronique Souchaud
  - Konstantin Spirin
  - David G. Starkweather
  - Rainer Steffens
  - Grzegorz Szwoch
  - Thierry Thomas
  - Yu-En-Yun
  - Vo Duc Khanh
  - Phillip Wood
  - Bug Zhao
  - Haibo Zheng

# Institution
#-------------

 GREYC Image / CNRS UMR 6072 / FRANCE

 The CImg Library project started in 2000, at the INRIA-Sophia
 Antipolis/France ( http://www-sop.inria.fr/ ), in the ROBOTVIS / ODYSSEE Team.
 Since October 2004, it is maintained and developed in the Image team of
 the GREYC Lab (CNRS, UMR 6072), in Caen/France.
 Team web page : http://www.greyc.ensicaen.fr/EquipeImage/

# Licenses
#----------

 The source code of the CImg Library is distributed under
 two distinct licenses :

 - The main library file 'CImg.h' is *dual-licensed* :
   It can be either distributed under the CeCILL-C or CeCILL license.
   (see files 'Licence_CeCILL-C_V1-en.txt' and 'Licence_CeCILL_V2-en.txt').
   Both are Free-Software licenses :

     * CeCILL-C is adapted to the distribution of
       library components, and is close in its terms to the well known GNU LGPL license
       (the 'CImg.h' file can thus be used in closed-source products under certain
       conditions, please read carefully the license file).

     * CeCILL is close to (and even compatible with) the GNU GPL license.

 - Most of the other files are distributed under the CeCiLL license
   (file 'Licence_CeCILL_V2-en.txt'). See each file header to see what license applies.

 These two CeCiLL licenses ( http://www.cecill.info/index.en.html ) have been
 created under the supervision of the three biggest research institutions on
 computer sciences in France :

   - CNRS  ( http://www.cnrs.fr/ )
   - CEA   ( http://www.cea.fr/ )
   - INRIA ( http://www.inria.fr/ )

 You have to RESPECT these licenses. More particularly, please carefully read
 the license terms before using the CImg library in commercial products.

# Package structure :
#--------------------

  The main package directory CImg/ is organized as follows :

  - README.txt                 : This file.
  - Licence_CeCILL-C_V1-en.txt : A copy of the CeCiLL-C license file.
  - Licence_CeCILL_V2-en.txt   : A copy of the CeCiLL license.
  - CImg.h                     : The single header file that constitutes the library itself.
  - examples/                  : A directory containing a lot of example programs performing
                                 various things, using the CImg library.
  - html/                      : A directory containing a copy of the CImg web page in html
                                 format. The reference documentation is generated
              		         automatically with the tool 'doxygen' (http://www.doxygen.org).
  - resources/                 : A directory containing some resources files for compiling
                                 CImg examples or packages with various C++ compilers and OS.
  - plugins/                   : A directory containing CImg plug-ins files that can be used to
                                 add specific extra functionalities to the CImg library.

# Getting started
#-----------------

  If you are new to CImg, you should first try to compile the different examples
  provided in the 'examples/' directory, to see what CImg is capable of
  (as CImg is a template-based library, no prior compilation of the library is mandatory).
  Look at the 'resources/' directory to ease this compilation on different plateforms.

  Then, you can look at the documentation 'html/reference/' to learn more about CImg
  functions and classes. Finally, you can participate to the 'Forum' section
  of the CImg web page and ask for help if needed.

# Current list of available CImg plug-ins
#-----------------------------------------

 --------------------------------------------------------------------------------
  - VTK legacy format ('plugins/vtk.h') (April 2011).

    This plug-in allows to save 3d scenes as VTK files.

    by Haz-Edine Assemlal (http://www.cim.mcgill.ca/~assemlal/)

 --------------------------------------------------------------------------------
  - CImg IPL 2nd edition ('plugins/cimg_ipl.h') (September 2009).

    This plug-in allows the conversion between CImg and IplImage structures
    (used in openCV).

    by Hon-Kwok Fung (oldfung - at - graduate.hku.hk)

    PS : This plug-in seems to correct some problems with the first edition,
    when image pixels have a padding offset. Need to be tested before removing
    the old one !

  --------------------------------------------------------------------------------
  - CImg IPL 1st edition ('plugins/cimgIPL.h') (November 2008).

    This plug-in allows the conversion between CImg and IplImage structures
    (used in openCV).

    by Haibo Zheng (haibo.zheng - at - gmail.com)

  --------------------------------------------------------------------------------
  - Draw gradient ('plugins/draw_gradient.h') (November 2008).

    This plug-in can be used to draw color gradient in images.

    by Jerome Boulanger (http://www.irisa.fr/vista/Equipe/People/Jerome.Boulanger.html),

  --------------------------------------------------------------------------------
  - Add file format ('plugins/add_fileformat.h') (September 2007).

    This plug-in shows how to easily add support for your own file format in
    CImg. This can be interesting, since the additional format will be recognized
    in functions 'CImg<T>::save()' and 'CImg<T>::load()' functions.

    by David Tschumperle (http://tschumperle.users.greyc.fr/).
       IMAGE Team / GREYC (CNRS UMR 6072), Caen / FRANCE.
       Home page of the team :  http://www.greyc.ensicaen.fr/EquipeImage/

  --------------------------------------------------------------------------------
  - JPEG Buffer ('plugins/jpeg_buffer.h') (July 2007).

    This plug-in provides functions to read/write images stored in jpeg format
    directly in memory buffers. Interesting when dealing for instance with
    images coming from webcams and stored in memory.

    by Paolo Prete.

  --------------------------------------------------------------------------------
  - NL Means ('plugins/nlmeans.h') (May 2006).

    Implementation of the Non-Local Means algorithm as described in [1] and [2].
    The variance of the noise can be automatically estimated using the method
    inspired from [3].

    [1] Buades, A.; Coll, B.; Morel, J.-M.: A non-local algorithm for image
        denoising. IEEE Computer Society Conference on Computer Vision and Pattern
        Recognition, 2005. CVPR 2005. Vol 2,  20-25 June 2005 Page(s):60 - 65

    [2] Buades, A. Coll, B. and Morel, J.: A review of image denoising algorithms,
        with a new one. Multiscale Modeling and Simulation: A SIAM
        Interdisciplinary Journal 4 (2004) 490-530

    [3] Gasser, T. Sroka,L. Jennen Steinmetz,C. Residual variance and residual
        pattern nonlinear regression. Biometrika 73 (1986) 625-659

    by Jerome Boulanger (http://www.irisa.fr/vista/Equipe/People/Jerome.Boulanger.html),
       Charles Kervrann and Patrick Bouthemy thanks to ACI IMPBio (MODYNCELL5D Project).
       VISTA / IRISA-INRIA, Rennes / FRANCE
       Home page of the team :  http://www.irisa.fr/vista/
       MIA / INRA, Unite de Jouy-en-Josas / FRANCE.

  --------------------------------------------------------------------------------
  - Plug in for Matlab mex files ('plugins/cimgmatlab.h') (May 2006).

    Implement a CImg<T> constructor from a matlab array, a CImg<T> assignment
    operator from a matlab array and a method that exports a CImg<T> object to
    a Matlab array.
    For a bit more, http://www.itu.dk/people/francois/cimgmatlab.html

    by Francois Lauze (http://www.itu.dk/people/francois/index.html)
       The IT University of Copenhagen, Image Group.

  --------------------------------------------------------------------------------

# End of file
#------------
