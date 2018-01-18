#ifndef __DIFFUSIONSOLVER_H__
#define __DIFFUSIONSOLVER_H__

#include <CImgPlugin/SOFACImg.h>

#ifdef WIN32
#	define EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#   define IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#   ifdef _MSC_VER
#       pragma warning(disable : 4231)
#       pragma warning(disable : 4910)
#   endif
#else
#	define EXPORT_DYNAMIC_LIBRARY
#   define IMPORT_DYNAMIC_LIBRARY
#endif


/** Solving diffusion on 3D regular grids (stored as CImg)
 * @param size (in) the size of the regular domain in the 3 directions.
 * @param img (in-out) the image to diffuse. It contains the Dirichlet boundary values and contains warm-start values (must be set to 0 for no warm-start).
 * @param mask (in) represents the type of each voxel. A negative value represents the exterior of the domain, a positive value for the interior and 0 for Dirichlet boundary conditions. The boundary exterior/interior implicitely represents a Neumann boundary condition imposing null gradients along normals.
 *
 * @warning the material map must be normalized between [0,1]
 * @warning at least a one pixel outside border
 *
 * @author: matthieu.nesme@inria.fr
 */
template < typename _Real >
struct DiffusionSolver
{
    /// the scalar type
    typedef _Real Real;

    /// the regular grid type
    typedef cimg_library::CImg<Real> ImageType;

    /// the voxel type mask
    typedef cimg_library::CImg<char> MaskType;


    /// voxel type
    static const char OUTSIDE;
    static const char INSIDE;
    static const char DIRICHLET;



    /// for multi-threaded implementations
    static void setNbThreads( unsigned nb );
    static void setDefaultNbThreads();
    static void setMaxNbThreads();
    static int getMaxNbThreads();




    /// Gauss-Seidel implementation
    /// @param sor (Successive Over Relaxation coef)  0<sor<2 should converge, 1<sor<2 can converge faster than pure GS (sor=1)
    static void solveGS( ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, Real sor=1, const ImageType* material=NULL, Real minValueThreshold=0 );

    /// Jacobi implementation
    static void solveJacobi( ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, const ImageType* material=NULL, Real minValueThreshold=0 );

    /// Conjugate Gradient implementation (matrix-free)
    static void solveCG( ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, const ImageType* material=NULL );

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(__DIFFUSIONSOLVER_CPP)
    extern template struct IMPORT_DYNAMIC_LIBRARY DiffusionSolver<float>;
#endif


#endif // __DIFFUSIONSOLVER_H__
