#define DIFFUSIONSOLVER_CPP

#include "DiffusionSolver.h"
#include <assert.h>
#include <iostream>

#include <vector>

#ifdef USING_OMP_PRAGMAS
#include <omp.h>
#endif



template < typename Real >
void DiffusionSolver< Real >::setNbThreads( unsigned nb )
{
#ifdef USING_OMP_PRAGMAS
    omp_set_num_threads( std::min( nb, (unsigned)omp_get_num_procs() ) );
#endif
}

template < typename Real >
void DiffusionSolver< Real >::setDefaultNbThreads()
{
#ifdef USING_OMP_PRAGMAS
    omp_set_num_threads( omp_get_num_procs() / 2 );
#endif
}

template < typename Real >
void DiffusionSolver< Real >::setMaxNbThreads()
{
#ifdef USING_OMP_PRAGMAS
    omp_set_num_threads( omp_get_num_procs() );
#endif
}


template < typename Real >
int DiffusionSolver< Real >::getMaxNbThreads()
{
#ifdef USING_OMP_PRAGMAS
    return omp_get_max_threads();
#else
    return 1;
#endif
}






// limitation = consider at least a one pixel outside border
// and do not check for image boundaries



/// Low-level functor to compute the new value of a voxel depending on its neighbours for homogeneous material on a regular domain
template<class Real, class ImageType, class MaskType>
struct Uniform
{
    typedef DiffusionSolver<Real> DiffusionSolverReal;
    inline static Real value( unsigned long off, const ImageType& img, const MaskType& mask, size_t lineSize, size_t sliceSize, Real, Real, Real, const ImageType* =NULL )
    {
        const Real*v=&img[off];
        const char*m=&mask[off];
        Real res = 0;
        unsigned nb = 0;
        if( *(m-1) != DiffusionSolverReal::OUTSIDE ) { res += *(v-1); ++nb; }
        if( *(m+1) != DiffusionSolverReal::OUTSIDE ) { res += *(v+1); ++nb; }
        if( *(m-lineSize) != DiffusionSolverReal::OUTSIDE ) { res += *(v-lineSize); ++nb; }
        if( *(m+lineSize) != DiffusionSolverReal::OUTSIDE ) { res += *(v+lineSize); ++nb; }
        if( *(m-sliceSize) != DiffusionSolverReal::OUTSIDE ) { res += *(v-sliceSize); ++nb; }
        if( *(m+sliceSize) != DiffusionSolverReal::OUTSIDE ) { res += *(v+sliceSize); ++nb; }
        if( !nb ) return *v;
        return res / (Real)nb;
    }

    inline static Real cgvalue( unsigned long off, const ImageType& x, const MaskType& mask, size_t lineSize, size_t sliceSize, Real, Real, Real, const ImageType* =NULL )
    {
        Real r = 0;

        unsigned int nb = 6;

        char m = mask[off-1]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off-1]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;
             m = mask[off+1]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off+1]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;
             m = mask[off-lineSize]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off-lineSize]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;
             m = mask[off+lineSize]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off+lineSize]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;
             m = mask[off-sliceSize]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off-sliceSize]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;
             m = mask[off+sliceSize]; if( m==DiffusionSolverReal::INSIDE ) r -= x[off+sliceSize]; else if (m==DiffusionSolverReal::OUTSIDE) --nb;

        return r + nb * x[off];
    }

    inline static Real cgrhs( unsigned long off, const ImageType& img, const MaskType& mask, size_t lineSize, size_t sliceSize, Real, Real, Real, const ImageType* =NULL )
    {
        Real res = 0;
        if( mask[off]==DiffusionSolverReal::INSIDE )
        {
            if( mask[off-1]==DiffusionSolverReal::DIRICHLET ) res += img[off-1];
            if( mask[off+1]==DiffusionSolverReal::DIRICHLET ) res += img[off+1];
            if( mask[off-lineSize]==DiffusionSolverReal::DIRICHLET ) res += img[off-lineSize];
            if( mask[off+lineSize]==DiffusionSolverReal::DIRICHLET ) res += img[off+lineSize];
            if( mask[off-sliceSize]==DiffusionSolverReal::DIRICHLET ) res += img[off-sliceSize];
            if( mask[off+sliceSize]==DiffusionSolverReal::DIRICHLET ) res += img[off+sliceSize];
        }
        return res;
    }

};



/// Low-level functor to compute the new value of a voxel depending on its neighbours for heterogeneous material or/and on rectangular domain
template<class Real, class ImageType, class MaskType, class Coef>
struct NonUniform
{
    typedef DiffusionSolver<Real> DiffusionSolverReal;
    inline static Real value( unsigned long off, const ImageType& img, const MaskType& mask, size_t lineSize, size_t sliceSize, Real hx2, Real hy2, Real hz2, const ImageType* material )
    {
        const Real*v=&img[off];
        const char*m=&mask[off];
        Real res = 0;
        Real nb = 0;
        Real coef;
        if( *(m-1) != DiffusionSolverReal::OUTSIDE ) {         coef = Coef::getCoef( material, off, off-1, hx2 );         res += coef * *(v-1);         nb+=coef; }
        if( *(m+1) != DiffusionSolverReal::OUTSIDE ) {         coef = Coef::getCoef( material, off, off+1, hx2 );         res += coef * *(v+1);         nb+=coef; }
        if( *(m-lineSize) != DiffusionSolverReal::OUTSIDE ) {  coef = Coef::getCoef( material, off, off-lineSize, hy2 );  res += coef * *(v-lineSize);  nb+=coef; }
        if( *(m+lineSize) != DiffusionSolverReal::OUTSIDE ) {  coef = Coef::getCoef( material, off, off+lineSize, hy2 );  res += coef * *(v+lineSize);  nb+=coef; }
        if( *(m-sliceSize) != DiffusionSolverReal::OUTSIDE ) { coef = Coef::getCoef( material, off, off-sliceSize, hz2 ); res += coef * *(v-sliceSize); nb+=coef; }
        if( *(m+sliceSize) != DiffusionSolverReal::OUTSIDE ) { coef = Coef::getCoef( material, off, off+sliceSize, hz2 ); res += coef * *(v+sliceSize); nb+=coef; }
        if( !nb ) return *v;
        return res / nb;
    }



    inline static Real cgvalue( unsigned long off, const ImageType& x, const MaskType& mask, size_t lineSize, size_t sliceSize, Real hx2, Real hy2, Real hz2, const ImageType* material )
    {
        Real r = 0;

        Real nb = 0;

        Real coef;

        char m = mask[off-1];           coef = Coef::getCoef( material, off, off-1, hx2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off-1]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;
             m = mask[off+1];           coef = Coef::getCoef( material, off, off+1, hx2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off+1]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;
             m = mask[off-lineSize];    coef = Coef::getCoef( material, off, off-lineSize, hy2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off-lineSize]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;
             m = mask[off+lineSize];    coef = Coef::getCoef( material, off, off+lineSize, hy2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off+lineSize]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;
             m = mask[off-sliceSize];   coef = Coef::getCoef( material, off, off-sliceSize, hz2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off-sliceSize]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;
             m = mask[off+sliceSize];   coef = Coef::getCoef( material, off, off+sliceSize, hz2 ); if( m==DiffusionSolverReal::INSIDE ) r -= coef*x[off+sliceSize]; if (m!=DiffusionSolverReal::OUTSIDE) nb+=coef;

        return r + nb * x[off];
    }


    inline static Real cgrhs( unsigned long off, const ImageType& img, const MaskType& mask, size_t lineSize, size_t sliceSize, Real hx2, Real hy2, Real hz2, const ImageType* material )
    {
        Real res = 0;
        if( mask[off]==DiffusionSolverReal::INSIDE )
        {
            if( mask[off-1]==DiffusionSolverReal::DIRICHLET )         { res += Coef::getCoef( material, off, off-1, hx2 ) * img[off-1]; }
            if( mask[off+1]==DiffusionSolverReal::DIRICHLET )         { res += Coef::getCoef( material, off, off+1, hx2 ) * img[off+1]; }
            if( mask[off-lineSize]==DiffusionSolverReal::DIRICHLET )  { res += Coef::getCoef( material, off, off-lineSize, hy2 ) * img[off-lineSize]; }
            if( mask[off+lineSize]==DiffusionSolverReal::DIRICHLET )  { res += Coef::getCoef( material, off, off+lineSize, hy2 ) * img[off+lineSize]; }
            if( mask[off-sliceSize]==DiffusionSolverReal::DIRICHLET ) { res += Coef::getCoef( material, off, off-sliceSize, hz2 ) * img[off-sliceSize]; }
            if( mask[off+sliceSize]==DiffusionSolverReal::DIRICHLET ) { res += Coef::getCoef( material, off, off+sliceSize, hz2 ) * img[off+sliceSize]; }
        }
        return res;
    }

};


template<class Real, class ImageType>
struct MaterialCoef
{
    inline static Real getCoef( const ImageType* material, unsigned long off, unsigned long offneighbour, Real spacing ) { return getMaterialCoef( (*material)[off], (*material)[offneighbour] ) * spacing; }

private:

    inline static Real getMaterialCoef( Real current, Real neighbour ) { return (current+neighbour)*.5; }
};


template<class Real>
struct RectangularCoef
{
    inline static Real getCoef( const void*, unsigned long, unsigned long, Real spacing ) { return spacing; }
};







/// Low-level functor to set the new value
template<class Real>
struct GS
{
    inline static void set( Real& o, Real& n, Real, Real=1 ) { o = n; }
};

/// Low-level functor to set the value with SOR
template<class Real>
struct SOR
{
    inline static void set( Real& o, Real& n, Real p, Real w ) { o = (1.0-w)*p + w*n; }
};

/// Multithreaded Gauss-Seidel
template < class Real, class ImageType, class MaskType, typename Value, typename Set >
void genericColoredGSImpl(ImageType& img, const MaskType& mask, unsigned iterations, Real threshold, Real sor, Real hx2, Real hy2, Real hz2, const ImageType *material, Real minValueThreshold )
{
    assert( img.width() == mask.width() );
    assert( img.height() == mask.height() );
    assert( img.depth() == mask.depth() );
    assert( img.spectrum() == mask.spectrum() );
    assert( img.spectrum() == 1 );

    typedef DiffusionSolver<Real> DiffusionSolverReal;

    const size_t lineSize = img.width();
    const size_t sliceSize = lineSize * img.height();


    std::vector<unsigned long> F; F.reserve(img.size()/2);
    std::vector<unsigned long> T; T.reserve(img.size()/2);

    cimg_forXYZ( img, x, y, z )
    {
        bool color;
        if( z%2 )
            if( y%2 )
                if( x%2 ) color = true; else color = false;
            else
                if( x%2 ) color = false; else color = true;
        else
            if( y%2 )
                if( x%2 ) color = false; else color = true;
            else
                if( x%2 ) color = true; else color = false;

        if( color ) T.push_back( x+y*lineSize+z*sliceSize );
        else F.push_back( x+y*lineSize+z*sliceSize );
    }

    std::vector<unsigned long> *OK = &F, *KO = &T;


    bool change = true;
    unsigned it=0;
    Real average;



    for(  ; change && it<iterations ; ++it )
    {
        change = false;

        // TODO find a way to only loop over good colors
#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for shared(it,OK,img,mask,material,minValueThreshold,sor,change) private(average)
#endif
#ifdef WIN32
        for( long int i = 0 ; i<OK->size() ; ++i )
#else
        for( size_t i = 0 ; i<OK->size() ; ++i )
#endif
        {
            const unsigned long& off = (*OK)[i];

            char m = mask[off];

            if( m == DiffusionSolverReal::OUTSIDE || m == DiffusionSolverReal::DIRICHLET )
            {
                continue;
            }
            else
            {
                // limitation = consider at least a one pixel outside border
                // and do not check for image boundaries

                Real& v = img[off];

                average = Value::value( off, img, mask, lineSize, sliceSize, hx2, hy2, hz2, material );

                if( std::fabs(average) < minValueThreshold ) average = (Real)0;

                if( !change && std::fabs(average-v)>threshold ) change = true;

                Set::set( v, average, v, sor );
            }
        }

        std::swap( KO, OK );
    }

    std::cerr<<"DiffusionSolver::solveGaussSeidel "<<it<<" iterations"<<std::endl;
}


/// Naive Gauss-Seidel
template < class Real, class ImageType, class MaskType, typename Value, typename Set >
void genericGSImpl(ImageType& img, const MaskType& mask, unsigned iterations, Real threshold, Real sor, Real hx2, Real hy2, Real hz2, const ImageType *material, Real minValueThreshold )
{
    assert( img.width() == mask.width() );
    assert( img.height() == mask.height() );
    assert( img.depth() == mask.depth() );
    assert( img.spectrum() == mask.spectrum() );
    assert( img.spectrum() == 1 );

    typedef DiffusionSolver<Real> DiffusionSolverReal;


    bool change = true;
    unsigned it=0;
    Real average;

    const size_t lineSize = img.width();
    const size_t sliceSize = lineSize * img.height();

    for(  ; change && it<iterations ; ++it )
    {
        change = false;

        cimg_foroff(img,off)
        {
            char m = mask[off];

            if( m == DiffusionSolverReal::OUTSIDE || m == DiffusionSolverReal::DIRICHLET )
            {
                continue;
            }
            else
            {
                // limitation = consider at least a one pixel outside border
                // and do not check for image boundaries

                Real& v = img[off];

                average = Value::value( off, img, mask, lineSize, sliceSize, hx2, hy2, hz2, material );

                if( std::fabs(average) < minValueThreshold ) average = (Real)0;

                if( !change && std::fabs(average-v)>threshold ) change = true;

                Set::set( v, average, v, sor );
            }
        }
    }

    std::cerr<<"DiffusionSolver::solveGaussSeidel "<<it<<" iterations"<<std::endl;
}


/// Multithreaded Jacobi
template < class Real, class ImageType, class MaskType, typename Value, typename Set >
void genericJacobiImpl(ImageType& img, const MaskType& mask, unsigned iterations, Real threshold, Real hx2, Real hy2, Real hz2, const ImageType *material, Real minValueThreshold )
{
    assert( img.width() == mask.width() );
    assert( img.height() == mask.height() );
    assert( img.depth() == mask.depth() );
    assert( img.spectrum() == mask.spectrum() );
    assert( img.spectrum() == 1 );

    typedef DiffusionSolver<Real> DiffusionSolverReal;


    ImageType tmp(img);

    ImageType* previous = &img;
    ImageType* current = &tmp;


    const size_t lineSize = img.width();
    const size_t sliceSize = lineSize * img.height();



    bool change = true;
    unsigned it=0;
    Real average;

    for(  ; change && it<iterations ; ++it )
    {
        change = false;

#ifdef USING_OMP_PRAGMAS
        #pragma omp parallel for shared(it,previous,current,mask,material,minValueThreshold,change) private(average)
#endif
#ifdef WIN32
        for( long int off = 0 ; off<img.size() ; ++off  )
#else
        for( unsigned long off = 0 ; off<img.size() ; ++off )
#endif
        {
            char m = mask[off];

            if( m == DiffusionSolverReal::OUTSIDE || m == DiffusionSolverReal::DIRICHLET )
            {
                continue;
            }
            else
            {
                // limitation = consider at least a one pixel outside border
                // and do not check for image boundaries

                average = Value::value( off, *previous, mask, lineSize, sliceSize, hx2, hy2, hz2, material );

                if( std::fabs(average) < minValueThreshold ) average = (Real)0;

                const Real& p = (*previous)[off];

                if( !change && std::fabs( average-p )>threshold ) change = true;

                Set::set( (*current)[off], average, p );
            }
        }
        std::swap( current, previous );
    }

    std::cerr<<"DiffusionSolver::solveJacobi "<<it<<" iterations"<<std::endl;
}



template < typename Real >
void DiffusionSolver< Real >::solveGS(ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, Real sor, const ImageType *material, Real minValueThreshold )
{
    if( spacingX!=spacingY || spacingY!=spacingZ )
    {
        if( getMaxNbThreads() == 1 )
        {
            if( sor == 1 )
            {
                if( !material ) genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, RectangularCoef<Real> >, GS<Real> >( img,mask,iterations,threshold,1,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
                else genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,1,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
            }
            else
            {
                if( !material ) genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, RectangularCoef<Real> >, SOR<Real> >( img,mask,iterations,threshold,sor,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
                else genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, SOR<Real> >( img,mask,iterations,threshold,sor,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
            }
        }
        else
        {
            if( sor == 1 )
            {
                if( !material ) genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, RectangularCoef<Real> >, GS<Real> >( img,mask,iterations,threshold,1,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
                else genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,1,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
            }
            else
            {
                if( !material ) genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, RectangularCoef<Real> >, SOR<Real> >( img,mask,iterations,threshold,sor,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
                else genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType>  >, SOR<Real> >( img,mask,iterations,threshold,sor,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
            }
        }
    }
    else
    {
        if( getMaxNbThreads() == 1 )
        {
            if( sor == 1 )
            {
                if( !material ) genericGSImpl< Real, ImageType, MaskType, Uniform<Real, ImageType, MaskType>, GS<Real> >( img,mask,iterations,threshold,1,1,1,1,material,minValueThreshold);
                else genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,1,1,1,1,material,minValueThreshold);
            }
            else
            {
                if( !material ) genericGSImpl< Real, ImageType, MaskType, Uniform<Real, ImageType, MaskType>, SOR<Real> >( img,mask,iterations,threshold,sor,1,1,1,material,minValueThreshold);
                else genericGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, SOR<Real> >( img,mask,iterations,threshold,sor,1,1,1,material,minValueThreshold);
            }
        }
        else
        {
            if( sor == 1 )
            {
                if( !material ) genericColoredGSImpl< Real, ImageType, MaskType, Uniform<Real, ImageType, MaskType>, GS<Real> >( img,mask,iterations,threshold,1,1,1,1,material,minValueThreshold);
                else genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,1,1,1,1,material,minValueThreshold);
            }
            else
            {
                if( !material ) genericColoredGSImpl< Real, ImageType, MaskType, Uniform<Real, ImageType, MaskType>, SOR<Real> >( img,mask,iterations,threshold,sor,1,1,1,material,minValueThreshold);
                else genericColoredGSImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType>  >, SOR<Real> >( img,mask,iterations,threshold,sor,1,1,1,material,minValueThreshold);
            }
        }
    }
}


template < typename Real >
void DiffusionSolver< Real >::solveJacobi(ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, const ImageType *material, Real minValueThreshold )
{
    if( spacingX!=spacingY || spacingY!=spacingZ )
    {
        if( !material ) genericJacobiImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, RectangularCoef<Real> >, GS<Real> >( img,mask,iterations,threshold,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
        else genericJacobiImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ,material,minValueThreshold);
    }
    else
    {
        if( !material ) genericJacobiImpl< Real, ImageType, MaskType, Uniform<Real, ImageType, MaskType>, GS<Real> >( img,mask,iterations,threshold,1,1,1,material,minValueThreshold);
        else genericJacobiImpl< Real, ImageType, MaskType, NonUniform<Real, ImageType, MaskType, MaterialCoef<Real,ImageType> >, GS<Real> >( img,mask,iterations,threshold,1,1,1,material,minValueThreshold);
    }
}



/// @internal non assembled matrix multiplication
template < class Real, class ImageType, class MaskType, class Value >
void matrixmult(ImageType& res, const ImageType& x, const MaskType& mask, size_t lineSize, size_t sliceSize, Real spacingX, Real spacingY, Real spacingZ, const ImageType* material)
{
    typedef DiffusionSolver<Real> DiffusionSolverReal;

#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,x,mask,material,lineSize,sliceSize)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<x.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<x.size() ; ++off )
#endif
    {
        if( mask[off] == DiffusionSolverReal::INSIDE )
        {
            res[off] = Value::cgvalue( off, x, mask, lineSize, sliceSize,spacingX,spacingY,spacingZ, material );
        }
        else res[off] = 0;
    }
}

/// @internal non assembled dot product
template < typename Real, typename ImageType >
Real img_dot( const ImageType& i, const ImageType& j )
{
//    return i.dot(j);
    Real d = 0;

#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(i,j) reduction(+:d)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<i.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<i.size() ; ++off )
#endif
        d += i[off]*j[off];
    return d;
}

template < typename ImageType >
void img_eq( ImageType& res, const ImageType& in )
{
//    res = in;
#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,in)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<res.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<res.size() ; ++off )
#endif
        res[off] = in[off];
}

template < typename ImageType >
void img_peq( ImageType& res, const ImageType& in )
{
//    res += in;
#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,in)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<res.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<res.size() ; ++off )
#endif
        res[off] += in[off];
}

template < typename Real, typename ImageType >
void img_peq( ImageType& res, const ImageType& in, Real a )
{
//    res += a*in;
#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,in,a)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<res.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<res.size() ; ++off )
#endif
        res[off] += a*in[off];
}

template < typename Real, typename ImageType >
void img_meq( ImageType& res, const ImageType& in, Real a )
{
//    res -= a*in;
#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,in,a)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<res.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<res.size() ; ++off )
#endif
        res[off] -= a*in[off];
}

template < typename Real, typename ImageType >
void img_teq( ImageType& res, Real a )
{
//    res *= a;
#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(res,a)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<res.size() ; ++off )
#else
    for( unsigned long off = 0 ; off<res.size() ; ++off )
#endif
        res[off] *= a;
}



/// Multi-threaded Conjugate Gradient
template < class Real, class ImageType, class MaskType, typename Value >
void genericCGImpl(ImageType& img, const MaskType& mask, unsigned iterations, Real threshold, Real spacingX, Real spacingY, Real spacingZ, const ImageType *material )
{

    const size_t lineSize = img.width();
    const size_t sliceSize = lineSize * img.height();

    threshold *= threshold; // compare square norms

    ImageType r( img.width(), img.height(), img.depth(), img.spectrum() );

    // r = A * img
    matrixmult<Real,ImageType,MaskType,Value>( r, img, mask, lineSize, sliceSize, spacingX, spacingY, spacingZ, material );
    // r = - A * img
    img_teq(r, -1);
    // r = b - A * img

#ifdef USING_OMP_PRAGMAS
    #pragma omp parallel for shared(r,mask,img)
#endif
#ifdef WIN32
    for( long int off = 0 ; off<img.size() ; ++off  )
#else
    for( unsigned long off = 0 ; off<img.size() ; ++off )
#endif
    {
        r[off] += Value::cgrhs( off, img, mask, lineSize, sliceSize,spacingX,spacingY,spacingZ, material );
    }


//    ImageType p(r);
    ImageType p( img.width(), img.height(), img.depth(), img.spectrum() );
    img_eq( p, r );

    Real rnorm, rnormold = img_dot<Real,ImageType>(r,r);


    Real alpha;

    ImageType Ap( img.width(), img.height(), img.depth(), img.spectrum() );

    unsigned it=0;
    for(  ; it<iterations ; ++it )
    {
//        std::cerr<<"CG norm: "<<rnormold<<std::endl;

        matrixmult<Real,ImageType,MaskType,Value>( Ap, p, mask, lineSize, sliceSize, spacingX, spacingY, spacingZ, material ); // Ap = A * p

//        alpha = img_dot<Real,ImageType>(r,p) / img_dot<Real,ImageType>(p,Ap);
        alpha = rnormold / img_dot<Real,ImageType>(p,Ap);

        img_peq( img, p, alpha ); // img += alpha * p
        img_meq( r, Ap, alpha ); // r -= alpha * Ap
        rnorm = img_dot<Real,ImageType>(r,r);
        if( /*std::sqrt*/(rnorm) < threshold ) break;

        // p = r + beta*p
        img_teq( p, rnorm/rnormold );
        img_peq( p, r );

        rnormold = rnorm;
    }

    std::cerr<<"DiffusionSolver::solveCG "<<it<<" iterations"<<std::endl;
}



template < typename Real >
void DiffusionSolver< Real >::solveCG( ImageType& img, const MaskType& mask, Real spacingX, Real spacingY, Real spacingZ, unsigned iterations, Real threshold, const ImageType* material )
{
    if( spacingX!=spacingY || spacingY!=spacingZ )
    {
        if( material )
            genericCGImpl< Real,ImageType,MaskType,NonUniform<Real,ImageType,MaskType, MaterialCoef<Real,ImageType> > >( img, mask, iterations, threshold, spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ, material );
        else
            genericCGImpl< Real,ImageType,MaskType,NonUniform<Real,ImageType,MaskType, RectangularCoef<Real> > >( img, mask, iterations, threshold, spacingX*spacingX,spacingY*spacingY,spacingZ*spacingZ, material );
    }
    else
    {
        if( material )
            genericCGImpl< Real,ImageType,MaskType,NonUniform<Real,ImageType,MaskType, MaterialCoef<Real,ImageType> > >( img, mask, iterations, threshold, 1, 1, 1, material );
        else
            genericCGImpl< Real,ImageType,MaskType,Uniform<Real,ImageType,MaskType> >( img, mask, iterations, threshold, 1, 1, 1, material );
    }
}





// precompilation for single and double floating points
template class DIFFUSION_SOLVER_DYNAMIC_LIBRARY DiffusionSolver<float>;
//template class DIFFUSION_SOLVER_DYNAMIC_LIBRARY DiffusionSolver<double>;

