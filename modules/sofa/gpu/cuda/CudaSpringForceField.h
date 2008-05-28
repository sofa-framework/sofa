#ifndef SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/forcefield/MeshSpringForceField.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class DataTypes>
class CudaKernelsSpringForceField;

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <class TCoord, class TDeriv, class TReal>
class SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef SpringForceField<DataTypes> Main;
    typedef typename Main::Inherit Inherit;
    typedef typename Main::Spring Spring;
    typedef SpringForceFieldInternalData<DataTypes> Data;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef gpu::cuda::CudaKernelsSpringForceField<DataTypes> Kernels;

    //enum { BSIZE=16 };
    struct GPUSpring
    {
        int index; ///< -1 if no spring
        //float initpos;
        float ks;
        //float kd;
        GPUSpring() : index(-1), /*initpos(0),*/ ks(0)/*, kd(0)*/ {}
        void set(int index, float /*initpos*/, float ks, float /*kd*/)
        {
            this->index = index;
            //this->initpos = initpos;
            this->ks = ks;
            //this->kd = kd;
        }
    };
    struct GPUSpring2
    {
        //int index; ///< -1 if no spring
        float initpos;
        //float ks;
        float kd;
        GPUSpring2() : /*index(-1),*/ initpos(0), /*ks(0),*/ kd(0) {}
        void set(int /*index*/, float initpos, float /*ks*/, float kd)
        {
            //this->index = index;
            this->initpos = initpos;
            //this->ks = ks;
            this->kd = kd;
        }
    };
    struct GPUSpringSet
    {
        int vertex0; ///< index of the first vertex connected to a spring
        int nbVertex; ///< number of vertices to process to compute all springs
        int nbSpringPerVertex; ///< max number of springs connected to a vertex
        gpu::cuda::CudaVector<GPUSpring> springs; ///< springs attached to each points (layout per bloc of NBLOC vertices, with first spring of each vertex, then second spring, etc)
        gpu::cuda::CudaVector<Real> dfdx; ///< only used for StiffSpringForceField
        GPUSpringSet() : vertex0(0), nbVertex(0), nbSpringPerVertex(0) {}
        void init(int v0, int nbv, int nbsperv)
        {
            vertex0 = v0;
            nbVertex = nbv;
            nbSpringPerVertex = nbsperv;
            int nbloc = (nbVertex+BSIZE-1)/BSIZE;
            springs.resize(2*nbloc*nbSpringPerVertex*BSIZE);
        }
        void set(int vertex, int spring, int index, float initpos, float ks, float kd)
        {
            int bloc = vertex/BSIZE;
            int b_x  = vertex%BSIZE;
            springs[ 2*bloc*BSIZE*nbSpringPerVertex // start of the bloc
                    + 2*spring*BSIZE                 // offset to the spring
                    + b_x                          // offset to the vertex
                   ].set(index, initpos, ks, kd);
            (*(GPUSpring2*)&(springs[ 2*bloc*BSIZE*nbSpringPerVertex // start of the bloc
                    + 2*spring*BSIZE                 // offset to the spring
                    + b_x+BSIZE                    // offset to the vertex
                                    ])).set(index, initpos, ks, kd);
        }
    };
    GPUSpringSet springs1; ///< springs from model1 to model2
    GPUSpringSet springs2; ///< springs from model2 to model1 (only used if model1 != model2)

    static void init(Main* m, bool stiff);
    static void addForce(Main* m, bool stiff, VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);
    static void addDForce (Main* m, bool stiff, VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor);
};

//
// SpringForceField
//

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaSpringForceField_DeclMethods(T) \
    template<> void SpringForceField< T >::init(); \
    template<> void SpringForceField< T >::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2); \
    template<> void StiffSpringForceField< T >::init(); \
    template<> void StiffSpringForceField< T >::addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2); \
    template<> void StiffSpringForceField< T >::addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor);

CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaSpringForceField_DeclMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

#undef CudaSpringForceField_DeclMethods

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
