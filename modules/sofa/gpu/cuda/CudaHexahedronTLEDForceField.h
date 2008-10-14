#ifndef SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H
#define SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/topology/MeshTopology.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;

class CudaHexahedronTLEDForceField : public core::componentmodel::behavior::ForceField<CudaVec3fTypes>
{
public:
    typedef CudaVec3fTypes::Real Real;
    typedef CudaVec3fTypes::Coord Coord;
    typedef component::topology::MeshTopology::Hexa Element;
    typedef component::topology::MeshTopology::SeqHexas VecElement;

    /** Static data associated with each element
    */
    struct GPUElement
    {
        /// @name index of the 8 connected vertices
        /// @{
        int v[8];
        /// @}
    };

    CudaVector<GPUElement> elems;

    /// Varying data associated with each element
    struct GPUElementState
    {
        int dummy[8];
    };

    CudaVector<GPUElementState> state;

    int nbVertex; ///< number of vertices to process to compute all elements
    int nbElementPerVertex; ///< max number of elements connected to a vertex
    /// Index of elements attached to each points (layout per bloc of NBLOC vertices, with first element of each vertex, then second
    /// element, etc). Note that each integer is actually equat the the index of the element * 8 + the index of this vertex inside
    /// the element.
    CudaVector<int> velems;

    /// Material properties
    Data<Real> poissonRatio;
    Data<Real> youngModulus;
    /// Lame coefficients
    float Lambda, Mu;

    /// TLED configuration
    Data<Real> timestep;
    Data<unsigned int> viscoelasticity;
    Data<unsigned int> anisotropy;

    CudaHexahedronTLEDForceField();
    virtual ~CudaHexahedronTLEDForceField();
    void init();
    void reinit();
    void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    void addDForce (VecDeriv& df, const VecDeriv& dx);
    double getPotentialEnergy(const VecCoord&) { return 0.0; }

    /// Compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();

    /// Compute Jacobian determinants
    float ComputeDetJ(const Element& e, const VecCoord& x, float DhDr[8][3]);
    /// Compute element volumes for hexahedral elements
    float CompElVolHexa(const Element& e, const VecCoord& x);
    void ComputeCIJK(float C[8][8][8]);
    /// Compute shape function global derivatives for hexahedral elements
    void ComputeDhDxHexa(const Element& e, const VecCoord& x, float Vol, float DhDx[8][3]);
    /// Computes matrices used for Hourglass control into hexahedral elements
    void ComputeBmat(const Element& e, const VecCoord& x, float B[8][3]);
    void ComputeHGParams(const Element& e, const VecCoord& x, float DhDx[8][3], float volume, float HG[8][8]);

protected:

    void init(int nbe, int nbv, int nbelemperv)
    {
        elems.resize(nbe);
        state.resize(nbe);
        nbVertex = nbv;
        nbElementPerVertex = nbelemperv;
//         int nbloc = (nbVertex+BSIZE-1)/BSIZE;
//         velems.resize(nbloc*nbElementPerVertex*BSIZE);
//         for (unsigned int i=0; i<velems.size(); i++)
//         {
//             velems[i] = 0;
//         }
    }

//     void setV(int vertex, int num, int index)
//     {
//         int bloc = vertex/BSIZE;
//         int b_x  = vertex%BSIZE;
//         velems[ bloc*BSIZE*nbElementPerVertex // start of the bloc
//               + num*BSIZE                     // offset to the element
//               + b_x                           // offset to the vertex
//               ] = index+1;
//     }

    void setE(int i, const Element& indices)
    {
        GPUElement& e = elems[i];
        for (unsigned int j=0; j<indices.size(); j++)
        {
            e.v[j] = indices[j];
        }
    }
};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
