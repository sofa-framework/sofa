/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v16.08                  *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H
#define SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/MeshTopology.h>


namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;

class CudaHexahedronTLEDForceField : public core::behavior::ForceField<CudaVec3fTypes>
{
public:
    SOFA_CLASS(CudaHexahedronTLEDForceField,SOFA_TEMPLATE(core::behavior::ForceField,CudaVec3fTypes));
    typedef CudaVec3fTypes::Real Real;
    typedef CudaVec3fTypes::Coord Coord;
    typedef component::topology::MeshTopology::Hexa Element;
    typedef component::topology::MeshTopology::SeqHexahedra VecElement;

    int nbVertex;                           // number of vertices
    int nbElems;                            // number of elements
    int nbElementPerVertex;                 // max number of elements connected to a vertex

    // Material properties
    Data<Real> poissonRatio;
    Data<Real> youngModulus;
    float Lambda, Mu;                       // Lame coefficients

    // TLED configuration
    Data<Real> timestep;                    // time step of the simulation
    Data<unsigned int> isViscoelastic;      // flag = 1 to enable viscoelasticity
    Data<unsigned int> isAnisotropic;       // flag = 1 to enable transverse isotropy
    Data<Vec3f> preferredDirection;         // uniform preferred direction for transverse isotropy

    CudaHexahedronTLEDForceField();
    virtual ~CudaHexahedronTLEDForceField();
    void init();
    void reinit();
//    void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& dataF, const DataVecCoord& dataX, const DataVecDeriv& /*dataV*/ ) ;
//    void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/);
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& datadF, const DataVecDeriv& datadX ) ;
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* , const DataVecCoord&) const { return 0.0; }

    // Computes lambda and mu based on Young's modulus and Poisson ratio
    void updateLameCoefficients();

    /// Computes Jacobian determinants
    float ComputeDetJ(const Element& e, const VecCoord& x, float DhDr[8][3]);
    /// Computes element volumes for hexahedral elements
    float CompElVolHexa(const Element& e, const VecCoord& x);
    void ComputeCIJK(float C[8][8][8]);
    /// Computes shape function global derivatives for hexahedral elements
    void ComputeDhDxHexa(const Element& e, const VecCoord& x, float Vol, float DhDx[8][3]);
    /// Computes matrices used for Hourglass control into hexahedral elements
    void ComputeBmat(const Element& e, const VecCoord& x, float B[8][3]);
    void ComputeHGParams(const Element& e, const VecCoord& x, float DhDx[8][3], float volume, float HG[8][8]);

protected:

};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
