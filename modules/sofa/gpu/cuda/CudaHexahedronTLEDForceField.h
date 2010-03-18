/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H
#define SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/topology/MeshTopology.h>

// Total Lagrangian Explicit Dynamics algorithm from
// @InProceedings{Comas2008ISBMS,
// author = {Comas, O. and Taylor, Z. and Allard, J. and Ourselin, S. and Cotin, S. and Passenger, J.},
// title = {Efficient nonlinear FEM for soft tissue modelling and its GPU implementation within the open source framework SOFA},
// booktitle = {In Proceedings of ISBMS 2008},
// year = {2008},
// month = {July 7-8},
// address = {London, United Kingdom}
// }

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
    SOFA_CLASS(CudaHexahedronTLEDForceField,SOFA_TEMPLATE(core::componentmodel::behavior::ForceField,CudaVec3fTypes));
    typedef CudaVec3fTypes::Real Real;
    typedef CudaVec3fTypes::Coord Coord;
    typedef component::topology::MeshTopology::Hexa Element;
    typedef component::topology::MeshTopology::SeqHexahedra VecElement;

    int nbVertex; // number of vertices
    int nbElems; //  number of elements
    int nbElementPerVertex; // max number of elements connected to a vertex

    /// Material properties
    Data<Real> poissonRatio;
    Data<Real> youngModulus;
    /// Lame coefficients
    float Lambda, Mu;

    /// TLED configuration
    Data<Real> timestep;    // time step of the simulation
    Data<unsigned int> viscoelasticity; // flag to enable viscoelasticity
    Data<unsigned int> anisotropy;      // flag to enable transverse isotropy
    Data<Vec3f> preferredDirection;     // uniform preferred direction for transverse isotropy

    CudaHexahedronTLEDForceField();
    virtual ~CudaHexahedronTLEDForceField();
    void init();
    void reinit();
    void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/);
    double getPotentialEnergy(const VecCoord&)  const { return 0.0; }

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

};

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif
