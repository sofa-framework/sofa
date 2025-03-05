/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

/************************************************************************************************************************************
*                                                                                                                                   *
*                          About this GPU implementation of the Total Lagrangian Explicit Dynamics                                  *
*                                                    (TLED) algorithm                                                               *
*                                                                                                                                   *
*************************************************************************************************************************************
                                                                                                                                    *
This work was carried out by Olivier Comas and Zeike Taylor and funded by INRIA and CSIRO.                                          *
This implementation was optimised for graphics cards Geforce 8800GTX                                                                *
                                                                                                                                    *
 - Preventative Health Flagship, CSIRO ICT, AEHRC, Brisbane, Australia                                                              *
   http://aehrc.com/biomedical_imaging/surgical_simulation.html                                                                     *
 - INRIA, Shaman team                                                                                                               *
   http://www.inria.fr/recherche/equipes/shaman.en.html                                                                             *
                                                                                                                                    *
 (1) For more details about the CUDA implementation of TLED into SOFA                                                               *
@InProceedings{Comas2008,                                                                                                           *
Author = {Comas, O. and Taylor, Z. and Allard, J. and Ourselin, S. and Cotin, S. and Passenger, J.},                                *
Title = {Efficient nonlinear FEM for soft tissue modelling and its GPU implementation within the open source framework SOFA},       *
Booktitle = {In Proceedings of ISBMS 2008},                                                                                         *
Year = {2008},                                                                                                                      *
Month = {July 7-8},                                                                                                                 *
Address = {London, United Kingdom}                                                                                                  *
}                                                                                                                                   *
                                                                                                                                    *
(2) For more details about the models implemented by the TLED algorithm and its validation                                          *
@article{Taylor2009,                                                                                                                *
Author = {Taylor, Z.A. and Comas, O. and Cheng, M. and Passenger, J. and Hawkes, D.J. and Atkinson, D. and Ourselin, S.},           *
Journal = {Medical Image Analysis},                                                                                                 *
Month = {April},                                                                                                                    *
Number = {2},                                                                                                                       *
Pages = {234-244},                                                                                                                  *
Title = {On modelling of anisotropic viscoelasticity for soft tissue simulation: Numerical solution and {GPU} execution},           *
Volume = {13},                                                                                                                      *
Year = {2009}                                                                                                                       *
}                                                                                                                                   *
                                                                                                                                    *
************************************************************************************************************************************/

#ifndef SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H
#define SOFA_CUDA_CUDA_HEXAHEDRON_TLED_FORCEFIELD_H

#include <vector_types.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>


namespace sofa::gpu::cuda
{

using namespace sofa::defaulttype;

class CudaHexahedronTLEDForceField : public core::behavior::ForceField<CudaVec3fTypes>
{
public:
    SOFA_CLASS(CudaHexahedronTLEDForceField,SOFA_TEMPLATE(core::behavior::ForceField,CudaVec3fTypes));
    typedef CudaVec3fTypes::Real Real;
    typedef CudaVec3fTypes::Coord Coord;
    typedef component::topology::container::constant::MeshTopology::Hexa Element;
    typedef component::topology::container::constant::MeshTopology::SeqHexahedra VecElement;

    int nbVertex;                           // number of vertices
    int nbElems;                            // number of elements
    int nbElementPerVertex;                 // max number of elements connected to a vertex

    // Material properties
    Data<Real> poissonRatio; ///< Poisson ratio in Hooke's law
    Data<Real> youngModulus; ///< Young modulus in Hooke's law
    float Lambda, Mu;                       // Lame coefficients

    // TLED configuration
    Data<Real> timestep; ///< Simulation timestep
    Data<unsigned int> isViscoelastic; ///< Viscoelasticity flag
    Data<unsigned int> isAnisotropic; ///< Anisotropy flag
    Data<Vec3f> preferredDirection; ///< Transverse isotropy direction

    CudaHexahedronTLEDForceField();
    virtual ~CudaHexahedronTLEDForceField();
    void init() override;
    void reinit() override;
//    void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& dataF, const DataVecCoord& dataX, const DataVecDeriv& /*dataV*/ ) override;
//    void addDForce (VecDeriv& /*df*/, const VecDeriv& /*dx*/);
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& datadF, const DataVecDeriv& datadX ) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix*) final {}
    void buildDampingMatrix(core::behavior::DampingMatrix*) final {}
    SReal getPotentialEnergy(const sofa::core::MechanicalParams* , const DataVecCoord&) const override { return 0.0; }

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

    // Store the 8 node indices per element. Since the type is int4, two consecutive elements are
    // needed to access the 8 indices.
    int4* m_device_nodesPerElement { nullptr };

    float4* m_device_DhC0 { nullptr };
    float4* m_device_DhC1 { nullptr };
    float4* m_device_DhC2 { nullptr };

    float* m_device_detJ { nullptr };

    float* m_device_hourglassControl { nullptr };

    float3* m_device_preferredDirection { nullptr };

    // Rate-dependant stress (isochoric part)
    float4* m_device_Di1 { nullptr };
    float4* m_device_Di2 { nullptr };

    // Rate-dependant stress (volumetric part)
    float4* m_device_Dv1 { nullptr };
    float4* m_device_Dv2 { nullptr };

    int2* m_device_forceCoordinates { nullptr };

    float4* m_device_F0 { nullptr };
    float4* m_device_F1 { nullptr };
    float4* m_device_F2 { nullptr };
    float4* m_device_F3 { nullptr };
    float4* m_device_F4 { nullptr };
    float4* m_device_F5 { nullptr };
    float4* m_device_F6 { nullptr };
    float4* m_device_F7 { nullptr };
};

} // namespace sofa::gpu::cuda


#endif
