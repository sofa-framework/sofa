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
#pragma once
#include <SofaSimpleFem/config.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::forcefield
{
template<class DataTypes>
class TriangleFEMUtils
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord;
    typedef typename DataTypes::Deriv    Deriv;
    typedef typename Coord::value_type   Real;

    typedef type::Vec<6, Real> Displacement;					    ///< the displacement vector
    typedef type::Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef type::Mat<3, 3, Real > Transformation;						///< matrix for rigid transformations like rotations
    typedef type::Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix

    typedef sofa::core::topology::BaseMeshTopology::Index Index;

    ////////////// small displacements method
    void computeDisplacementSmall(Displacement& D, const type::fixed_array<Coord, 3>& rotatedInitCoord, const Coord& pAB, const Coord& pAC);
    void applyStiffnessSmall(VecCoord& f, Real h, const VecCoord& x, const SReal& kFactor) {}

    ////////////// large displacements method
    void computeDisplacementLarge(Displacement& D, const Transformation& R_0_2, const type::fixed_array<Coord, 3>& rotatedInitCoord,const Coord& pA, const Coord& pB, const Coord& pC);
    
    void computeRotationLarge(Transformation& r, const Coord& pA, const Coord& pB, const Coord& pC);
    void applyStiffnessLarge(VecCoord& f, Real h, const VecCoord& x, const SReal& kFactor) {}
    
    // in global coordinate
    void computeStrainDisplacementGlobal(StrainDisplacement& J, SReal& area, const Coord& pA, const Coord& pB, const Coord& pC);
    // in local coordinate, a = Coord (0, 0, 0)
    void computeStrainDisplacementLocal(StrainDisplacement& J, SReal& area, const Coord& pB, const Coord& pC);
  
    // Compute strain, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    void computeStrain(type::Vec<3, Real>& strain, const StrainDisplacement& J, const Displacement& D, bool fullMethod = false);
    // Compute stress, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    void computeStress(type::Vec<3, Real>& stress, const MaterialStiffness& K, const type::Vec<3, Real>& strain, bool fullMethod = false);
};

} //namespace sofa::component::forcefield
