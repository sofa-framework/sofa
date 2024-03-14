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
#include <sofa/component/solidmechanics/fem/elastic/config.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::fem::elastic
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
    constexpr void computeDisplacementSmall(Displacement& D, const type::fixed_array<Coord, 3>& rotatedInitCoord, const Coord& pAB, const Coord& pAC) const;

    ////////////// large displacements method
    constexpr void computeDisplacementLarge(Displacement& D, const Transformation& R_0_2, const type::fixed_array<Coord, 3>& rotatedInitCoord,const Coord& pA, const Coord& pB, const Coord& pC) const;
    constexpr void computeRotationLarge(Transformation& r, const Coord& pA, const Coord& pB, const Coord& pC) const;
    constexpr void computeForceLarge(Displacement& F, const StrainDisplacement& J, const type::Vec<3, Real>& stress) const;
    
    // in global coordinate
    constexpr void computeStrainDisplacementGlobal(StrainDisplacement& J, const Coord& pA, const Coord& pB, const Coord& pC) const;
    // in local coordinate, a = Coord (0, 0, 0)
    constexpr void computeStrainDisplacementLocal(StrainDisplacement& J, const Coord& pB, const Coord& pC) const;

    // Compute strain, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    constexpr void computeStrain(type::Vec<3, Real>& strain, const StrainDisplacement& J, const Displacement& D, bool fullMethod = false) const;
    // Compute stress, if full is set to true, full matrix multiplication is performed not taking into account potential 0 values
    constexpr void computeStress(type::Vec<3, Real>& stress, const MaterialStiffness& K, const type::Vec<3, Real>& strain, bool fullMethod = false) const;
};

} //namespace sofa::component::solidmechanics::fem::elastic
