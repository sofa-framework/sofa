/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_H
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRALFEMFORCEFIELDANDMASS_H

#include <sofa/component/forcefield/HexahedralFEMForceFieldAndMass.h>

namespace sofa
{

namespace component
{

namespace topology
{
class MultilevelHexahedronSetTopologyContainer;
}

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;

/**

@InProceedings{NPF06,
author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
title        = "Animating Shapes at Arbitrary Resolution with Non-Uniform Stiffness",
booktitle    = "Eurographics Workshop in Virtual Reality Interaction and Physical Simulation (VRIPHYS)",
month        = "nov",
year         = "2006",
organization = "Eurographics",
address      = "Madrid",
url          = "http://www-evasion.imag.fr/Publications/2006/NPF06"
}

indices ordering (same as in HexahedronSetTopology):

     Y  7---------6
     ^ /         /|
     |/    Z    / |
     3----^----2  |
     |   /     |  |
     |  4------|--5
     | /       | /
     |/        |/
     0---------1-->X

*/


template<class DataTypes>
class NonUniformHexahedralFEMForceFieldAndMass : virtual public HexahedralFEMForceFieldAndMass<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef HexahedralFEMForceFieldAndMass<DataTypes> HexahedralFEMForceFieldAndMassT;
    typedef HexahedralFEMForceField<DataTypes> HexahedralFEMForceFieldT;

    typedef typename HexahedralFEMForceFieldAndMassT::VecElement VecElement;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedralFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedralFEMForceFieldAndMassT::ElementMass ElementMass;
    typedef typename HexahedralFEMForceFieldAndMassT::Element Element;

    typedef typename defaulttype::Mat<8, 8, Real> Mat88;
    typedef typename defaulttype::Vec<3, int> Vec3i;

public:

    using HexahedralFEMForceFieldAndMass<DataTypes>::serr;
    using HexahedralFEMForceFieldAndMass<DataTypes>::sout;
    using HexahedralFEMForceFieldAndMass<DataTypes>::sendl;

    NonUniformHexahedralFEMForceFieldAndMass();

    virtual void init();
    virtual void reinit();

    // handle topological changes
    virtual void handleTopologyChange(core::componentmodel::topology::Topology*);

protected:
    /// condensate matrice from finest level to the actual mechanical level
    virtual void computeMechanicalMatricesByCondensation( ElementStiffness &K,
            ElementMass &M,
            Real& totalMass,
            const int elementIndex);

    void initLarge(const int i);

    void initPolar(const int i);

private:
    void computeHtfineH(const Mat88& H, const ElementStiffness& fine, ElementStiffness& HtfineH ) const;
    void addHtfineHtoCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;
    void subtractHtfineHfromCoarse(const Mat88& H, const ElementStiffness& fine, ElementStiffness& coarse ) const;

    void computeMechanicalMatricesByCondensation_Recursive( ElementStiffness &K,
            ElementMass &M,
            Real& totalMass,
            const ElementStiffness &K_fine,
            const ElementMass &M_fine,
            const Real& mass_fine,
            const int level,
            const helper::vector<bool>& fineChildren) const;

    // [childId][childNodeId][parentNodeId] -> weight
    helper::fixed_array<Mat88, 8> _H; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)

    helper::vector < Mat88 > __H; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)

    typedef struct
    {
        MaterialStiffness	C;
        ElementStiffness	K;
        ElementMass			M;
        Real				mass;
    } Material;

    Material _material; // TODO: enable combination of multiple materials

    MultilevelHexahedronSetTopologyContainer*	_multilevelTopology;
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
