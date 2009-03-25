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

    typedef typename helper::fixed_array<helper::fixed_array<float,8>,8 > Mat88;
    typedef typename helper::fixed_array<int,3> Vec3i;

public:

    Data<bool> _oldMethod;

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

    void computeMechanicalMatricesByCondensation( ElementStiffness &K,
            ElementMass &M,
            Real& totalMass,
            const ElementStiffness &K_fine,
            const ElementMass &M_fine,
            const Real& mass_fine,
            const int level,
            const helper::vector<bool>& fineChildren);

    void computeMechanicalMatricesByCondensationDirectlyFromFinestToCoarse( ElementStiffness &K,
            ElementMass &M,
            Real& totalMass,
            const int elementIndex);

    /// add a matrix of a fine element to its englobing coarser matrix
    void addFineToCoarse( ElementStiffness& coarse, const ElementStiffness& fine, int indice );
    void computeHtfineHAndAddFineToCoarse( ElementStiffness& HtfineH ,ElementStiffness& coarse, const ElementStiffness& fine, const Mat88& H );

    /// remove a fine hexa given by its idx (i.e. remove its M and K into its coarse embedding hexa)
    void removeFineHexa( const unsigned int fineIdx );



    void initLarge(const int i);

    void initPolar(const int i);

private:
    static const float FINE_TO_COARSE[8][8][8]; ///< interpolation matrices from finer level to a coarser (to build stiffness and mass matrices)


    typedef struct
    {
        //Mat88 interpolation;
        //int coarseHexaIdx;
        //ElementStiffness K;
        //ElementMass M;
        Real mass;
        ElementStiffness HtKH;
        ElementMass HtMH;
    } AFine;

    std::map<Vec3i,AFine > _mapFineToCorse; ///< finest hexa idx in regular grid -> coarse parent coarse hexa idx + H

    MultilevelHexahedronSetTopologyContainer*	_multilevelTopology;
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
