/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_H
#define SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_H
#include "config.h"


#include <SofaNonUniformFem/NonUniformHexahedronFEMForceFieldAndMass.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;

/** Need a SparseGridTopology with _sparseGrid->_nbVirtualFinerLevels >= this->_nbVirtualFinerLevels

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


*/


template<class DataTypes>
class NonUniformHexahedronFEMForceFieldDensity :  public NonUniformHexahedronFEMForceFieldAndMass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(NonUniformHexahedronFEMForceFieldDensity,DataTypes), SOFA_TEMPLATE(NonUniformHexahedronFEMForceFieldAndMass,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

#ifdef SOFA_NEW_HEXA
    typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra VecElement;
#else
    typedef sofa::core::topology::BaseMeshTopology::SeqCubes VecElement;
#endif

    typedef HexahedronFEMForceFieldAndMass<DataTypes> HexahedronFEMForceFieldAndMassT;
    typedef HexahedronFEMForceField<DataTypes> HexahedronFEMForceFieldT;

    typedef typename HexahedronFEMForceFieldAndMassT::ElementStiffness ElementStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MaterialStiffness MaterialStiffness;
    typedef typename HexahedronFEMForceFieldAndMassT::MassT MassT;
    typedef typename HexahedronFEMForceFieldAndMassT::ElementMass ElementMass;

protected:

    NonUniformHexahedronFEMForceFieldDensity():NonUniformHexahedronFEMForceFieldAndMass<DataTypes>()
        ,densityFile(core::objectmodel::Base::initData(&densityFile,"densityFile","RAW File containing gray scale density"))
        ,dimensionDensityFile(core::objectmodel::Base::initData(&dimensionDensityFile, "dimensionDensityFile", "dimension of the RAW file"))
    {
    }
public:
    void init();
    void draw(const core::visual::VisualParams* vparams);
    // 	virtual void rein
    void drawSphere(double r, int lats, int longs, const Coord &pos);

protected:
    sofa::core::objectmodel::DataFileName densityFile;
    Data< Vec<3,unsigned int> > dimensionDensityFile;
    vector< vector < vector<unsigned char > > >voxels;
    // 	  vector< int > stiffnessFactor;
    void computeCoarseElementStiffness( ElementStiffness &K, ElementMass &coarseMassElement, const int elementIndice,  int level);

    void computeMaterialStiffness(MaterialStiffness &m, double youngModulus, double poissonRatio);
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedronFEMForceFieldDensity<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_NON_UNIFORM_FEM_API NonUniformHexahedronFEMForceFieldDensity<Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_NONUNIFORMHEXAHEDRONFEMFORCEFIELDDENSITY_H
