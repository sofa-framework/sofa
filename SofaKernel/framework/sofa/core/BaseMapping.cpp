/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/core/BaseMapping.h>
#include <sofa/core/objectmodel/BaseNode.h>

namespace sofa
{

namespace core
{

BaseMapping::BaseMapping()
    : f_mapForces(initData(&f_mapForces, true, "mapForces", "Are forces mapped ?"))
    , f_mapConstraints(initData(&f_mapConstraints, true, "mapConstraints", "Are constraints mapped ?"))
    , f_mapMasses(initData(&f_mapMasses, true, "mapMasses", "Are masses mapped ?"))
    , f_mapMatrices(initData(&f_mapMatrices, false, "mapMatrices", "Are matrix explicit mapped?"))
{
    this->addAlias(&f_mapForces, "isMechanical");
    this->addAlias(&f_mapMasses, "isMechanical");
}

/// Destructor
BaseMapping::~BaseMapping()
{}

//void BaseMapping::computeLocalCoordinates()
//{
//    serr<<"Mapping "<< getName() <<", BaseMapping::computeLocalCoordinates() is not implemented for this class. It may be still implemented in the init() method." << sendl;
//}


bool BaseMapping::setFrom(BaseState*  )
{
    this->serr<<"BaseMapping::setFrom is not implemented for " << this->getName()<< sendl;
    return false;
}


bool BaseMapping::setTo( BaseState*  )
{
    this->serr<<"BaseMapping::setTo is not implemented for " << this->getName()<< sendl;
    return false;
}


bool BaseMapping::areForcesMapped() const
{
    return f_mapForces.getValue();
}

bool BaseMapping::areConstraintsMapped() const
{
    return f_mapConstraints.getValue();
}

bool BaseMapping::areMassesMapped() const
{
    return f_mapMasses.getValue();
}

bool BaseMapping::areMatricesMapped() const
{
    return f_mapMatrices.getValue();
}

void BaseMapping::setForcesMapped(bool b)
{
    f_mapForces.setValue(b);
}

void BaseMapping::setConstraintsMapped(bool b)
{
    f_mapConstraints.setValue(b);
}

void BaseMapping::setMassesMapped(bool b)
{
    f_mapMasses.setValue(b);
}
void BaseMapping::setMatricesMapped(bool b)
{
    f_mapMatrices.setValue(b);
}


void BaseMapping::setNonMechanical()
{
    setForcesMapped(false);
    setConstraintsMapped(false);
    setMassesMapped(false);
}

/// Return true if this mapping should be used as a mechanical mapping.
bool BaseMapping::isMechanical() const
{
    return areForcesMapped() || areConstraintsMapped() || areMassesMapped();
}

/// Get the (sparse) jacobian matrix of this mapping, as used in applyJ/applyJT.
/// This matrix should have as many columns as DOFs in the input mechanical states
/// (one after the other in case of multi-mappings), and as many lines as DOFs in
/// the output mechanical states.
///
/// applyJ(out, in) should be equivalent to $out = J in$.
/// applyJT(out, in) should be equivalent to $out = J^T in$.
///
/// @TODO Note that if the mapping provides this matrix, then a default implementation
/// of all other related methods could be provided, or optionally used to verify the
/// provided implementations for debugging.
const sofa::defaulttype::BaseMatrix* BaseMapping::getJ(const MechanicalParams* /*mparams*/)
{
    serr << "BaseMapping::getJ() NOT IMPLEMENTED BY " << getClassName() << sendl;

    return getJ();
}

const sofa::defaulttype::BaseMatrix* BaseMapping::getJ()
{
    serr << helper::logging::Message::Deprecated << "Calling deprecated getJ() method in " << getClassName() << ". Use getJ(const MechanicalParams *) instead." << sendl;
    return NULL;
}

sofa::defaulttype::BaseMatrix* BaseMapping::createMappedMatrix(const behavior::BaseMechanicalState* /*state1*/, const behavior::BaseMechanicalState* /*state2*/, func_createMappedMatrix)
{
    serr << "BaseMapping::createMappedMatrix() NOT IMPLEMENTED BY " << getClassName() << sendl;
    return NULL;
}


bool BaseMapping::testMechanicalState(BaseState* state)
{
    bool isMecha = false;
    if(state)
    {
        behavior::BaseMechanicalState* toMechaModel = state->toBaseMechanicalState();
        isMecha = (toMechaModel) ? true : false;
    }
    return isMecha;
}

bool BaseMapping::insertInNode( objectmodel::BaseNode* node )
{
    if( isMechanical() ) node->addMechanicalMapping(this);
    else node->addMapping(this);
    Inherit1::insertInNode(node);
    return true;
}

bool BaseMapping::removeInNode( objectmodel::BaseNode* node )
{
    node->removeMechanicalMapping(this);
    node->removeMapping(this);
    Inherit1::removeInNode(node);
    return true;
}

} // namespace core

} // namespace sofa
