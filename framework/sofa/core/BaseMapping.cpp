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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace core
{

BaseMapping::BaseMapping()
    : f_mapForces(initData(&f_mapForces, true, "mapForces", "Are forces mapped ?"))
    , f_mapConstraints(initData(&f_mapConstraints, true, "mapConstraints", "Are constraints mapped ?"))
    , f_mapMasses(initData(&f_mapMasses, true, "mapMasses", "Are masses mapped ?"))
{
    this->addAlias(&f_mapForces, "isMechanical");
    this->addAlias(&f_mapMasses, "isMechanical");
}

/// Destructor
BaseMapping::~BaseMapping()
{}

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
    serr << "Calling deprecated getJ() method in " << getClassName() << ". Use getJ(const MechanicalParams *) instead." << sendl;
    return getJ();
}

const sofa::defaulttype::BaseMatrix* BaseMapping::getJ()
{
    serr << "BaseMechanicalMapping::getJ() NOT IMPLEMENTED BY " << getClassName() << sendl;
    return NULL;
}

bool BaseMapping::testMechanicalState(BaseState* state)
{
    bool isMecha = false;
    if(state)
    {
        behavior::BaseMechanicalState* toMechaModel = dynamic_cast<behavior::BaseMechanicalState* > (state);
        isMecha = (toMechaModel) ? true : false;
    }
    return isMecha;
}

} // namespace core

} // namespace sofa
