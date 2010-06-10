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
#include "BaseForceField.h"

namespace sofa
{

namespace core
{

namespace behavior
{

void BaseForceField::addMBKdx(double /*mFactor*/, double bFactor, double kFactor)
{
    if (kFactor != 0.0 || bFactor != 0.0)
        addDForce(kFactor, bFactor);
}

void BaseForceField::addMBKv(double /*mFactor*/, double bFactor, double kFactor)
{
    if (kFactor != 0.0 || bFactor != 0.0)
        addDForceV(kFactor, bFactor);
}

void BaseForceField::addBToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/, double /*bFact*/)
{
}

void BaseForceField::addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double /*mFact*/, double bFact, double kFact)
{
    if (kFact != 0.0)
        addKToMatrix(matrix, kFact);
    if (bFact != 0.0)
        addBToMatrix(matrix, bFact);
}

} // namespace behavior

} // namespace core

} // namespace sofa
