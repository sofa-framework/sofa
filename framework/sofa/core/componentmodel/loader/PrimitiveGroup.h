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

#ifndef SOFA_CORE_COMPONENTMODEL_LOADER_PRIMITIVEGROUP_H_
#define SOFA_CORE_COMPONENTMODEL_LOADER_PRIMITIVEGROUP_H_

#include <sofa/core/core.h>
#include <sofa/core/componentmodel/loader/Material.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace loader
{

class PrimitiveGroup
{
public:
    int p0, nbp;
    std::string materialName;
    std::string groupName;
    int materialId;
    inline friend std::ostream& operator << (std::ostream& out, const PrimitiveGroup &g)
    {
        out << g.groupName << " " << g.materialName << " " << g.materialId << " " << g.p0 << " " << g.nbp;
        return out;
    }
    inline friend std::istream& operator >> (std::istream& in, PrimitiveGroup &g)
    {
        in >> g.groupName >> g.materialName >> g.materialId >> g.p0 >> g.nbp;
        return in;
    }
    PrimitiveGroup() : p0(0), nbp(0), materialId(-1) {}
};

} // namespace loader

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif /* SOFA_CORE_COMPONENTMODEL_LOADER_PRIMITIVEGROUP_H_ */
