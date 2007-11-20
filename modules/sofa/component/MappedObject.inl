/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPEDOBJECT_INL
#define SOFA_COMPONENT_MAPPEDOBJECT_INL

#include <sofa/component/MappedObject.h>
#include <sofa/core/componentmodel/topology/Topology.h>

namespace sofa
{

namespace component
{

template <class DataTypes>
MappedObject<DataTypes>::MappedObject()
    : f_X( dataField(&f_X, "position", "position vector") )
    , f_V( dataField(&f_V, "velocity", "velocity vector") )
{
}

template <class DataTypes>
MappedObject<DataTypes>::~MappedObject()
{
}

template <class DataTypes>
void MappedObject<DataTypes>::init()
{
    if (getX()->size() == 0)
    {
        core::componentmodel::topology::Topology* topo = dynamic_cast<core::componentmodel::topology::Topology*>(this->getContext()->getTopology());
        if (topo!=NULL && topo->hasPos() && topo->getContext() == this->getContext())
        {
            VecCoord& x = *getX();
            int nbp = topo->getNbPoints();
//             std::cout<<"Setting "<<nbp<<" points from topology."<<std::endl;
            x.resize(nbp);
            for (int i=0; i<nbp; i++)
            {
                DataTypes::set(x[i], topo->getPX(i), topo->getPY(i), topo->getPZ(i));
            }
        }
    }
}

} // namespace component

} // namespace sofa

#endif

