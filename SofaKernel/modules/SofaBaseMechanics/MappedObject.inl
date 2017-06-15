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
#ifndef SOFA_COMPONENT_MAPPEDOBJECT_INL
#define SOFA_COMPONENT_MAPPEDOBJECT_INL

#include <SofaBaseMechanics/MappedObject.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace container
{

template <class DataTypes>
MappedObject<DataTypes>::MappedObject()
    : f_X( initData(&f_X, "position", "position vector") )
    , f_V( initData(&f_V, "velocity", "velocity vector") )
{
}

template <class DataTypes>
MappedObject<DataTypes>::~MappedObject()
{
}

template <class DataTypes>
void MappedObject<DataTypes>::init()
{
    if (getSize() == 0)
    {
        sofa::core::topology::BaseMeshTopology* topo = this->getContext()->getMeshTopology();
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

}

} // namespace component

} // namespace sofa

#endif

