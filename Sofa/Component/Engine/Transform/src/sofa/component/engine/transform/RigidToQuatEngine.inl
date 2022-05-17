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
#include <sofa/component/engine/transform/RigidToQuatEngine.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::engine::transform
{
template <class DataTypes>
RigidToQuatEngine<DataTypes>::RigidToQuatEngine()
    : f_positions( initData (&f_positions, "positions", "Positions (Vector of 3)") )
    , f_orientations( initData (&f_orientations, "orientations", "Orientations (Quaternion)") )
    , f_orientationsEuler( initData (&f_orientationsEuler, "orientationsEuler", "Orientations (Euler angle)") )
    , f_rigids( initData (&f_rigids, "rigids", "Rigid (Position + Orientation)") )
{
    //
    addAlias(&f_positions,"position");
    addAlias(&f_orientations,"orientation");
    addAlias(&f_rigids,"rigid");

    addInput(&f_rigids);

    addOutput(&f_positions);
    addOutput(&f_orientations);
    addOutput(&f_orientationsEuler);
}

template <class DataTypes>
RigidToQuatEngine<DataTypes>::~RigidToQuatEngine()
{

}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::doUpdate()
{
    helper::ReadAccessor< Data< type::vector<RigidVec3> > > rigids = f_rigids;
    helper::WriteOnlyAccessor< Data< type::vector<Vec3> > > positions = f_positions;
    helper::WriteOnlyAccessor< Data< type::vector<Quat> > > orientations = f_orientations;
    helper::WriteOnlyAccessor< Data< type::vector<Vec3> > > orientationsEuler = f_orientationsEuler;

    unsigned int sizeRigids = rigids.size();
    positions.resize(sizeRigids);
    orientations.resize(sizeRigids);
    orientationsEuler.resize(sizeRigids);
    for (unsigned int i=0 ; i< sizeRigids ; i++)
    {
        RigidVec3 r = rigids[i];
        positions[i] = r.getCenter();
        orientations[i] = r.getOrientation();
        orientationsEuler[i] = orientations[i].toEulerVector();
    }
}

} //namespace sofa::component::engine::transform
