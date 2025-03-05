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
#include <sofa/component/engine/transform/QuatToRigidEngine.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::engine::transform
{
template <class DataTypes>
QuatToRigidEngine<DataTypes>::QuatToRigidEngine()
    : f_positions( initData (&f_positions, "positions", "Positions (Vector of 3)") )
    , f_orientations( initData (&f_orientations, "orientations", "Orientations (Quaternion)") )
    , f_colinearPositions( initData (&f_colinearPositions, "colinearPositions", "Optional positions to restrict output to be colinear in the quaternion Z direction") )
    , f_rigids( initData (&f_rigids, "rigids", "Rigid (Position + Orientation)") )
{
    //
    addAlias(&f_positions,"position");
    addAlias(&f_orientations,"orientation");
    addAlias(&f_colinearPositions,"colinearPosition");
    addAlias(&f_rigids,"rigid");

    addInput(&f_positions);
    addInput(&f_orientations);
    addInput(&f_colinearPositions);

    addOutput(&f_rigids);
}

template <class DataTypes>
QuatToRigidEngine<DataTypes>::~QuatToRigidEngine()
{

}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::doUpdate()
{
    const type::vector<Vec3>& positions = f_positions.getValue();
    const type::vector<Quat>& orientations = f_orientations.getValue();
    const type::vector<Vec3>& colinearPositions = f_colinearPositions.getValue();

    type::vector<RigidVec3>& rigids = *(f_rigids.beginWriteOnly());

    unsigned int sizeRigids = positions.size();

    const int nbPositions = positions.size();
    const int nbOrientations = orientations.size();
    if (!nbOrientations)
    {
        msg_warning() << "No orientations";
        sizeRigids = 0;
    }
    else if (nbOrientations == 1)
    { // We will use the same orientation for all rigids
        sizeRigids = nbPositions;
    }
    else if(nbOrientations > 1 && nbPositions != nbOrientations)
    {
        msg_warning() << "Size of positions and orientations are not equal";
        sizeRigids = std::min(nbPositions, nbOrientations);
    }

    rigids.clear();
    for (unsigned int i=0 ; i< sizeRigids ; i++)
    {
        Vec3 pos = positions[i];
        Quat q = orientations[i % nbOrientations];
        if (i < colinearPositions.size())
        {
            Vec3 colP = colinearPositions[i];
            Vec3 dir = q.rotate(Vec3(0,0,1));
            pos = colP + dir*(dot(dir,(pos-colP))/dir.norm());
        }
        RigidVec3 r(pos, q);
        rigids.push_back(r);
    }

    f_rigids.endEdit();
}

} //namespace sofa::component::engine::transform
