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
#ifndef SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_INL
#define SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_INL

#include <SofaGeneralEngine/QuatToRigidEngine.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace engine
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
}

template <class DataTypes>
QuatToRigidEngine<DataTypes>::~QuatToRigidEngine()
{

}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::init()
{
    addInput(&f_positions);
    addInput(&f_orientations);
    addInput(&f_colinearPositions);

    addOutput(&f_rigids);

    setDirtyValue();
}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void QuatToRigidEngine<DataTypes>::update()
{

    const helper::vector<Vec3>& positions = f_positions.getValue();
    const helper::vector<Quat>& orientations = f_orientations.getValue();
    const helper::vector<Vec3>& colinearPositions = f_colinearPositions.getValue();

    cleanDirty();

    helper::vector<RigidVec3>& rigids = *(f_rigids.beginWriteOnly());

    unsigned int sizeRigids = positions.size();

    int nbPositions = positions.size();
    int nbOrientations = orientations.size();
    if (!nbOrientations)
    {
        serr << "Warnings : no orientations" << sendl;
        sizeRigids = 0;
    }
    else if (nbOrientations == 1)
    { // We will use the same orientation for all rigids
        sizeRigids = nbPositions;
    }
    else if(nbOrientations > 1 && nbPositions != nbOrientations)
    {
        serr << "Warnings : size of positions and orientations are not equal" << sendl;
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

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ENGINE_QUATTORIGIDENGINE_INL
