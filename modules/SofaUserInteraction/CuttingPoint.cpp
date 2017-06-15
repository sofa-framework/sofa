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
#include <SofaUserInteraction/CuttingPoint.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace collision
{

void CuttingPoint::init()
{
    this->core::objectmodel::BaseObject::init();
    mstate = getContext()->get<core::behavior::MechanicalState<DataTypes> >();
    topology = this->getContext()->getMeshTopology();

    // Initialize functions and parameters
    f_points.createTopologicalEngine(topology);
    f_points.registerTopologicalData();
}

void CuttingPoint::setPointSubSet(SetIndexArray pSub)
{
    SetIndexArray& points = *f_points.beginEdit();
    points = pSub;
    f_points.endEdit();
}

void CuttingPoint::setPoint(unsigned int p)
{
    SetIndexArray& points = *f_points.beginEdit();
    points.resize(1);
    points[0] = p;
    f_points.endEdit();
}

void CuttingPoint::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    if (!mstate) return;
    glDisable (GL_LIGHTING);
    if (lastPos != pos)
    {
        glColor4f (1,1,0,1);
        glLineWidth(3);
        glBegin (GL_LINES);
        helper::gl::glVertexT(lastPos);
        helper::gl::glVertexT(pos);
        glEnd();
        glLineWidth(1);
    }
    if (newPos != pos)
    {
        glColor4f (0.5f,0.5f,0,1);
        glLineWidth(1);
        glBegin (GL_LINES);
        helper::gl::glVertexT(pos);
        helper::gl::glVertexT(newPos);
        glEnd();
    }
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    const SetIndexArray& points = getPointSubSet();
    if (!points.empty())
    {
        if (id.getValue() != -1 && prevID.getValue() == -1 && nextID.getValue() == -1)
            glPointSize(20);
        else
            glPointSize(10);
        if (cutInProgress.getValue())
            glColor4f (1,1,1,1);
        else if (canBeCut())
            glColor4f (1,1,0,1);
        else
            glColor4f (0.5f,0.5f,0.5f,1);

        glBegin (GL_POINTS);
        for (SetIndexArray::const_iterator it = points.begin();
                it != points.end();
                ++it)
        {
            helper::gl::glVertexT(x[*it]);
        }
        glEnd();
        glPointSize(1);
    }

    /*
    if (!points.empty())
    {
        int n = points.size();
        glPointSize(5);
        glColor4f (1,1,0,1);
        glBegin (GL_POINTS);
        helper::gl::glVertexT(x[points[0]]);
        glColor4f (0.5f,0.5f,0,1);
        for (int i=1;i<n;++i)
            helper::gl::glVertexT(x[points[i]]);
        glEnd();
        glPointSize(1);
        glLineWidth(3);
        glBegin (GL_LINES);
        for (int i=1;i<n;++i)
        {
            if (i < 3)
            {
                glColor4f (1,1,0,1);
                helper::gl::glVertexT(x[points[0]]);
                helper::gl::glVertexT(x[points[i]]);
            }
            else
            {
                glColor4f (0.5f,0.5f,0,1);
                helper::gl::glVertexT(x[points[i]]);
                helper::gl::glVertexT(x[points[(i-2)%(n-3) + 3]]);
            }
        }
        glEnd();
        glLineWidth(1);
    }
    */
#endif /* SOFA_NO_OPENGL */
}

} //collision
} //component
} // sofa
