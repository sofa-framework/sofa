/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef OGLTETRAHEDRALMODEL_INL_
#define OGLTETRAHEDRALMODEL_INL_

#include "OglTetrahedralModel.h"

#include <sofa/helper/gl/GLshader.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::defaulttype;

template<class DataTypes>
OglTetrahedralModel<DataTypes>::OglTetrahedralModel()
    :depthTest(initData(&depthTest, (bool) true, "depthTest", "Set Depth Test")),
     blending(initData(&blending, (bool) true, "blending", "Set Blending"))
{
}

template<class DataTypes>
OglTetrahedralModel<DataTypes>::~OglTetrahedralModel()
{
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topo);
    context->get(nodes);


    if (!nodes)
    {
        std::cerr << "OglTetrahedralModel : Error : no MechanicalState found." << std::endl;
        return;
    }

    if (!topo)
    {
        std::cerr << "OglTetrahedralModel : Error : no BaseMeshTopology found." << std::endl;
        return;
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::drawTransparent()
{
    if (!getContext()->getShowVisualModels()) return;

    if(blending.getValue())
        glEnable(GL_BLEND);
    if(depthTest.getValue())
        glDepthMask(GL_FALSE);

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    core::componentmodel::topology::BaseMeshTopology::SeqTetras::const_iterator it;

#ifdef GL_LINES_ADJACENCY_EXT
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& vec = topo->getTetras();
    VecCoord& x = *nodes->getX();
    Coord v;

    glBegin(GL_LINES_ADJACENCY_EXT);
    for(it = vec.begin() ; it != vec.end() ; it++)
    {

        for (unsigned int i=0 ; i< 4 ; i++)
        {
            v = x[(*it)[i]];
            glVertex3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]);
        }
    }
    glEnd();
#endif
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

template<class DataTypes>
bool OglTetrahedralModel<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& vec = topo->getTetras();
    core::componentmodel::topology::BaseMeshTopology::SeqTetras::const_iterator it;
    VecCoord& x = *nodes->getX();
    Coord v;

    for(it = vec.begin() ; it != vec.end() ; it++)
    {
        for (unsigned int i=0 ; i< 4 ; i++)
        {
            v = x[(*it)[i]];

            if (minBBox[0] > v[0]) minBBox[0] = v[0];
            if (minBBox[1] > v[1]) minBBox[1] = v[1];
            if (minBBox[2] > v[2]) minBBox[2] = v[2];
            if (maxBBox[0] < v[0]) maxBBox[0] = v[0];
            if (maxBBox[1] < v[1]) maxBBox[1] = v[1];
            if (maxBBox[2] < v[2]) maxBBox[2] = v[2];
        }
    }
    return true;
}

}
}
}

#endif //OGLTETRAHEDRALMODEL_H_
