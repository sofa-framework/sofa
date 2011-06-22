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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef OGLTETRAHEDRALMODEL_INL_
#define OGLTETRAHEDRALMODEL_INL_

#include "OglTetrahedralModel.h"

#include <sofa/helper/gl/GLSLShader.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::defaulttype;

template<class DataTypes>
OglTetrahedralModel<DataTypes>::OglTetrahedralModel()
    : nodes(NULL), topo(NULL)
    , depthTest(initData(&depthTest, (bool) true, "depthTest", "Set Depth Test"))
    , blending(initData(&blending, (bool) true, "blending", "Set Blending"))

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
        serr << "OglTetrahedralModel : Error : no MechanicalState found." << sendl;
        return;
    }

    if (!topo)
    {
        serr << "OglTetrahedralModel : Error : no BaseMeshTopology found." << sendl;
        return;
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::drawTransparent(const core::visual::VisualParams*)
{
    if (!getContext()->getShowVisualModels()) return;

    if(blending.getValue())
        glEnable(GL_BLEND);
    if(depthTest.getValue())
        glDepthMask(GL_FALSE);

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    //core::topology::BaseMeshTopology::SeqHexahedra::const_iterator it;
    core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;

#ifdef GL_LINES_ADJACENCY_EXT

    const core::topology::BaseMeshTopology::SeqTetrahedra& vec = topo->getTetrahedra();

    const VecCoord& x = *nodes->getX();
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
    /*
    	const core::topology::BaseMeshTopology::SeqHexahedra& vec = topo->getHexahedra();

    	VecCoord& x = *nodes->getX();
    	Coord v;


    	const unsigned int hexa2tetrahedra[24] = { 0, 5, 1, 6,
    										   0, 1, 3, 6,
    										   1, 3, 6, 2,
    										   6, 3, 0, 7,
    										   6, 7, 0, 5,
    										   7, 5, 4, 0 };



    	glBegin(GL_LINES_ADJACENCY_EXT);
    	for(it = vec.begin() ; it != vec.end() ; it++)
    	{

    		for (unsigned int i=0 ; i<6 ; i++)
    		{
    			for (unsigned int j=0 ; j<4 ; j++)
    			{
    				//glVertex3f((GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][0], (GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][1], (GLfloat)x[(*it)[hexa2tetrahedra[i][j]]][2]);
    				glVertex3f((GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][0], (GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][1], (GLfloat)x[(*it)[hexa2tetrahedra[i*4 + j]]][2]);
    			}
    		}
    	}
    	glEnd();
    	*/
#else

#endif
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

template<class DataTypes>
bool OglTetrahedralModel<DataTypes>::addBBox(double* minBBox, double* maxBBox)
{
    if (nodes && topo)
    {
        const core::topology::BaseMeshTopology::SeqTetrahedra& vec = topo->getTetrahedra();
        core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;
        const VecCoord& x = *nodes->getX();
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

    return false;
}

}
}
}

#endif //OGLTETRAHEDRALMODEL_H_
