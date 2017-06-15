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
#ifndef OGLTETRAHEDRALMODEL_INL_
#define OGLTETRAHEDRALMODEL_INL_

#include "CudaTetrahedralVisualModel.h"

#include <sofa/helper/gl/GLSLShader.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::defaulttype;

template<class TCoord, class TDeriv, class TReal>
OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::OglTetrahedralModel()
    : needUpdateTopology(true)
    , depthTest(initData(&depthTest, (bool) true, "depthTest", "Set Depth Test"))
    , blending(initData(&blending, (bool) true, "blending", "Set Blending"))
    , useVBO( initData( &useVBO, false, "useVBO", "true to activate Vertex Buffer Object") )
{
}

template<class TCoord, class TDeriv, class TReal>
OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::~OglTetrahedralModel()
{
}

template<class TCoord, class TDeriv, class TReal>
void OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topo);
    context->get(nodes);


    if (!nodes)
    {
        serr << "No mecha." << sendl;
        return;
    }

    if (!topo)
    {
        serr << "No topo." << sendl;
        return;
    }

    updateTopology();
}

template<class TCoord, class TDeriv, class TReal>
void OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::drawTransparent(const core::visual::VisualParams*)
{
    //if (!getContext()->getShowVisualModels()) return;

//	glDisable(GL_CULL_FACE);
//	glBegin(GL_LINES_ADJACENCY_EXT);
//		glVertex3f(5.0,0.0,0.0);
//		glVertex3f(0.0,0.0,0.0);
//		glVertex3f(0.0,-5.0,0.0);
//		glVertex3f(2.5,-2.5,-3.0);
//	glEnd();

    if(blending.getValue())
        glEnable(GL_BLEND);
    if(depthTest.getValue())
        glDepthMask(GL_FALSE);

    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

#ifdef GL_LINES_ADJACENCY_EXT
    //TODO: Const ? Read-Only ?
    //VecCoord& x = *nodes->getX();
    Data<VecCoord>* d_x = nodes->write(core::VecCoordId::position());
    VecCoord& x = *d_x->beginEdit();

    bool vbo = useVBO.getValue();

    GLuint vbo_x = vbo ? x.bufferRead(true) : 0;
    if (vbo_x)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
        glVertexPointer (3, (sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), NULL);
    }
    else
        glVertexPointer (3, (sizeof(Real)==sizeof(double))?GL_DOUBLE:GL_FLOAT, sizeof(Coord), x.hostRead());

    glEnableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    if (tetras.size() > 0)
    {
        GLuint vbo_t = vbo ? tetras.bufferRead(true) : 0;
        if (vbo_t)
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_t);
            glDrawElements(GL_LINES_ADJACENCY_EXT, tetras.size() * 4, GL_UNSIGNED_INT, NULL);
        }
        else
            glDrawElements(GL_LINES_ADJACENCY_EXT, tetras.size() * 4, GL_UNSIGNED_INT, tetras.hostRead());
    }

    if (vbo)
    {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    d_x->endEdit();
#endif
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}

template<class TCoord, class TDeriv, class TReal>
bool OglTetrahedralModel< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addBBox(SReal* minBBox, SReal* maxBBox)
{
    const core::topology::BaseMeshTopology::SeqTetrahedra& vec = topo->getTetrahedra();
    core::topology::BaseMeshTopology::SeqTetrahedra::const_iterator it;
    const VecCoord& x = nodes->read(core::ConstVecCoordId::position())->getValue();
    Coord v;

    for(it = vec.begin() ; it != vec.end() ; ++it)
    {
        for (unsigned int i=0 ; i< 4 ; ++i)
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
