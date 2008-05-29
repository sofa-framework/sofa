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
        std::cerr << "No mecha." << std::endl;
        return;
    }

    if (!topo)
    {
        std::cerr << "No topo." << std::endl;
        return;
    }
}

template<class DataTypes>
void OglTetrahedralModel<DataTypes>::drawTransparent()
{
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
    const core::componentmodel::topology::BaseMeshTopology::SeqTetras& vec = topo->getTetras();
    core::componentmodel::topology::BaseMeshTopology::SeqTetras::const_iterator it;

#ifdef GL_LINES_ADJACENCY_EXT
    VecCoord& x = *nodes->getX();
    Coord v;

    glBegin(GL_LINES_ADJACENCY_EXT);
    for(it = vec.begin() ; it != vec.end() ; it++)
    {

        for (unsigned int i=0 ; i< 4 ; i++)
        {
            v = x[(*it)[i]];
            glVertex3f(v[0], v[1], v[2]);
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
