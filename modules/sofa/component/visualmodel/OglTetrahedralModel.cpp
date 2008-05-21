#include "OglTetrahedralModel.h"

#include <sofa/helper/gl/GLshader.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(OglTetrahedralModel)

int OglTetrahedralModelClass = sofa::core::RegisterObject("Tetrahedral model for OpenGL display")
        .add< OglTetrahedralModel >()
        ;

OglTetrahedralModel::OglTetrahedralModel()
{
}

OglTetrahedralModel::~OglTetrahedralModel()
{
}

void OglTetrahedralModel::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    topo = context->core::objectmodel::BaseContext::get<topology::TetrahedronSetTopology<Vec3fTypes> >();
    nodes = context->get< sofa::component::MechanicalObject<Vec3fTypes> >();

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

void OglTetrahedralModel::drawVisual()
{
//	glDisable(GL_CULL_FACE);
//	glBegin(GL_LINES_ADJACENCY_EXT);
//		glVertex3f(5.0,0.0,0.0);
//		glVertex3f(0.0,0.0,0.0);
//		glVertex3f(0.0,-5.0,0.0);
//		glVertex3f(2.5,-2.5,-3.0);
//	glEnd();


    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    helper::vector<topology::Tetrahedron> vec = topo->getTetras();
    helper::vector<topology::Tetrahedron>::iterator it;
    Vec3fTypes::VecCoord& x = *nodes->getX();
    Vec3f v;

#ifdef GL_LINES_ADJACENCY_EXT
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
}

bool OglTetrahedralModel::addBBox(double* minBBox, double* maxBBox)
{
    helper::vector<topology::Tetrahedron> vec = topo->getTetras();
    helper::vector<topology::Tetrahedron>::iterator it;
    Vec3fTypes::VecCoord& x = *nodes->getX();
    Vec3f v;

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
