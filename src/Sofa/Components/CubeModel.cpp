#include "CubeModel.h"

#include <GL/glut.h>

namespace Sofa
{

namespace Components
{

CubeModel::CubeModel()
    : previous(NULL), next(NULL)
{
    clear();
}

void CubeModel::clear()
{
    for (unsigned int i=0; i<elems.size(); i++)
        delete elems[i];
    elems.clear();
    getX()->resize(0);
    getV()->resize(0);
    getDx()->resize(0);
    getF()->resize(0);
}

void CubeModel::addCube (const Vector3& min, const Vector3 &max)
{
    setCube(elems.size(), min, max);
}

void CubeModel::setCube(unsigned int index, const Vector3& min, const Vector3 &max)
{
    while (elems.size() <= index)
    {
        getX()->push_back(Vector3(0,0,0));
        getV()->push_back(Vector3(0,0,0));
        getF()->push_back(Vector3(0,0,0));
        getDx()->push_back(Vector3(0,0,0));
        getX()->push_back(Vector3(0,0,0));
        getV()->push_back(Vector3(0,0,0));
        getF()->push_back(Vector3(0,0,0));
        getDx()->push_back(Vector3(0,0,0));
        elems.push_back(new Cube(elems.size(), this));
    }
    (*getX())[2*index+0] = min;
    (*getX())[2*index+1] = max;
}

void CubeModel::draw()
{
    if (!isActive() || !getContext()->getShowCollisionModels()) return;
    //std::cout << "SPHdraw"<<elems.size()<<std::endl;
    glDisable(GL_LIGHTING);
    glColor3f(1.0, 1.0, 1.0);
    for (unsigned int i=0; i<elems.size(); i++)
    {
        static_cast<Cube*>(elems[i])->draw();
    }
    if (getPrevious()!=NULL && dynamic_cast<Abstract::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<Abstract::VisualModel*>(getPrevious())->draw();
}

} // namespace Components

} // namespace Sofa
