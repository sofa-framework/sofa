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
#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>

#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/common/Simulation.h>



namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;


template<class DataTypes>
TSphereModel<DataTypes>::TSphereModel()
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , translation(initData(&translation,Vector3(),"translation","Translation of the spheres"))
    , filename(initData(&filename, "fileSphere", "File .sph describing the spheres"))
    , mstate(NULL)
{
    addAlias(&filename,"filename");
}

template<class DataTypes>
TSphereModel<DataTypes>::TSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , translation(initData(&translation,Vector3(),"translation","Translation of the spheres"))
    , filename(initData(&filename, "fileSphere", "File .sph describing the spheres"))
    , mstate(_mstate)
{
    addAlias(&filename,"filename");
}


template<class DataTypes>
void TSphereModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal &r = *radius.beginEdit();

    if ((int)r.size() < size)
    {
        r.reserve(size);
        while ((int)r.size() < size)
            r.push_back(defaultRadius.getValue());
    }
    else
    {
        r.resize(size);
    }

    radius.endEdit();
}


template<class DataTypes>
void TSphereModel<DataTypes>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (mstate==NULL)
    {
        serr<<"TSphereModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    if (radius.getValue().empty() && !filename.getValue().empty())
    {
        load(filename.getFullPath().c_str());
    }
    else
    {
        const int npoints = mstate->getX()->size();
        resize(npoints);
    }

    applyTranslation(translation.getValue().x(),translation.getValue().y(),translation.getValue().z());
}

template<class DataTypes>
void TSphereModel<DataTypes>::applyTranslation(const double dx, const double dy, const double dz)
{
    for( int index = 0; index< mstate->getSize(); ++index)
    {
        // Collision element iterator
        TSphere<DataTypes> s(this, index);
        s.translate(dx,dy,dz);
    }
}

template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* ,int index)
{
    TSphere<DataTypes> t(this,index);

    Vector3 p = t.p();
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutSolidSphere(t.r(), 32, 16);
    glPopMatrix();
}
template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (getContext()->getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(0,getContext()->getShowWireFrame());

        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);

        glColor4fv(getColor4f());
        // Check topological modifications
        const int npoints = mstate->getX()->size();

        std::vector<Vector3> points;
        std::vector<float> radius;
        for (int i=0; i<npoints; i++)
        {
            TSphere<DataTypes> t(this,i);
            Vector3 p = t.p();
            points.push_back(p);
            radius.push_back(t.r());
        }

        vparams->drawTool()->setLightingEnabled(true); //Enable lightning
        vparams->drawTool()->drawSpheres(points, radius, Vec<4,float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning

    }
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}


template <class DataTypes>
void TSphereModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated)
        return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            TSphere<DataTypes> p(this,i);
            const typename TSphere<DataTypes>::Real r = p.r();
            const Coord minElem = p.center() - Coord(r,r,r);
            const Coord maxElem = p.center() + Coord(r,r,r);

            cubeModel->setParentOf(i, minElem, maxElem);

        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template <class DataTypes>
void TSphereModel<DataTypes>::computeContinuousBoundingTree(double dt, int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated)
        return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            TSphere<DataTypes> p(this,i);
            const Vector3& pt = p.p();
            const Vector3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }

            typename TSphere<DataTypes>::Real r = p.r();
            cubeModel->setParentOf(i, minElem - Vector3(r,r,r), maxElem + Vector3(r,r,r));
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template <class DataTypes>
typename TSphereModel<DataTypes>::Real TSphereModel<DataTypes>::getRadius(const int i) const
{
    if(i < (int) this->radius.getValue().size())
        return radius.getValue()[i];
    else
        return defaultRadius.getValue();
}

template <class DataTypes>
void TSphereModel<DataTypes>::setRadius(const int i, const typename TSphereModel<DataTypes>::Real _r)
{
    VecReal &r = *radius.beginEdit();

    if ((int)r.size() <= i)
    {
        r.reserve(i+1);
        while ((int)r.size() <= i)
            r.push_back(defaultRadius.getValue());
    }

    r[i] = _r;
    radius.endEdit();
}

template <class DataTypes>
void TSphereModel<DataTypes>::setRadius(const typename TSphereModel<DataTypes>::Real r)
{
    *defaultRadius.beginEdit() = r;
    defaultRadius.endEdit();

    radius.beginEdit()->clear();
    radius.endEdit();
}

template <class DataTypes>
class TSphereModel<DataTypes>::Loader : public helper::io::SphereLoader
{
public:
    TSphereModel<DataTypes>* dest;
    Loader(TSphereModel<DataTypes>* dest)
        : dest(dest) {}

    void addSphere(SReal x, SReal y, SReal z, SReal r)
    {
        dest->addSphere(Vector3(x,y,z),r);
    }
};

template <class DataTypes>
bool TSphereModel<DataTypes>::load(const char* filename)
{
    this->resize(0);
    std::string sphereFilename(filename);
    if (!sofa::helper::system::DataRepository.findFile (sphereFilename))
        serr<<"TSphere File \""<< filename <<"\" not found"<< sendl;

    Loader loader(this);
    return loader.load(filename);
}

template <class DataTypes>
int TSphereModel<DataTypes>::addSphere(const Vector3& pos, Real r)
{
    int i = size;
    resize(i+1);
    if((int) mstate->getX()->size() != i+1)
        mstate->resize(i+1);

    setSphere(i, pos, r);
    return i;
}

template <class DataTypes>
void TSphereModel<DataTypes>::setSphere(int i, const Vector3& pos, Real r)
{
    helper::WriteAccessor<Data<VecCoord> > xData = *mstate->write(core::VecCoordId::position());
    xData.wref()[i] = pos;
    setRadius(i,r);
}

} // namespace collision

} // namespace component

} // namespace sofa

