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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_CPP

#include <sofa/component/forcefield/PlaneForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


template<>
void PlaneForceField<Rigid3dTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v1 = v;

    //this->dfdd.resize(p1.size());
    this->contacts.clear();
    f1.resize(p1.size());

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = p1[i].getCenter()*planeNormal.getValue().getVCenter()-planeD.getValue();
        if (d<0)
        {
            //serr<<"PlaneForceField<DataTypes>::addForce, d = "<<d<<sendl;
            Real forceIntensity = -this->stiffness.getValue()*d;
            //serr<<"PlaneForceField<DataTypes>::addForce, stiffness = "<<stiffness.getValue()<<sendl;
            Real dampingIntensity = -this->damping.getValue()*d;
            //serr<<"PlaneForceField<DataTypes>::addForce, dampingIntensity = "<<dampingIntensity<<sendl;
            Deriv force;
            force.getVCenter() = planeNormal.getValue().getVCenter()*forceIntensity - v1[i].getVCenter()*dampingIntensity;
            //serr<<"PlaneForceField<DataTypes>::addForce, force = "<<force<<sendl;
            f1[i]+=force;
            //this->dfdd[i] = -this->stiffness;
            this->contacts.push_back(i);
        }
    }
}

template<>
void PlaneForceField<Rigid3dTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& df, const DataVecDeriv& dx)
{
    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;

    df1.resize(dx1.size());
    const Real fact = (Real)(-this->stiffness.getValue() * mparams->kFactor());
    for (unsigned int i=0; i<this->contacts.size(); i++)
    {
        unsigned int p = this->contacts[i];
        assert(p<dx1.size());
        df1[p].getVCenter() += planeNormal.getValue().getVCenter() * (fact * (dx1[p].getVCenter()*planeNormal.getValue().getVCenter()));
    }
}

template<>
void PlaneForceField<Rigid3dTypes>::setPlane(const Deriv& normal, Real d)
{
    Real n = normal.getVCenter().norm();
    planeNormal.beginEdit()->getVCenter() = normal.getVCenter() / n;
    planeNormal.endEdit();
    planeD.setValue( d / n );
}

template<>
void PlaneForceField<Rigid3dTypes>::drawPlane(float size)
{
    if (size == 0.0f) size = (float)drawSize.getValue();

    helper::ReadAccessor<VecCoord> p1 = *this->mstate->getX();

    defaulttype::Vec3d normal; normal = planeNormal.getValue().getVCenter();

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;


    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);


    std::vector< defaulttype::Vector3 > points;

    points.push_back(corners[0]);
    points.push_back(corners[1]);
    points.push_back(corners[2]);

    points.push_back(corners[0]);
    points.push_back(corners[2]);
    points.push_back(corners[3]);

    simulation::getSimulation()->DrawUtility().setPolygonMode(2,false); //Cull Front face

    simulation::getSimulation()->DrawUtility().drawTriangles(points, defaulttype::Vec<4,float>(color.getValue()[0],color.getValue()[1],color.getValue()[2],0.5));
    simulation::getSimulation()->DrawUtility().setPolygonMode(0,false); //No Culling
    glDisable(GL_CULL_FACE);

    std::vector< defaulttype::Vector3 > pointsLine;
    // lines for points penetrating the plane

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;


    defaulttype::Vector3 point1,point2;
    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = p1[i].getCenter()*planeNormal.getValue().getVCenter()-planeD.getValue();
        Coord p2 = p1[i];
        p2 += planeNormal.getValue()*(-d);
        if (d<0)
        {
            point1 = DataTypes::getCPos(p1[i]);
            point2 = DataTypes::getCPos(p2);
        }
        pointsLine.push_back(point1);
        pointsLine.push_back(point2);
    }
    simulation::getSimulation()->DrawUtility().drawLines(pointsLine, 1, defaulttype::Vec<4,float>(1,0,0,1));
}

template<>
void PlaneForceField<Rigid3dTypes>::updateStiffness(const VecCoord& vx)
{
    helper::ReadAccessor<VecCoord> x = vx;

    this->contacts.clear();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if (localRange.getValue()[0] >= 0)
        ibegin = localRange.getValue()[0];

    if (localRange.getValue()[1] >= 0 && (unsigned int)localRange.getValue()[1]+1 < iend)
        iend = localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = x[i].getCenter()*planeNormal.getValue().getVCenter()-planeD.getValue();
        if (d<0)
        {
            this->contacts.push_back(i);
        }
    }
}

template<>
void PlaneForceField<Rigid3dTypes>::rotate( Deriv axe, Real angle )
{
    defaulttype::Vec3d axe3d(1,1,1); axe3d = axe.getVCenter();
    defaulttype::Vec3d normal3d; normal3d = planeNormal.getValue().getVCenter();
    defaulttype::Vec3d v = normal3d.cross(axe3d);
    if (v.norm2() < 1.0e-10) return;
    v.normalize();
    v = normal3d * cos ( angle ) + v * sin ( angle );
    planeNormal.beginEdit()->getVCenter() = v;
    planeNormal.endEdit();
}


template<>
bool PlaneForceField<Rigid3dTypes>::addBBox(double* minBBox, double* maxBBox)
{
    if (!bDraw.getValue()) return false;

    defaulttype::Vec3d normal; normal = planeNormal.getValue().getVCenter();
    double size=10.0;

    // find a first vector inside the plane
    defaulttype::Vec3d v1;
    if( 0.0 != normal[0] ) v1 = defaulttype::Vec3d(-normal[1]/normal[0], 1.0, 0.0);
    else if ( 0.0 != normal[1] ) v1 = defaulttype::Vec3d(1.0, -normal[0]/normal[1],0.0);
    else if ( 0.0 != normal[2] ) v1 = defaulttype::Vec3d(1.0, 0.0, -normal[0]/normal[2]);
    v1.normalize();
    // find a second vector inside the plane and orthogonal to the first
    defaulttype::Vec3d v2;
    v2 = v1.cross(normal);
    v2.normalize();

    defaulttype::Vec3d center = normal*planeD.getValue();
    defaulttype::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    for (unsigned int i=0; i<4; i++)
    {
        for (int c=0; c<3; c++)
        {
            if (corners[i][c] > maxBBox[c]) maxBBox[c] = corners[i][c];
            if (corners[i][c] < minBBox[c]) minBBox[c] = corners[i][c];
        }
    }
    return true;
}


SOFA_DECL_CLASS(PlaneForceField)

int PlaneForceFieldClass = core::RegisterObject("Repulsion applied by a plane toward the exterior (half-space)")
#ifndef SOFA_FLOAT
        .add< PlaneForceField<Vec3dTypes> >()
        .add< PlaneForceField<Vec2dTypes> >()
        .add< PlaneForceField<Vec1dTypes> >()
        .add< PlaneForceField<Vec6dTypes> >()
        .add< PlaneForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PlaneForceField<Vec3fTypes> >()
        .add< PlaneForceField<Vec2fTypes> >()
        .add< PlaneForceField<Vec1fTypes> >()
        .add< PlaneForceField<Vec6fTypes> >()
//.add< PlaneForceField<Rigid3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec3dTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec2dTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec1dTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec6dTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec3fTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec2fTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec1fTypes>;
template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Vec6fTypes>;
//template class SOFA_COMPONENT_FORCEFIELD_API PlaneForceField<Rigid3fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
