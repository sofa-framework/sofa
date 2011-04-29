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
#ifndef SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL

#include <sofa/component/forcefield/BuoyantForceField.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>
#include <iostream>
#include <math.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::topology;

template <class DataTypes>
BuoyantForceField<DataTypes>::BuoyantForceField():
    m_fluidModel(initData(&m_fluidModel, (Real)1.0f, "fluidModel", "1 for a plane, 2 for a box")),
    m_minBox(initData(&m_minBox, Coord(-100.0, -100,-100.0), "min", "Lower bound of the liquid box")),
    m_maxBox(initData(&m_maxBox, Coord(100.0, 100,0.0), "max", "Upper bound of the liquid box")),
    m_heightPlane(initData(&m_heightPlane, (Real)0.0f, "heightPlane", "height of the fluid orthogonal to the gravity")),
    m_fluidDensity(initData(&m_fluidDensity, (Real)1.0f, "fluidDensity", "Fluid Density")),
    m_fluidViscosity(initData(&m_fluidViscosity, (Real)1e-3, "fluidViscosity", "Fluid Viscosity")),
    m_atmosphericPressure(initData(&m_atmosphericPressure, (Real)101325.0f, "atmosphericPressure", "atmospheric pressure")),
    m_enableViscosity(initData(&m_enableViscosity, true, "enableViscosity", "enable the effects of viscosity")),
    m_turbulentFlow(initData(&m_turbulentFlow, false, "turbulentFlow", "true for turbulent flow, false for laminar")),
    m_flipNormals(initData(&m_flipNormals, false, "flipNormals", "flip normals to inverse the forces applied on the object")),
    m_showPressureForces(initData(&m_showPressureForces, false, "showPressureForces", "Show the pressure forces applied on the surface of the mesh if true")),
    m_showViscosityForces(initData(&m_showViscosityForces, false, "showViscosityForces", "Show the viscosity forces applied on the surface of the mesh if true")),
    m_showBoxOrPlane(initData(&m_showBoxOrPlane, false, "showBoxOrPlane", "Show the box or the plane"))
{

}



template <class DataTypes>
BuoyantForceField<DataTypes>::~BuoyantForceField()
{

}

//check if some useful parameters changed and adjust the others depending on them
template <class DataTypes>
bool BuoyantForceField<DataTypes>::checkParameters()
{
    bool change = false;
    bool recomputeFluidSurface = false;

    if (m_fluidModel == 2.0f)
    {
        if (fluidModel != AABOX)
        {
            fluidModel = AABOX;
            change = true;

            if (this->f_printLog.getValue())
            {
                std::cout << "BuoyantForceField:: " << this->getName() << " fluid is modeled now with a box" << std::endl;
            }
        }
    }
    else
    {
        if (fluidModel != PLANE)
        {
            fluidModel = PLANE;
            change = true;
            if (this->f_printLog.getValue())
            {
                std::cout << "BuoyantForceField:: " << this->getName() << " fluid is modeled now with a plane" << std::endl;
            }
        }
    }

    if ( m_minBoxPrev != m_minBox.getValue() ||  m_maxBoxPrev!= m_maxBox.getValue())
    {
        for ( unsigned int i = 0 ; i < 3 ; i++)
        {
            if (m_minBox.getValue()[i] > m_maxBox.getValue()[i])
            {
                std::cout << "Switch value " << i << " between min and max" << std::endl;

                Coord tempMin = m_minBox.getValue();
                tempMin[i] = m_maxBox.getValue()[i];

                Coord tempMax = m_maxBox.getValue();
                tempMax[i] = m_minBox.getValue()[i];

                m_minBox.setValue(tempMin);
                m_maxBox.setValue(tempMax);
            }
        }
        m_minBoxPrev = m_minBox.getValue();
        m_maxBoxPrev = m_maxBox.getValue();

        recomputeFluidSurface = true;
        change = true;

        if (this->f_printLog.getValue())
        {
            std::cout << "BuoyantForceField:: " << this->getName() << " change bounding box: <" <<  m_minBox.getValue() << "> - <" << m_maxBox.getValue() << ">"<< std::endl;
        }
    }


    if (m_gravity!= this->getContext()->getGravity())
    {
        m_gravity = this->getContext()->getGravity();
        m_gravityNorm = m_gravity.norm();
        recomputeFluidSurface = true;
        change = true;

        if (this->f_printLog.getValue())
        {
            std::cout << "BuoyantForceField:: " << this->getName() << " new gravity : " << m_gravity << std::endl;
        }
    }

    if (recomputeFluidSurface)
    {
        //the surface in a box is the upper face defined by the gravity
        //it's the one with the minimum angle between the normal and the gravity

        if (!m_gravityNorm)
        {
            std::cout << "ERROR(BuoyantForceField::init()) : unable to determine fluid surface because there is no gravity" << std::endl;
        }
        else
        {
            Deriv direction1 = Coord(m_maxBox.getValue()[0], m_minBox.getValue()[1], m_minBox.getValue()[2])
                    - m_minBox.getValue();
            Deriv direction2 = Coord(m_minBox.getValue()[0], m_maxBox.getValue()[1], m_minBox.getValue()[2])
                    - m_minBox.getValue();
            Deriv direction3 = Coord(m_minBox.getValue()[0], m_minBox.getValue()[1], m_maxBox.getValue()[2])
                    - m_minBox.getValue();

            Real angle1 = fabs(dot(direction1, m_gravity)/( m_gravityNorm * direction1.norm()));
            Real angle2 = fabs(dot(direction2, m_gravity)/( m_gravityNorm * direction2.norm()));
            Real angle3 = fabs(dot(direction3, m_gravity)/( m_gravityNorm * direction3.norm()));

            Deriv minDirection;
            if (angle1 >= angle2 && angle1 >= angle3)
            {
                minDirection = direction1 ;
            }
            else if (angle2 >= angle1 && angle2 >= angle3)
            {
                minDirection = direction2 ;
            }
            else if (angle3 >= angle1 && angle3 >= angle1)
            {
                minDirection = direction3;
            }
            else
            {
                std::cout << "ERROR(BuoyantForceField) : case not anticipated in init()" << std::endl;
            }

            Deriv origin = m_minBox.getValue();

            //as the box is axis-oriented
            if (   dot(minDirection, m_gravity) < 0 )
            {
                origin += minDirection;
            }

            m_fluidSurfaceOrigin = Coord(origin[0], origin[1], origin[2]);
            m_fluidSurfaceDirection = Coord(minDirection[0], minDirection[1], minDirection[2]);
        }
    }
    return change;
}

template <class DataTypes>
void BuoyantForceField<DataTypes>::init()
{
    checkParameters();

    this->core::behavior::ForceField<DataTypes>::init();
//	m_TetraTopo = this->getContext()->getMeshTopology();
    this->getContext()->get(m_tetraTopology);

    if (this->f_printLog.getValue())
    {
        std::cout << "BuoyantForceField " << this->getName() << " coupled with " << m_tetraTopology->getName() << std::endl;
    }

    if (!m_tetraTopology)
    {
        std::cout << "WARNING: BuoyantForceField requires tetrahedral topology" <<std::endl;
    }
    else
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "BuoyantForceField::" << this->getName() <<  " The topology " << m_tetraTopology->getName() << " is well tetrahedrical" << std::endl;
        }

        m_tetraTopology->getContext()->get(m_tetraGeo);
        if (!m_tetraGeo)
        {
            std::cout << "ERROR(BuoyantForceField):Cannot get the geometry from the topology" <<std::endl;
        }

        m_tetraTopology->getContext()->get(m_tetraContainer);
        if (!m_tetraContainer)
        {
            std::cout << "ERROR(BuoyantForceField):Cannot get the container from the topology" <<std::endl;
        }

        if (m_fluidDensity.getValue() <= 0.f)
        {
            serr << "Warning(BuoyantForceField):The density of the fluid is negative!" << sendl;
        }

        m_surfaceTriangles.clear();

        //get all the triangles from the tetrahedral topology
        const sofa::helper::vector<Triangle> &triangleArray=m_tetraTopology->getTriangles();

        if (this->f_printLog.getValue())
        {
            std::cout << "BuoyantForceField::" << this->getName() << " There are " << triangleArray.size()<< " triangles in the topology" << std::endl;
        }

        unsigned int nb_surface_triangles = 0;
        for (unsigned int i=0; i<triangleArray.size(); ++i)
        {
            if (m_tetraTopology->getTetrahedraAroundTriangle(i).size()==1)
            {
                m_surfaceTriangles.push_back(i);
                nb_surface_triangles+=1;
            }
        }

        if (this->f_printLog.getValue())
        {
            std::cout << "BuoyantForceField::" << this->getName() << " There are " << nb_surface_triangles << " triangles on the surface of the topology" << std::endl;
        }
    }
    if (this->f_printLog.getValue())
    {
        if (m_flipNormals.getValue())
        {
            std::cout << "BuoyantForceField::" << this->getName() << " Normals are flipped to inverse the forces" << std::endl;
        }
    }
}



template <class DataTypes>
void BuoyantForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v =d_v.getValue();

    if (m_tetraContainer)
    {
        checkParameters();

        m_debugForce.clear();
        m_debugPosition.clear();
        m_debugViscosityForce.clear();

        //get number of tetrahedra
        int nbTetrahedra = m_tetraContainer->getNbTetrahedra();

        if (!nbTetrahedra)
        {
            serr << "Error(BuoyantForceField):No tetrahedron found in the topology" << sendl;
        }
        else
        {
            if (!m_gravityNorm)
            {
                serr << "Error(BuoyantForceField):Buoyancy works with gravity, but gravity is null  here" << sendl;
                d_f.endEdit();
                return;
            }

            //compute the immersed volume but maybe useless
            Real immersedVolume = static_cast<Real>(0.0f);
            for (int i = 0 ; i < m_tetraContainer->getNbTetras() ; i++)
            {
                Tetra tetra = m_tetraContainer->getTetra(i);
                int nbPointsInside = isTetraInFluid(tetra, x);

                if ( nbPointsInside > 0)
                {
                    immersedVolume += m_tetraGeo->computeTetrahedronVolume(i);
                }
            }
            m_immersedVolume.setValue(immersedVolume);
//		           std::cout << "Immersed Volume >> " << m_immersedVolume.getValue() << std::endl;

            //if there is a part of the volume of the object immersed
            if ( m_immersedVolume.getValue() > static_cast<Real>(0.f))
            {
                //for each triangle of the surface
                for (unsigned int i = 0 ; i < m_surfaceTriangles.size() ; i++)
                {
                    Triangle tri = m_tetraTopology->getTriangle(m_surfaceTriangles[i]);
                    //test if the triangle is in the fluid
                    if (isTriangleInFluid(tri , x))
                    {
                        //get the normal and the area of the triangle
                        Deriv normal = m_tetraGeo->computeTriangleNormal(m_surfaceTriangles[i]);
                        if (m_flipNormals.getValue())
                        {
                            normal = -normal;
                        }
                        Real area = m_tetraGeo->computeTriangleArea(m_surfaceTriangles[i]);

                        //get the centroid of the current triangle
                        Deriv centreTriangle =  m_tetraGeo->computeTriangleCenter(m_surfaceTriangles[i]);

                        //get the distance between the centroid and the surface of the fluid
                        Real z = fabs(distanceFromFluidSurface(centreTriangle) );

                        //the pressure applied by the fluid on the current triangle
                        Real pressure = m_atmosphericPressure.getValue() + m_fluidDensity.getValue() * m_gravityNorm * z;
                        //the force acting on the triangle due to the pressure
                        Deriv triangleForcePressure = normal * (  area * pressure /  normal.norm() );
                        //the force acting on the points of the triangle due to the pressure
                        Deriv pointForcePressure = triangleForcePressure / static_cast<Real>(3.0f);
                        //std::cout << "pointForcePressure = " << pointForcePressure << std::endl;

                        //the drag force
                        Real dragForce = (Real)0.f;
                        if ( m_enableViscosity.getValue())
                        {
                            if (m_turbulentFlow.getValue())
                            {
                                dragForce = - (Real)0.5f * m_fluidDensity.getValue() * area;
                            }
                            else //laminar flow
                            {
                                Coord circumcenter = m_tetraGeo->computeTriangleCircumcenter(m_surfaceTriangles[i]);
                                Coord firstCorner = x[tri[0]];
                                Real radius = (circumcenter - firstCorner).norm();
                                dragForce = - (Real)6.0f * M_PI * m_fluidViscosity.getValue() * radius;
                            }
                        }

                        //apply the force on the points
                        for ( int j = 0 ; j < 3 ; j++)
                        {
                            Deriv velocity = v[tri[j]];
                            f[tri[j]] += pointForcePressure;
                            if ( m_enableViscosity.getValue())
                            {
                                Deriv viscosityForce;
                                if ( m_turbulentFlow.getValue())
                                {
                                    viscosityForce = velocity  * ( dragForce * velocity.norm() );
                                }
                                else //laminar flow
                                {
                                    viscosityForce = velocity  * ( dragForce);
                                }

                                //apply the viscosity forces
                                f[tri[j]] +=viscosityForce;

                                if (this->m_showViscosityForces.getValue())
                                {
                                    m_debugViscosityForce.push_back(viscosityForce);
                                }
                            }

                            //push back the forces for debug
                            if (this->m_showPressureForces.getValue())
                            {
                                m_debugForce.push_back(pointForcePressure);
                            }
                            if (this->m_showPressureForces.getValue() || this->m_showViscosityForces.getValue())
                            {
                                m_debugPosition.push_back(x[tri[j]]);
                            }
                        }
                    }
                }
            }
        }


    }
//
    d_f.endEdit();
}

template <class DataTypes>
typename BuoyantForceField<DataTypes>::Real BuoyantForceField<DataTypes>::distanceFromFluidSurface(const Deriv &x)
{
    if (fluidModel == AABOX)
    {
        Deriv originToPoint = x - m_fluidSurfaceOrigin;
        Deriv projectedOfPointOnDirection = m_fluidSurfaceDirection * dot(m_fluidSurfaceDirection, originToPoint) / m_fluidSurfaceDirection.norm2();

        return projectedOfPointOnDirection.norm();
    }
    else
    {
        return (dot(m_gravity/ m_gravityNorm, x) + m_heightPlane.getValue());
    }
}

template <class DataTypes>
bool BuoyantForceField<DataTypes>::isPointInFluid(const Coord &x)
{
    if ( fluidModel == AABOX)
    {
        if ( (m_maxBox.getValue() == Coord()) && (m_minBox.getValue() == Coord()) )
            return true;

        return ( (x[0] >= m_minBox.getValue()[0])
                && (x[0] <= m_maxBox.getValue()[0])
                && (x[1] >= m_minBox.getValue()[1])
                && (x[1] <= m_maxBox.getValue()[1])
                && (x[2] >= m_minBox.getValue()[2])
                && (x[2] <= m_maxBox.getValue()[2]) );
        return false;
    }
    else
    {
        //signed distance between the current point and the surface of the fluid
        Real distance = distanceFromFluidSurface(x);

        if ( distance > 0 )
        {
            return true;
        }
        return false;
    }
}

template <class DataTypes>
int BuoyantForceField<DataTypes>::isTriangleInFluid(const Triangle &tri, const VecCoord& x)
{
    int nbPointsInFluid = 0;

    for (unsigned int i = 0 ; i < 3 ; i++)
    {
        if (isPointInFluid(x[ tri[i] ] )  )
        {
            nbPointsInFluid++;
        }
    }

    return nbPointsInFluid;
}

template <class DataTypes>
int BuoyantForceField<DataTypes>::isTetraInFluid(const Tetra &tetra, const VecCoord& x)
{
    int nbPointsInFluid = 0;

    for (unsigned int i = 0 ; i < 4 ; i++)
    {
        if (isPointInFluid(x[ tetra[i] ] )  )
        {
            nbPointsInFluid++;
        }
    }

    return nbPointsInFluid;
}

template <class DataTypes>
bool BuoyantForceField<DataTypes>::isCornerInTetra(const Tetra &tetra, const VecCoord& x) const
{
    if ( fluidModel == AABOX)
    {
        Deriv a = x[tetra[0]];
        Deriv b = x[tetra[ (0 + 1)%4 ]];
        Deriv c = x[tetra[ (0 + 2)%4 ]];
        Deriv d = x[tetra[ (0 + 3)%4 ]];

        Deriv ab = b - a;
        Deriv ac = c - a;
        Deriv ad = d - a;

        for ( int i = 0 ; i < 1 ; i++)
        {
            for (int j = 0 ; j < 1 ; j++)
            {
                for (int k = 0 ; k < 1 ; k++)
                {
                    Deriv corner(
                        i%2 ? m_minBox.getValue()[0] : m_maxBox.getValue()[0],
                        j%2 ? m_minBox.getValue()[1] : m_maxBox.getValue()[1],
                        k%2 ? m_minBox.getValue()[2] : m_maxBox.getValue()[2]
                    );

                    Real c0 = dot(ab.cross(ac), corner - a);
                    Real c1 = dot((c-b).cross(b-d), corner - b);
                    Real c2 = dot((b-a).cross(d-a), corner - a);
                    Real c3 = dot((c-a).cross(d-a), corner - a);

                    if ( c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0 )
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

template <class DataTypes>
typename BuoyantForceField<DataTypes>::Real BuoyantForceField<DataTypes>::getImmersedVolume(const Tetra &tetra, const VecCoord& x)
{
    Real immersedVolume = static_cast<Real>(0.f);

    int nbPointsInside = isTetraInFluid(tetra, x);

    if ( nbPointsInside > 0)
    {
        //the whole tetra is in the fluid
        if ( nbPointsInside == 4)
        {

            int index = m_tetraTopology->getTetrahedronIndex(tetra[0], tetra[1], tetra[2], tetra[3]);
//            Deriv ab = x[tetra[1]] - x[tetra[0]];
//            Deriv ac = x[tetra[2]] - x[tetra[0]];
//            Deriv ad = x[tetra[3]] - x[tetra[0]];

            return m_tetraGeo->computeTetrahedronVolume(index);
        }
        //the tetra is partially in the fluid
//        else if ( nbPointsInside < 4)
//        {
//            int firstPointInside = 0;
//
//            //find the first point which is in the fluid
//            for ( int i = 0 ; i < 4 ; i++)
//            {
//                if (isPointInFluid(x[ tetra[i] ] )  )
//                {
//                    firstPointInside = i;
//                    break;
//                }
//            }
//
//            Deriv a = x[tetra[ firstPointInside ] ];
//            Deriv b = x[tetra[ (firstPointInside + 1)%4 ]];
//            Deriv c = x[tetra[ (firstPointInside + 2)%4 ]];
//            Deriv d = x[tetra[ (firstPointInside + 3)%4 ]];
//
//            //check if a corner of the box (fluid) is inside the tetrahedron
//            if ( isCornerInTetra(tetra, x))
//            {
//                //then a corner of the box (fluid) is in a tetrahedron and it's a annoying
//                return 0.f;
//            }

        return immersedVolume;

//        }
    }

    return immersedVolume;
}

template<class DataTypes>
void BuoyantForceField<DataTypes>::draw()
{
    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT, GL_LINE);

    glDisable(GL_LIGHTING);


    if (m_showBoxOrPlane.getValue())
    {
        Coord min = m_minBox.getValue();
        Coord max = m_maxBox.getValue();

        if ( fluidModel == AABOX)
        {
            //                glEnable (GL_BLEND);
            //                glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glLineWidth(5.0f);
            glBegin(GL_LINE_LOOP);
            glColor4f(0.f, 0.f, 1.0f, 1.f);
            glVertex3d(min[0],min[1],min[2]);
            glVertex3d(min[0],min[1],max[2]);
            glVertex3d(min[0],max[1],max[2]);
            glVertex3d(min[0],max[1],min[2]);
            glEnd();

            glBegin(GL_LINE_LOOP);
            glColor4f(0.f, 1.f, 1.0f, 1.f);
            glVertex3d(min[0],max[1],min[2]);
            glVertex3d(min[0],max[1],max[2]);
            glVertex3d(max[0],max[1],max[2]);
            glVertex3d(max[0],max[1],min[2]);
            glEnd();

            glBegin(GL_LINE_LOOP);
            glColor4f(1.f, 0.f, 1.0f, 1.f);
            glVertex3d(min[0],min[1],max[2]);
            glVertex3d(max[0],min[1],max[2]);
            glVertex3d(max[0],max[1],max[2]);
            glVertex3d(min[0],max[1],max[2]);
            glEnd();

            glBegin(GL_LINE_LOOP);
            glColor4f(1.f, 1.f, 1.0f, 1.f);
            glVertex3d(max[0],min[1],min[2]);
            glVertex3d(max[0],min[1],max[2]);
            glVertex3d(max[0],max[1],max[2]);
            glVertex3d(max[0],max[1],min[2]);
            glEnd();

            glBegin(GL_LINE_LOOP);
            glColor4f(1.f, 0.f, 0.0f, 1.f);
            glVertex3d(min[0],min[1],min[2]);
            glVertex3d(min[0],min[1],max[2]);
            glVertex3d(max[0],min[1],max[2]);
            glVertex3d(max[0],min[1],min[2]);
            glEnd();

            glBegin(GL_LINE_LOOP);
            glColor4f(1.f, 1.f, 0.0f, 1.f);
            glVertex3d(min[0],min[1],min[2]);
            glVertex3d(max[0],min[1],min[2]);
            glVertex3d(max[0],max[1],min[2]);
            glVertex3d(min[0],max[1],min[2]);
            glEnd();

            glColor4f(1.f, 0.f, 0.0f, 1.f);
            glBegin(GL_LINES);
            glVertex3d(m_fluidSurfaceOrigin[0],m_fluidSurfaceOrigin[1],m_fluidSurfaceOrigin[2]);
            glVertex3d(m_fluidSurfaceOrigin[0] +m_fluidSurfaceDirection[0],
                    m_fluidSurfaceOrigin[1] +m_fluidSurfaceDirection[1],
                    m_fluidSurfaceOrigin[2] +m_fluidSurfaceDirection[2]);
            glEnd();

            glColor4f(0.f, 1.f, 1.0f, 1.f);

            glPointSize(10.0f);
            glBegin(GL_POINTS);
            glVertex3d(min[0],min[1],min[2]);
            glVertex3d(max[0],min[1],min[2]);
            glVertex3d(max[0],max[1],min[2]);
            glVertex3d(min[0],max[1],min[2]);
            glVertex3d(min[0],min[1],max[2]);
            glVertex3d(max[0],min[1],max[2]);
            glVertex3d(min[0],max[1],max[2]);
            glVertex3d(max[0],max[1],max[2]);
            glColor4f(1.f, 0.f, 0.0f, 1.f); glPointSize(20.0f);
            glVertex3d(m_fluidSurfaceOrigin[0],m_fluidSurfaceOrigin[1],m_fluidSurfaceOrigin[2]);
            glEnd();
        }
        else if (fluidModel == PLANE)
        {
            Coord n = m_gravity;
            if (n!=Coord())
            {
                Coord firstPoint, secondPoint, thirdPoint, fourthPoint;
                Real largeValue = 1000.0;

                if ( n[2])
                {
                    firstPoint = Coord(largeValue, largeValue, - (m_heightPlane.getValue() + n[0] * largeValue + n[1] * largeValue) / n[2] );
                    secondPoint = Coord(-largeValue, largeValue, - (m_heightPlane.getValue() - n[0] * largeValue + n[1] * largeValue) / n[2] );
                    thirdPoint = Coord(largeValue, -largeValue, - (m_heightPlane.getValue() + n[0] * largeValue - n[1] * largeValue) / n[2] );
                    fourthPoint = Coord(-largeValue, -largeValue, - (m_heightPlane.getValue() - n[0] * largeValue - n[1] * largeValue) / n[2] );
                }
                else if (n[1])
                {
                    firstPoint = Coord(largeValue, largeValue, - (m_heightPlane.getValue() + n[0] * largeValue + n[2] * largeValue) / n[1] );
                    secondPoint = Coord(-largeValue, largeValue, - (m_heightPlane.getValue() - n[0] * largeValue + n[2] * largeValue) / n[1] );
                    thirdPoint = Coord(largeValue, -largeValue, - (m_heightPlane.getValue() + n[0] * largeValue - n[2] * largeValue) / n[1] );
                    fourthPoint = Coord(-largeValue, -largeValue, - (m_heightPlane.getValue() - n[0] * largeValue - n[2] * largeValue) / n[1] );
                }
                else
                {
                    firstPoint = Coord(largeValue, largeValue, - (m_heightPlane.getValue() + n[2] * largeValue + n[1] * largeValue) / n[0] );
                    secondPoint = Coord(-largeValue, largeValue, - (m_heightPlane.getValue() - n[2] * largeValue + n[1] * largeValue) / n[0] );
                    thirdPoint = Coord(largeValue, -largeValue, - (m_heightPlane.getValue() + n[2] * largeValue - n[1] * largeValue) / n[0] );
                    fourthPoint = Coord(-largeValue, -largeValue, - (m_heightPlane.getValue() - n[2] * largeValue - n[1] * largeValue) / n[0] );
                }

                //disable depth test to draw transparency
                glDisable(GL_DEPTH_TEST);

                glEnable (GL_BLEND);
                glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                glBegin(GL_QUADS);
                glColor4f(1.f, 1.f, 1.0f, 0.2f);
                glVertex3d(firstPoint[0],firstPoint[1],firstPoint[2]);
                glVertex3d(thirdPoint[0],thirdPoint[1],thirdPoint[2]);
                glVertex3d(fourthPoint[0],fourthPoint[1],fourthPoint[2]);
                glVertex3d(secondPoint[0],secondPoint[1],secondPoint[2]);
                glEnd();

                glEnable(GL_DEPTH_TEST);

            }
        }
    }
    if ( m_tetraTopology)
    {
        if (this->m_showPressureForces.getValue())
        {
            glColor4f(1.f, 0.f, 0.0f, 1.0f);

            glBegin(GL_LINES);
            for ( unsigned int i = 0 ; i < m_debugPosition.size() ; i++)
            {
                glVertex3d(m_debugPosition[i][0], m_debugPosition[i][1], m_debugPosition[i][2]);
                glVertex3d(m_debugPosition[i][0] - m_debugForce[i][0], m_debugPosition[i][1] -  m_debugForce[i][1], m_debugPosition[i][2] - m_debugForce[i][2]);
            }
            glEnd();
        }
        if (this->m_showViscosityForces.getValue())
        {
            glColor4f(0.f, 1.f, 1.0f, 1.0f);

            glBegin(GL_LINES);
            for ( unsigned int i = 0 ; i < m_debugPosition.size() ; i++)
            {
                glVertex3d(m_debugPosition[i][0], m_debugPosition[i][1], m_debugPosition[i][2]);
                glVertex3d(m_debugPosition[i][0] - m_debugViscosityForce[i][0], m_debugPosition[i][1] -  m_debugViscosityForce[i][1], m_debugPosition[i][2] - m_debugViscosityForce[i][2]);
            }
            glEnd();
        }
    }

    if (this->getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
