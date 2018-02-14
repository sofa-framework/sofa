/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL

#include <SofaBoundaryCondition/BuoyantForceField.h>
#include <sofa/core/visual/VisualParams.h>
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
    m_showBoxOrPlane(initData(&m_showBoxOrPlane, false, "showBoxOrPlane", "Show the box or the plane")),
    m_showFactorSize(initData(&m_showFactorSize, (Real)1.0, "showFactorSize", "Size factor applied to shown forces"))
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

            msg_info() << " fluid is modeled now with a box" ;
        }
    }
    else
    {
        if (fluidModel != PLANE)
        {
            fluidModel = PLANE;
            change = true;
            msg_info() <<  " fluid is modeled now with a plane" ;
        }
    }

    if ( m_minBoxPrev != m_minBox.getValue() ||  m_maxBoxPrev!= m_maxBox.getValue())
    {
        Coord tempMin = m_minBox.getValue() , tempMax = m_maxBox.getValue();

        for ( unsigned int i = 0 ; i < 3 ; i++)
            if (tempMin[i] > tempMax[i])
            {
                sout << "Switch value " << i << " between min and max" << sendl;
                tempMin[i] = m_maxBox.getValue()[i];
                tempMax[i] = m_minBox.getValue()[i];
            }
        m_minBoxPrev = tempMin;
        m_maxBoxPrev = tempMax;

        m_minBox.setValue(tempMin);
        m_maxBox.setValue(tempMax);

        recomputeFluidSurface = true;
        change = true;

        msg_info() << " change bounding box: <" <<  m_minBox.getValue() << "> - <" << m_maxBox.getValue() << ">" ;
    }


    if (m_gravity!= this->getContext()->getGravity())
    {
        m_gravity = this->getContext()->getGravity();
        m_gravityNorm = m_gravity.norm();
        recomputeFluidSurface = true;
        change = true;

        msg_info() << " new gravity : " << m_gravity ;
    }

    if (recomputeFluidSurface)
    {
        //the surface in a box is the upper face defined by the gravity
        //it's the one with the minimum angle between the normal and the gravity

        if (!m_gravityNorm)
        {
            //TODO(dmarchal) can someone explaine what is the consequence and how to get rid of this message.
            msg_warning() << " unable to determine fluid surface because there is no gravity" ;
        }
        else
        {
            int dir=1;

            if(fabs(m_gravity[0])>fabs(m_gravity[1]) && fabs(m_gravity[0])>fabs(m_gravity[2])) dir=(m_gravity[0]>0)?1:-1;
            else if(fabs(m_gravity[1])>fabs(m_gravity[0]) && fabs(m_gravity[1])>fabs(m_gravity[2])) dir=(m_gravity[1]>0)?2:-2;
            else if(fabs(m_gravity[2])>fabs(m_gravity[0]) && fabs(m_gravity[2])>fabs(m_gravity[1])) dir=(m_gravity[2]>0)?3:-3;

            switch(dir)
            {
            case -1:  m_fluidSurfaceDirection = Coord((Real)1.0,(Real)0.0,(Real)0.0);  m_fluidSurfaceOrigin = m_maxBox.getValue();  break;
            case 1: m_fluidSurfaceDirection = Coord((Real)-1.0,(Real)0.0,(Real)0.0);  m_fluidSurfaceOrigin = m_minBox.getValue();  break;
            case -2:  m_fluidSurfaceDirection = Coord((Real)0.0,(Real)1.0,(Real)0.0);  m_fluidSurfaceOrigin = m_maxBox.getValue();  break;
            case 2: m_fluidSurfaceDirection = Coord((Real)0.0,(Real)-1.0,(Real)0.0);  m_fluidSurfaceOrigin = m_minBox.getValue();  break;
            case -3:  m_fluidSurfaceDirection = Coord((Real)0.0,(Real)0.0,(Real)1.0);  m_fluidSurfaceOrigin = m_maxBox.getValue();  break;
            case 3: m_fluidSurfaceDirection = Coord((Real)0.0,(Real)0.0,(Real)-1.0);  m_fluidSurfaceOrigin = m_minBox.getValue();  break;
            }
        }
    }
    return change;
}

template <class DataTypes>
void BuoyantForceField<DataTypes>::init()
{
    checkParameters();

    this->core::behavior::ForceField<DataTypes>::init();
    this->getContext()->get(m_topology);

    if (!m_topology)
    {
        msg_warning() << " missing mesh topology" ;
        return;
    }

    msg_info() << " coupling with " << m_topology->getName();
    msg_info_when(m_flipNormals.getValue())<< " normals are flipped to inverse the forces" ;

    //TODO(dmarchal): can someone explaine what is the consequence and they way to fix the problem.
    msg_warning_when(m_fluidDensity.getValue() <= 0.f) << " the density of the fluid is negative." ;

    //get all the surfacic triangles from the topology
    m_triangles.clear();
    const seqTriangles &triangleArray=m_topology->getTriangles();
    for (unsigned int i=0; i<triangleArray.size(); ++i)
        if (m_topology->getTetrahedraAroundTriangle(i).size()<=1)
            m_triangles.push_back(i);

    std::stringstream buffer;
    buffer << " there are " << triangleArray.size()<< " triangles in the topology.  " << std::endl ;
    buffer << " there are " << m_triangles.size() << " triangles on the surface of the topology." << std::endl;
    msg_info() << buffer.str() ;
}



template <class DataTypes>
void BuoyantForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    if (!m_topology) return;
    if (!m_triangles.size()) return;
    if (!m_gravityNorm) return;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    checkParameters();

    m_showForce.clear();
    m_showPosition.clear();
    m_showViscosityForce.clear();

    for (unsigned int i = 0 ; i < m_triangles.size() ; i++)
    {
        const ID triangleID = m_triangles[i];
        const Triangle tri = m_topology->getTriangle(triangleID);

        Coord centreTriangle =  (x[tri[0]] + x[tri[1]] + x[tri[2]]) / (Real) 3.0;

        if (isPointInFluid(centreTriangle))
        {
            //get triangle attributes
            Coord normalArea = (x[tri[1]]-x[tri[0]]).cross( x[tri[2]]-x[tri[0]])*(Real)0.5; ;
            if (m_flipNormals.getValue())				normalArea = -normalArea;

            //get the distance between the centroid and the surface of the fluid
            Real z = (distanceFromFluidSurface(centreTriangle) );
            //the pressure applied by the fluid on the current triangle
            Real pressure = m_atmosphericPressure.getValue() + m_fluidDensity.getValue() * m_gravityNorm * z;
            //the force acting on the triangle due to the pressure
            Coord triangleForcePressure = normalArea * pressure ;
            //the force acting on the points of the triangle due to the pressure
            Coord pointForcePressure = triangleForcePressure / (Real)3.0;

            //apply force
            for ( int j = 0 ; j < 3 ; j++)
            {
                f[tri[j]] += pointForcePressure;
                if (this->m_showPressureForces.getValue())		m_showForce.push_back(pointForcePressure);
                if (this->m_showPressureForces.getValue() || this->m_showViscosityForces.getValue())	m_showPosition.push_back(x[tri[j]]);
            }

            // viscosity Forces
            if ( m_enableViscosity.getValue())
            {
                Real dragForce;

                if (m_turbulentFlow.getValue())		dragForce = - (Real)0.5f * m_fluidDensity.getValue() * normalArea.norm();
                else //laminar flow
                {
                    //Coord circumcenter = m_geo->computeTriangleCircumcenter(triangleID);
                    //Coord firstCorner = x[tri[0]];
                    //Real circumradius = (circumcenter - firstCorner).norm();

                    Real	a = (x[tri[1]]-x[tri[0]]).norm() ,	b = (x[tri[2]]-x[tri[0]]).norm() ,	c = (x[tri[1]]-x[tri[2]]).norm() ;
                    Real circumradius = a*b*c / sqrt( (a+b+c)*(a+b-c)*(a-b+c)*(-a+b+c) ) ;

                    dragForce = - (Real)6.0 * (Real)M_PI * m_fluidViscosity.getValue() * circumradius;
                }

                //apply force
                Deriv viscosityForce , velocity;
                for ( int j = 0 ; j < 3 ; j++)
                {
                    velocity = v[tri[j]];
                    if (m_turbulentFlow.getValue())		viscosityForce = velocity  * ( dragForce * velocity.norm() );
                    else								viscosityForce = velocity  * ( dragForce);
                    f[tri[j]] +=viscosityForce;
                    if (this->m_showViscosityForces.getValue()) m_showViscosityForce.push_back(viscosityForce);
                }

            }
        }
    }

    d_f.endEdit();
}

template <class DataTypes>
void BuoyantForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  d_df , const DataVecDeriv&  d_dx )
{
    if (!m_topology) return;

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    Real kf = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    for (unsigned int i = 0 ; i < m_triangles.size() ; i++)
    {
        const ID triangleID = m_triangles[i];
        const Triangle tri = m_topology->getTriangle(triangleID);

        Coord centreTriangle =  (x[tri[0]] + x[tri[1]] + x[tri[2]]) / (Real) 3.0;
        if (isPointInFluid(centreTriangle))
        {
            Coord normalArea = (x[tri[1]]-x[tri[0]]).cross( x[tri[2]]-x[tri[0]])*(Real)0.5;
            if (m_flipNormals.getValue())				normalArea = -normalArea;
            Deriv DcentreTriangle =  (dx[tri[0]] + dx[tri[1]] + dx[tri[2]]) / (Real) 3.0;
            Deriv dpointForcePressure = normalArea * m_fluidDensity.getValue() * m_gravityNorm * D_distanceFromFluidSurface(DcentreTriangle) / (Real)3.0;

            // BG. adding this term is less stable due to non linearity ?
            //Coord dnormalArea = ( (dx[tri[1]]-dx[tri[0]]).cross( x[tri[2]]-x[tri[0]]) + (x[tri[1]]-x[tri[0]]).cross( dx[tri[2]]-dx[tri[0]]) )*(Real)0.5;
            //dpointForcePressure += dnormalArea * (m_atmosphericPressure.getValue() + m_fluidDensity.getValue() * m_gravityNorm * distanceFromFluidSurface(centreTriangle) )  / (Real)3.0;

            for ( int j = 0 ; j < 3 ; j++) 	 df[tri[j]] += dpointForcePressure * kf;
        }
    }

    d_df.endEdit();
}



template <class DataTypes>
typename BuoyantForceField<DataTypes>::Real BuoyantForceField<DataTypes>::distanceFromFluidSurface(const Coord &x)
{
    if (fluidModel == AABOX)    return -dot(m_fluidSurfaceDirection, x - m_fluidSurfaceOrigin);
    else	return (dot(m_gravity/ m_gravityNorm, x) + m_heightPlane.getValue());
}


template <class DataTypes>
typename BuoyantForceField<DataTypes>::Real BuoyantForceField<DataTypes>::D_distanceFromFluidSurface(const Deriv &dx)
{
    if (fluidModel == AABOX) return -dot(m_fluidSurfaceDirection, dx);
    else     return dot(m_gravity/ m_gravityNorm, dx) ;
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



template<class DataTypes>
void BuoyantForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!this->mstate) return;

    vparams->drawTool()->saveLastState();
    vparams->drawTool()->setPolygonMode(1, vparams->displayFlags().getShowWireFrame());
    vparams->drawTool()->disableLighting();

    if (m_showBoxOrPlane.getValue())
    {
        Coord min = m_minBox.getValue();
        Coord max = m_maxBox.getValue();

        if ( fluidModel == AABOX)
        {
            vparams->drawTool()->drawLineLoop({
                {min[0], min[1], min[2]},
                {min[0], min[1], max[2]},
                {min[0], max[1], max[2]},
                {min[0], max[1], min[2]}},
                5.0f, {0.f, 0.f, 1.0f, 1.f});

            vparams->drawTool()->drawLineLoop({
                {min[0], max[1], min[2]},
                {min[0], max[1], max[2]},
                {max[0], max[1], max[2]},
                {max[0], max[1], min[2]}},
                5.0f, {0.f, 1.f, 1.0f, 1.f});

            vparams->drawTool()->drawLineLoop(
                {{min[0], min[1], max[2]},
                {max[0], min[1], max[2]},
                {max[0], max[1], max[2]},
                {min[0], max[1], max[2]}},
                5.f, {1.f, 0.f, 1.0f, 1.f});

            vparams->drawTool()->drawLineLoop(
                {{max[0], min[1], min[2]},
                {max[0], min[1], max[2]},
                {max[0], max[1], max[2]},
                {max[0], max[1], min[2]}},
                5.0f, {1.f, 1.f, 1.0f, 1.f});

            vparams->drawTool()->drawLineLoop(
                {{min[0],min[1],min[2]},
                {min[0],min[1],max[2]},
                {max[0],min[1],max[2]},
                {max[0],min[1],min[2]}},
                0.5f, {1.f, 0.f, 0.0f, 1.f});

            vparams->drawTool()->drawLineLoop(
                {{min[0],min[1],min[2]},
                {max[0],min[1],min[2]},
                {max[0],max[1],min[2]},
                {min[0],max[1],min[2]}},
                5.0f, {1.f, 1.f, 0.0f, 1.f});

            vparams->drawTool()->drawLines(
                {{m_fluidSurfaceOrigin[0],m_fluidSurfaceOrigin[1],m_fluidSurfaceOrigin[2]},
                {m_fluidSurfaceOrigin[0] +m_fluidSurfaceDirection[0],
                 m_fluidSurfaceOrigin[1] +m_fluidSurfaceDirection[1],
                 m_fluidSurfaceOrigin[2] +m_fluidSurfaceDirection[2]}},
                5.f, {1.f, 0.f, 0.0f, 1.f});


            vparams->drawTool()->drawPoints(
                {{min[0],min[1],min[2]},
                {max[0],min[1],min[2]},
                {max[0],max[1],min[2]},
                {min[0],max[1],min[2]},
                {min[0],min[1],max[2]},
                {max[0],min[1],max[2]},
                {min[0],max[1],max[2]},
                {max[0],max[1],max[2]}},
                10.0f, {0.f, 1.f, 1.0f, 1.f});

            vparams->drawTool()->drawPoints({{m_fluidSurfaceOrigin[0],m_fluidSurfaceOrigin[1],m_fluidSurfaceOrigin[2]}},
                                            20.0f, {1.f, 0.f, 0.0f, 1.f});
        }
        else if (fluidModel == PLANE)
        {
            if (m_gravityNorm)
            {
                Coord firstPoint, secondPoint, thirdPoint, fourthPoint;
                Real largeValue = 1000.0;

                int dir=1;

                if(fabs(m_gravity[0])>fabs(m_gravity[1]) && fabs(m_gravity[0])>fabs(m_gravity[2])) dir=(m_gravity[0]>0)?1:-1;
                else if(fabs(m_gravity[1])>fabs(m_gravity[0]) && fabs(m_gravity[1])>fabs(m_gravity[2])) dir=(m_gravity[1]>0)?2:-2;
                else if(fabs(m_gravity[2])>fabs(m_gravity[0]) && fabs(m_gravity[2])>fabs(m_gravity[1])) dir=(m_gravity[2]>0)?3:-3;

                switch(dir)
                {
                case 1:
                    firstPoint =  Coord(- m_heightPlane.getValue(), largeValue, largeValue    );
                    secondPoint = Coord(- m_heightPlane.getValue(),  -largeValue, largeValue  );
                    thirdPoint =  Coord(- m_heightPlane.getValue(), -largeValue, -largeValue  );
                    fourthPoint = Coord(- m_heightPlane.getValue(), largeValue, -largeValue  );
                    break;
                case -1:
                    firstPoint =  Coord( m_heightPlane.getValue(), largeValue, largeValue    );
                    secondPoint = Coord( m_heightPlane.getValue(),  -largeValue, largeValue  );
                    thirdPoint =  Coord( m_heightPlane.getValue(), -largeValue, -largeValue  );
                    fourthPoint = Coord( m_heightPlane.getValue(), largeValue, -largeValue  );
                    break;
                case 2:
                    firstPoint = Coord(largeValue  , - m_heightPlane.getValue(), largeValue   );
                    secondPoint = Coord(-largeValue, - m_heightPlane.getValue(), largeValue );
                    thirdPoint = Coord(-largeValue , - m_heightPlane.getValue(), -largeValue );
                    fourthPoint = Coord(largeValue , - m_heightPlane.getValue(), -largeValue );
                    break;
                case -2:
                    firstPoint = Coord(largeValue  ,  m_heightPlane.getValue(), largeValue   );
                    secondPoint = Coord(-largeValue,  m_heightPlane.getValue(), largeValue );
                    thirdPoint = Coord(-largeValue ,  m_heightPlane.getValue(), -largeValue );
                    fourthPoint = Coord(largeValue ,  m_heightPlane.getValue(), -largeValue );
                    break;
                case 3:
                    firstPoint = Coord(largeValue, largeValue, - m_heightPlane.getValue() );
                    secondPoint = Coord(-largeValue, largeValue, - m_heightPlane.getValue() );
                    thirdPoint = Coord(-largeValue, -largeValue, - m_heightPlane.getValue() );
                    fourthPoint = Coord(largeValue, -largeValue, - m_heightPlane.getValue() );
                    break;
                case -3:
                    firstPoint = Coord(largeValue, largeValue,  m_heightPlane.getValue() );
                    secondPoint = Coord(-largeValue, largeValue,  m_heightPlane.getValue() );
                    thirdPoint = Coord(-largeValue, -largeValue,  m_heightPlane.getValue() );
                    fourthPoint = Coord(largeValue, -largeValue,  m_heightPlane.getValue() );
                    break;
                }

                //disable depth test to draw transparency
                vparams->drawTool()->disableDepthTest();
                vparams->drawTool()->enableBlending();

                vparams->drawTool()->drawQuads(
                {{firstPoint[0],firstPoint[1],firstPoint[2]},
                 {secondPoint[0],secondPoint[1],secondPoint[2]},
                 {thirdPoint[0],thirdPoint[1],thirdPoint[2]},
                 {fourthPoint[0],fourthPoint[1],fourthPoint[2]}}, {1.f, 1.f, 1.0f, 0.2f});

                vparams->drawTool()->enableDepthTest();
            }
        }
    }


    if (vparams->displayFlags().getShowForceFields())
        if ( m_topology)
        {
            if (this->m_showPressureForces.getValue())
            {
                sofa::helper::vector<sofa::defaulttype::Vector3> lines;

                for ( unsigned int i = 0 ; i < m_showPosition.size() ; i++)
                {
                    lines.push_back({m_showPosition[i][0], m_showPosition[i][1], m_showPosition[i][2]});
                    lines.push_back({m_showPosition[i][0] - m_showForce[i][0]*m_showFactorSize.getValue(), m_showPosition[i][1] -  m_showForce[i][1]*m_showFactorSize.getValue(), m_showPosition[i][2] - m_showForce[i][2]*m_showFactorSize.getValue()});
                }

                vparams->drawTool()->drawLines(lines, 1.f, {1.f, 0.f, 0.0f, 1.0f});
            }
            if (this->m_showViscosityForces.getValue())
            {
                sofa::helper::vector<sofa::defaulttype::Vector3> lines;

                for ( unsigned int i = 0 ; i < m_showPosition.size() ; i++)
                {
                    lines.push_back({m_showPosition[i][0], m_showPosition[i][1], m_showPosition[i][2]});
                    lines.push_back({m_showPosition[i][0] - m_showViscosityForce[i][0]*m_showFactorSize.getValue(), m_showPosition[i][1] -  m_showViscosityForce[i][1]*m_showFactorSize.getValue(), m_showPosition[i][2] - m_showViscosityForce[i][2]*m_showFactorSize.getValue()});
                }
                vparams->drawTool()->drawLines(lines, 1.f, {0.f, 1.f, 1.0f, 1.0f});
            }
        }

    vparams->drawTool()->restoreLastState();
}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_INL
