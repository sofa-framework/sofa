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
#ifndef SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL

#include <sofa/component/forcefield/OscillatingTorsionPressureForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/TriangleSubsetData.inl>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/helper/gl/template.h>
#include <vector>
#include <set>

// #define DEBUG_TRIANGLEFEM

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace core::topology;


template <class DataTypes> OscillatingTorsionPressureForceField<DataTypes>::~OscillatingTorsionPressureForceField()
{
    //file.close();
}
// Handle topological changes
template <class DataTypes> void  OscillatingTorsionPressureForceField<DataTypes>::handleTopologyChange()
{
    std::list<const TopologyChange *>::const_iterator itBegin=_topology->beginChange();
    std::list<const TopologyChange *>::const_iterator itEnd=_topology->endChange();


    trianglePressureMap.handleTopologyEvents(itBegin,itEnd,_topology->getNbTriangles());

}
template <class DataTypes> void OscillatingTorsionPressureForceField<DataTypes>::init()
{
    //serr << "initializing OscillatingTorsionPressureForceField" << sendl;
    this->core::behavior::ForceField<DataTypes>::init();
    //file.open("testsofa.dat");
    // normalize axis:
    axis.setValue( axis.getValue() / axis.getValue().norm() );

    _topology = this->getContext()->getMeshTopology();

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().length()>0)
    {
        selectTrianglesFromString();
    }

    int numPts = _topology->getNbPoints();
    relMomentToApply.resize( numPts );
    pointActive.resize( numPts );
    vecFromCenter.resize( numPts );
    distFromCenter.resize( numPts );
    momentDir.resize( numPts );
    origVecFromCenter.resize( numPts );
    origCenter.resize( numPts );

    //trianglePressureMap.createTopologicalEngine(_topology);
    trianglePressureMap.setCreateParameter( (void *) this );
    trianglePressureMap.setDestroyParameter( (void *) this );
    //trianglePressureMap.registerTopologicalData();

    initTriangleInformation();

}


template<class DataTypes>
double OscillatingTorsionPressureForceField<DataTypes>::getAmplitude()
{
    double t = this->getContext()->getTime();
    double val = cos( 6.2831853 * frequency.getValue() * t );
    return val;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    Deriv force;
    Coord forceDir, deltaPos;
    Real avgRotAngle = 0;
    Real totalDist = 0;

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    // calculate average rotation angle:
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            vecFromCenter[i] = getVecFromRotAxis( x[i] );
            distFromCenter[i] = vecFromCenter[i].norm();
            if (distFromCenter[i] > 1e-10 && origVecFromCenter[i].norm() > 1e-10)
            {
                avgRotAngle += distFromCenter[i] * getAngle( vecFromCenter[i], origVecFromCenter[i] );
                totalDist += distFromCenter[i];
            }
        }
    avgRotAngle /= totalDist;

    rotationAngle = avgRotAngle;


    //double da = 360.0 / 6.2831853 * rotationAngle;
    //  file <<this->getContext()->getTime() << " " << getAmplitude()*0.01 << " " << avgRotAngle << std::endl;


    // calculate and apply penalty forces to ideal positions
    defaulttype::Quat quat( axis.getValue(), avgRotAngle );
    Real avgError = 0, maxError = 0;
    int pointCnt = 0;
    Real appliedMoment = 0;
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            Coord idealPos = quat.rotate( origVecFromCenter[i] ) + origCenter[i];
            deltaPos = idealPos - x[i];
            force = deltaPos * penalty.getValue();// * 100*deltaPos.norm();
            f[i] += force;
            // get amount of force that is a moment and store
            if (distFromCenter[i] > 1e-10 && origVecFromCenter[i].norm() > 1e-10)
            {
                momentDir[i] = axis.getValue().cross( vecFromCenter[i] ); momentDir[i].normalize();
                appliedMoment += dot( force, momentDir[i] ) * distFromCenter[i];
            }
            // error stats
            Real error = deltaPos.norm();
            if (error > maxError) maxError = error;
            avgError += error;
            pointCnt++;
        }
    avgError /= (Real)pointCnt;
    //std::cout << "  AE = " << avgError << "  ME = " << maxError << "  AM = " << appliedMoment << std::endl;

    // apply remaining moment
    //Real check = 0;
    Real remainingMoment = moment.getValue() * getAmplitude() - appliedMoment;
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            if (distFromCenter[i] > 1e-10)
            {
                force = momentDir[i] * remainingMoment * relMomentToApply[i] / distFromCenter[i];
                //check += force.norm() * distFromCenter[i];
                f[i] += force;
            }
        }
    //std::cout << "RM=" << remainingMoment << "  CHK=" << check << std::endl;
    //std::cout << "  RM = " << remainingMoment << "  ME = " << maxError << "  AM = " << appliedMoment << std::endl;
}

template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::initTriangleInformation()
{
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo;
    this->getContext()->get(triangleGeo);

    const VecCoord *x0 = triangleGeo->getDOF()->getX0();
    int idx[3];
    Real d[10];

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        (*it).second.area=triangleGeo->computeRestTriangleArea((*it).first);
        // calculate distances for corner and intermediate points
        for (int i=0; i<3; i++)
        {
            idx[i] = _topology->getTriangle((*it).first)[i];
            pointActive[idx[i]] = true;
            origVecFromCenter[idx[i]] = getVecFromRotAxis( (*x0)[idx[i]] );
            origCenter[idx[i]] = (*x0)[idx[i]] - origVecFromCenter[idx[i]];
            d[i] = origVecFromCenter[idx[i]].norm();
        }
        d[3] = (d[0]+d[1])/2;
        d[4] = (d[1]+d[2])/2;
        d[5] = (d[2]+d[0])/2;
        d[6] = (d[0]+d[3]+d[5])/3;
        d[7] = (d[1]+d[4]+d[3])/3;
        d[8] = (d[2]+d[5]+d[4])/3;
        d[9] = (d[0]+d[1]+d[2])/3;

        relMomentToApply[idx[0]] += (d[0]*d[0] + d[3]*d[3] + d[5]*d[5] + d[6]*d[6] + d[9]*d[9]) * d[0] * (*it).second.area / 3;
        relMomentToApply[idx[1]] += (d[1]*d[1] + d[4]*d[4] + d[3]*d[3] + d[7]*d[7] + d[9]*d[9]) * d[1] * (*it).second.area / 3;
        relMomentToApply[idx[2]] += (d[2]*d[2] + d[5]*d[5] + d[4]*d[4] + d[8]*d[8] + d[9]*d[9]) * d[2] * (*it).second.area / 3;
    }

    // normalize value to moment 1
    Real totalMoment = 0;
    for (unsigned int i=0; i<relMomentToApply.size(); i++) totalMoment += relMomentToApply[i];
    for (unsigned int i=0; i<relMomentToApply.size(); i++) relMomentToApply[i] /= totalMoment;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = *this->mstate->getX0();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    for (int n=0; n<_topology->getNbTriangles(); ++n)
    {
        if ((vArray[_topology->getTriangle(n)[0]]) && (vArray[_topology->getTriangle(n)[1]])&& (vArray[_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            trianglePressureMap[n]=t;
        }
    }
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesFromString()
{
    std::string inputString=triangleList.getValue();
    unsigned int i;
    do
    {
        const char *str=inputString.c_str();
        for(i=0; (i<inputString.length())&&(str[i]!=','); ++i) ;
        TrianglePressureInformation t;
        t.area = 0;
        if (i==inputString.length())
        {
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i;
        }
        else
        {
            inputString[i]='\0';
            trianglePressureMap[(unsigned int)atoi(str)]=t;
            inputString+=i+1;
        }
    }
    while (inputString.length()>0);
}


template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = *this->mstate->getX();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    typename topology::TriangleSubsetData<TrianglePressureInformation>::iterator it;

    for(it=trianglePressureMap.begin(); it!=trianglePressureMap.end(); it++ )
    {
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[0]]);
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[1]]);
        helper::gl::glVertexT(x[_topology->getTriangle((*it).first)[2]]);
    }
    glEnd();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL
