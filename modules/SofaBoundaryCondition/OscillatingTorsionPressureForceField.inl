/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL

#include <SofaBoundaryCondition/OscillatingTorsionPressureForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
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


template <class DataTypes> OscillatingTorsionPressureForceField<DataTypes>::~OscillatingTorsionPressureForceField()
{
    //file.close();
}


template <class DataTypes> void OscillatingTorsionPressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    //file.open("testsofa.dat");
    // normalize axis:
    axis.setValue( axis.getValue() / axis.getValue().norm() );

    _topology = this->getContext()->getMeshTopology();

    if (dmin.getValue()!=dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (triangleList.getValue().size()>0)
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

    trianglePressureMap.createTopologicalEngine(_topology);
    trianglePressureMap.registerTopologicalData();

    initTriangleInformation();
}


template<class DataTypes>
SReal OscillatingTorsionPressureForceField<DataTypes>::getAmplitude()
{
    SReal t = this->getContext()->getTime();
    SReal val = cos( 6.2831853 * frequency.getValue() * t );
    return val;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    Deriv force;
    Coord deltaPos;
    Real avgRotAngle = 0;
    Real totalDist = 0;

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


    //SReal da = 360.0 / 6.2831853 * rotationAngle;
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

    // apply remaining moment
    //Real check = 0;
    Real remainingMoment = (Real)( moment.getValue() * getAmplitude() - appliedMoment );
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            if (distFromCenter[i] > 1e-10)
            {
                force = momentDir[i] * remainingMoment * relMomentToApply[i] / distFromCenter[i];
                //check += force.norm() * distFromCenter[i];
                f[i] += force;
            }
        }
}


template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::initTriangleInformation()
{
    sofa::component::topology::TriangleSetGeometryAlgorithms<DataTypes>* triangleGeo;
    this->getContext()->get(triangleGeo);

    const VecCoord x0 = triangleGeo->getDOF()->read(core::ConstVecCoordId::restPosition())->getValue();
    int idx[3];
    Real d[10];

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        my_subset[i].area=triangleGeo->computeRestTriangleArea(my_map[i]);
        // calculate distances for corner and intermediate points
        for (int j=0; j<3; j++)
        {
            idx[j] = _topology->getTriangle(my_map[i])[j];
            pointActive[idx[j]] = true;
            origVecFromCenter[idx[j]] = getVecFromRotAxis( (x0)[idx[j]] );
            origCenter[idx[j]] = (x0)[idx[j]] - origVecFromCenter[idx[j]];
            d[j] = origVecFromCenter[idx[j]].norm();
        }
        d[3] = (d[0]+d[1])/2;
        d[4] = (d[1]+d[2])/2;
        d[5] = (d[2]+d[0])/2;
        d[6] = (d[0]+d[3]+d[5])/3;
        d[7] = (d[1]+d[4]+d[3])/3;
        d[8] = (d[2]+d[5]+d[4])/3;
        d[9] = (d[0]+d[1]+d[2])/3;

        relMomentToApply[idx[0]] += (d[0]*d[0] + d[3]*d[3] + d[5]*d[5] + d[6]*d[6] + d[9]*d[9]) * d[0] * my_subset[i].area / 3;
        relMomentToApply[idx[1]] += (d[1]*d[1] + d[4]*d[4] + d[3]*d[3] + d[7]*d[7] + d[9]*d[9]) * d[1] * my_subset[i].area / 3;
        relMomentToApply[idx[2]] += (d[2]*d[2] + d[5]*d[5] + d[4]*d[4] + d[8]*d[8] + d[9]*d[9]) * d[2] * my_subset[i].area / 3;
    }

    // normalize value to moment 1
    Real totalMoment = 0;
    for (unsigned int i=0; i<relMomentToApply.size(); i++) totalMoment += relMomentToApply[i];
    for (unsigned int i=0; i<relMomentToApply.size(); i++) relMomentToApply[i] /= totalMoment;

    trianglePressureMap.endEdit();

    return;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    std::vector<bool> vArray;
    unsigned int i;

    vArray.resize(x.size());

    for( i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> inputTriangles;

    for (int n=0; n<_topology->getNbTriangles(); ++n)
    {
        if ((vArray[_topology->getTriangle(n)[0]]) && (vArray[_topology->getTriangle(n)[1]])&& (vArray[_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            my_subset.push_back(t);
            inputTriangles.push_back(n);
        }
    }
    trianglePressureMap.endEdit();
    trianglePressureMap.setMap2Elements(inputTriangles);

    return;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesFromString()
{
    sofa::helper::vector<TrianglePressureInformation>& my_subset = *(trianglePressureMap).beginEdit();
    helper::vector<unsigned int> _triangleList = triangleList.getValue();

    trianglePressureMap.setMap2Elements(_triangleList);

    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        TrianglePressureInformation t;
        t.area = 0;
        my_subset.push_back(t);
    }

    trianglePressureMap.endEdit();

    return;
}


template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!p_showForces.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    glDisable(GL_LIGHTING);

    glBegin(GL_TRIANGLES);
    glColor4f(0,1,0,1);

    const sofa::helper::vector <unsigned int>& my_map = trianglePressureMap.getMap2Elements();

    for (unsigned int i = 0; i < my_map.size(); ++i)
    {
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[0]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[1]]);
        helper::gl::glVertexT(x[_topology->getTriangle(my_map[i])[2]]);
    }
    glEnd();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif /* SOFA_NO_OPENGL */
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_INL
