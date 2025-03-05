/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/mechanicalload/OscillatingTorsionPressureForceField.h>
#include <sofa/core/topology/TopologySubsetData.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <sofa/core/MechanicalParams.h>
#include <vector>
#include <set>

namespace sofa::component::mechanicalload
{

template <class DataTypes>
OscillatingTorsionPressureForceField<DataTypes>::OscillatingTorsionPressureForceField()
    : d_trianglePressureMap(initData(&d_trianglePressureMap, "trianglePressureMap", "Map between triangle indices and their pressure"))
    , d_moment(initData(&d_moment, "moment", "Moment force applied on the entire surface"))
    , d_triangleList(initData(&d_triangleList, "triangleList", "Indices of triangles separated with commas where a pressure is applied"))
    , d_axis(initData(&d_axis, Coord(0, 0, 1), "axis", "Axis of rotation and normal direction for the plane selection of triangles"))
    , d_center(initData(&d_center, "center", "Center of rotation"))
    , d_penalty(initData(&d_penalty, Real(1000), "penalty", "Strength of the penalty force"))
    , d_frequency(initData(&d_frequency, Real(1), "frequency", "frequency of oscillation"))
    , d_dmin(initData(&d_dmin, Real(0.0), "dmin", "Minimum distance from the origin along the normal direction"))
    , d_dmax(initData(&d_dmax, Real(0.0), "dmax", "Maximum distance from the origin along the normal direction"))
    , d_showForces(initData(&d_showForces, (bool)false, "showForces", "draw triangles which have a given pressure"))
    , rotationAngle(0)
{
    trianglePressureMap.setOriginalData(&d_trianglePressureMap);
    moment.setOriginalData(&d_moment);
    triangleList.setOriginalData(&d_triangleList);
    axis.setOriginalData(&d_axis);
    center.setOriginalData(&d_center);
    penalty.setOriginalData(&d_penalty);
    frequency.setOriginalData(&d_frequency);
    dmin.setOriginalData(&d_dmin);
    dmax.setOriginalData(&d_dmax);
    p_showForces.setOriginalData(&d_showForces);

}

template <class DataTypes>
OscillatingTorsionPressureForceField<DataTypes>::~OscillatingTorsionPressureForceField()
{
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::init()
{
    this->core::behavior::ForceField<DataTypes>::init();
    d_axis.setValue(d_axis.getValue() / d_axis.getValue().norm() );

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (d_dmin.getValue() != d_dmax.getValue())
    {
        selectTrianglesAlongPlane();
    }
    if (d_triangleList.getValue().size() > 0)
    {
        selectTrianglesFromString();
    }

    int numPts = m_topology->getNbPoints();
    relMomentToApply.resize( numPts );
    pointActive.resize( numPts );
    vecFromCenter.resize( numPts );
    distFromCenter.resize( numPts );
    momentDir.resize( numPts );
    origVecFromCenter.resize( numPts );
    origCenter.resize( numPts );

    d_trianglePressureMap.createTopologyHandler(m_topology);

    initTriangleInformation();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}


template<class DataTypes>
SReal OscillatingTorsionPressureForceField<DataTypes>::getAmplitude()
{
    SReal t = this->getContext()->getTime();
    const SReal val = cos(6.2831853 * d_frequency.getValue() * t );
    return val;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    if (this->d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    if (pointActive.size() < x.size())
    {
        msg_error() << "The component reads a position size different from the size used in the initialization: cannot continue.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    Deriv force;
    Coord deltaPos;
    Real avgRotAngle = 0;
    Real totalDist = 0;

    // calculate average rotation angle:
    for (unsigned int i=0; i<x.size(); i++)
        if (pointActive[i])
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

    // calculate and apply penalty forces to ideal positions
    const type::Quat<SReal> quat(d_axis.getValue(), avgRotAngle );
    Real avgError = 0, maxError = 0;
    int pointCnt = 0;
    Real appliedMoment = 0;
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            Coord idealPos = quat.rotate( origVecFromCenter[i] ) + origCenter[i];
            deltaPos = idealPos - x[i];
            force = deltaPos * d_penalty.getValue();// * 100*deltaPos.norm();
            f[i] += force;
            // get amount of force that is a moment and store
            if (distFromCenter[i] > 1e-10 && origVecFromCenter[i].norm() > 1e-10)
            {
                momentDir[i] = d_axis.getValue().cross(vecFromCenter[i] ); momentDir[i].normalize();
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
    Real remainingMoment = (Real)(d_moment.getValue() * getAmplitude() - appliedMoment );
    for (unsigned int i=0; i<x.size(); i++) if (pointActive[i])
        {
            if (distFromCenter[i] > 1e-10)
            {
                force = momentDir[i] * remainingMoment * relMomentToApply[i] / distFromCenter[i];
                f[i] += force;
            }
        }
}

template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */)
{
    //TODO: remove this line (avoid warning message) ...
    mparams->setKFactorUsed(true);
}

template <class DataTypes>
SReal OscillatingTorsionPressureForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}

template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix*)
{

}

template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::initTriangleInformation()
{
    const VecCoord& x0 = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    int idx[3];
    Real d[10];

    const sofa::type::vector<Index>& my_map = d_trianglePressureMap.getMap2Elements();
    sofa::type::vector<TrianglePressureInformation>& my_subset = *(d_trianglePressureMap).beginEdit();

    for (unsigned int i=0; i<my_map.size(); ++i)
    {
        const auto& t = this->m_topology->getTriangle(my_map[i]);

        const auto& n0 = DataTypes::getCPos(x0[t[0]]);
        const auto& n1 = DataTypes::getCPos(x0[t[1]]);
        const auto& n2 = DataTypes::getCPos(x0[t[2]]);

        my_subset[i].area = sofa::geometry::Triangle::area(n0, n1, n2);
        // calculate distances for corner and intermediate points
        for (int j=0; j<3; j++)
        {
            idx[j] = t[j];
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

    d_trianglePressureMap.endEdit();

    return;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesAlongPlane()
{
    const VecCoord& x = this->mstate->read(core::vec_id::read_access::restPosition)->getValue();
    std::vector<bool> vArray;

    vArray.resize(x.size());

    for( unsigned int i=0; i<x.size(); ++i)
    {
        vArray[i]=isPointInPlane(x[i]);
    }

    sofa::type::vector<TrianglePressureInformation>& my_subset = *(d_trianglePressureMap).beginEdit();
    type::vector<Index> inputTriangles;

    for (size_t n=0; n<m_topology->getNbTriangles(); ++n)
    {
        if ((vArray[m_topology->getTriangle(n)[0]]) && (vArray[m_topology->getTriangle(n)[1]])&& (vArray[m_topology->getTriangle(n)[2]]) )
        {
            // insert a dummy element : computation of pressure done later
            TrianglePressureInformation t;
            t.area = 0;
            my_subset.push_back(t);
            inputTriangles.push_back(n);
        }
    }
    d_trianglePressureMap.endEdit();
    d_trianglePressureMap.setMap2Elements(inputTriangles);

    return;
}


template <class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::selectTrianglesFromString()
{
    sofa::type::vector<TrianglePressureInformation>& my_subset = *(d_trianglePressureMap).beginEdit();
    type::vector<Index> _triangleList = d_triangleList.getValue();

    d_trianglePressureMap.setMap2Elements(_triangleList);

    for (unsigned int i = 0; i < _triangleList.size(); ++i)
    {
        TrianglePressureInformation t;
        t.area = 0;
        my_subset.push_back(t);
    }

    d_trianglePressureMap.endEdit();

    return;
}


template<class DataTypes>
void OscillatingTorsionPressureForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (!d_showForces.getValue())
        return;

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    vparams->drawTool()->disableLighting();
    const sofa::type::RGBAColor color = sofa::type::RGBAColor::green();
    std::vector<sofa::type::Vec3> vertices;

    const sofa::type::vector<Index>& my_map = d_trianglePressureMap.getMap2Elements();

    for (unsigned int i = 0; i < my_map.size(); ++i)
    {
        for(unsigned int j=0 ; j< 3 ; j++)
        {
            const Coord& c = x[m_topology->getTriangle(my_map[i])[j]];
            vertices.push_back(sofa::type::Vec3(c[0], c[1], c[2]));
        }
    }
    vparams->drawTool()->drawTriangles(vertices, color);

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);


}

template<class DataTypes>
bool OscillatingTorsionPressureForceField<DataTypes>::isPointInPlane(Coord p)
{
    Real d=dot(p, d_axis.getValue());
    if ((d > d_dmin.getValue()) && (d < d_dmax.getValue()))
        return true;
    else
        return false;
}

template<class DataTypes>
typename OscillatingTorsionPressureForceField<DataTypes>::Coord OscillatingTorsionPressureForceField<DataTypes>::getVecFromRotAxis( const Coord &x )
{
    Coord vecFromCenter = x - d_center.getValue();
    Coord axisProj = d_axis.getValue() * dot(vecFromCenter, d_axis.getValue() ) + d_center.getValue();
    return (x - axisProj);
}

template<class DataTypes>
typename OscillatingTorsionPressureForceField<DataTypes>::Real OscillatingTorsionPressureForceField<DataTypes>::getAngle( const Coord &v1, const Coord &v2 )
{
    Real dp = dot( v1, v2 ) / (v1.norm()*v2.norm());
    if (dp>1.0) dp=1.0; else if (dp<-1.0) dp=-1.0;
    Real angle = acos( dp );
    // check direction!
    if (dot(d_axis.getValue(), v1.cross(v2 ) ) > 0) angle *= -1;
    return angle;
}

} // namespace sofa::component::mechanicalload
