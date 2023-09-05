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

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/component/mechanicalload/PlaneForceField.h>
#include <sofa/helper/accessor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <cassert>
#include <iostream>
#include <sofa/type/BoundingBox.h>
#include <limits>
#include <sofa/core/behavior/BaseLocalForceFieldMatrix.h>
#include <sofa/simulation/Node.h>

namespace
{
    template<typename DataTypes>
    void get3DFrameFromDPosNormal(const typename DataTypes::DPos& dposnormal, sofa::type::Vec3& v1, sofa::type::Vec3& v2, sofa::type::Vec3& normal)
    {
        for (std::size_t i = 0; i < dposnormal.size() && i < 3u; i++)
            normal[i] = dposnormal[i];

        // find a first vector inside the plane
        if (std::abs(normal[0]) > 0.0)
        {
            v1 = sofa::type::Vec3(-normal[1] / normal[0], 1.0, 0.0);
        }
        else if (std::abs(normal[1]) > 0.0)
        {
            v1 = sofa::type::Vec3(1.0, -normal[0] / normal[1], 0.0);
        }
        else if (std::abs(normal[2]) > 0.0)
        {
            v1 = sofa::type::Vec3(1.0, 0.0, -normal[0] / normal[2]);
        }

        v1.normalize();

        // find a second vector inside the plane and orthogonal to the first
        v2 = v1.cross(normal);
        v2.normalize();
    }
}

namespace sofa::component::mechanicalload
{

using sofa::core::objectmodel::ComponentState ;
using namespace sofa::type; 
using sofa::type::Vec ;

template<class DataTypes>
PlaneForceField<DataTypes>::PlaneForceField() :
     d_planeNormal(initData(&d_planeNormal, "normal", "plane normal. (default=[0,1,0])"))
    , d_planeD(initData(&d_planeD, (Real)0, "d", "plane d coef. (default=0)"))
    , d_stiffness(initData(&d_stiffness, (Real)500, "stiffness", "force stiffness. (default=500)"))
    , d_damping(initData(&d_damping, (Real)5, "damping", "force damping. (default=5)"))
    , d_maxForce(initData(&d_maxForce, (Real)0, "maxForce", "if non-null , the max force that can be applied to the object. (default=0)"))
    , d_bilateral( initData(&d_bilateral, false, "bilateral", "if true the plane force field is applied on both sides. (default=false)"))
    , d_localRange( initData(&d_localRange, type::Vec<2,int>(-1,-1), "localRange", "optional range of local DOF indices. Any computation involving indices outside of this range are discarded (useful for parallelization using mesh partitionning)" ) )
    , d_drawIsEnabled(initData(&d_drawIsEnabled, false, "showPlane", "enable/disable drawing of plane. (default=false)"))
    , d_drawColor(initData(&d_drawColor, sofa::type::RGBAColor(0.0f,.5f,.2f,1.0f), "planeColor", "plane color. (default=[0.0,0.5,0.2,1.0])"))
    , d_drawSize(initData(&d_drawSize, (Real)10.0f, "showPlaneSize", "plane display size if draw is enabled. (default=10)"))
{
    Deriv n;
    DataTypes::set(n, 0, 1, 0);
    d_planeNormal.setValue(DataTypes::getDPos(n));
}

template<class DataTypes>
void PlaneForceField<DataTypes>::init(){
    if(this->d_componentState.getValue() == ComponentState::Valid){
        msg_warning(this) << "Calling an already fully initialized component.  You should use reinit instead." ;
    }

    Inherit::init() ;
    if( this->mstate == nullptr ){
        msg_error(this) << "Missing mechanical object.  This component will be considered as not valid and will do nothing.  "
                        << "To remove this error message you need to set a <MechanicalObject> in the context of this component."  ;
    }

    if( d_stiffness.getValue() < 0.0 ){
        msg_warning(this) << "The 'stiffness="<< d_stiffness.getValueString() << "' parameters is outside the validity range of [0, +INF[.  Continuing with the default value=500.0 .  "
                             "To remove this warning message you need to set the 'stiffness' attribute between [0, +INF[."
                             "  Emitted from ["<< this->getPathName() << "].";
        d_stiffness.setValue(Real(500)) ;
    }
    if( d_damping.getValue() < 0.0 ){
        msg_warning(this) << "The 'damping="<< d_damping.getValueString() <<"' parameters is outside the validity range of [0, +INF[.  Continuing with the default value=5.0 .  "
                             "To remove this warning message you need to set the 'damping' attribute between [0, +INF[." ;
        d_damping.setValue(Real(5)) ;
    }
    if( d_maxForce.getValue() < 0.0 ){
        msg_warning(this) << "The 'maxForce="<< d_maxForce.getValueString() << "' parameters is outside the validity range of [0, +INF[.  Continuing with the default value=0.0 (no max force).  "
                             "To remove this warning message you need to set the 'maxForce' attribute between [0, +INF[." ;
        d_maxForce.setValue(Real(0)) ;
    }

    Vec<2,int> tmp = d_localRange.getValue() ;
    if( d_localRange.isSet() && (tmp.x() < 0 || tmp.y() < 0 || tmp.x() > tmp.y()) ){
        msg_warning(this) << "The 'localRange="<< d_localRange.getValueString() << "' parameter is not valid as it needs two indices in numerical order.  "
                             "Continuing with the default value=[0, 0] (no local range).  "
                             "To remove this warning message you need to set the 'localRange' to correct value." ;

        tmp.set(-1,-1);
        d_localRange.setValue(tmp) ;
    }



    this->d_componentState.setValue(ComponentState::Valid) ;
}

template<class DataTypes>
void PlaneForceField<DataTypes>::setPlane(const Deriv& normal, Real d)
{
    DPos tmpN = DataTypes::getDPos(normal);
    Real n = tmpN.norm();
    d_planeNormal.setValue( tmpN / n);
    d_planeD.setValue( d / n );
}


template<class DataTypes>
SReal PlaneForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/,
                                                     const DataVecCoord&  /* x */) const
{
    msg_warning() << "Method getPotentialEnergy not implemented yet.";
    return 0.0;
}


template<class DataTypes>
void PlaneForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v)
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > f1 = f;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecCoord > > p1 = x;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > v1 = v;

    this->m_contacts.clear();
    f1.resize(p1.size());

    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (d_localRange.getValue()[0] >= 0)
        ibegin = d_localRange.getValue()[0];

    if (d_localRange.getValue()[1] >= 0 && (unsigned int)d_localRange.getValue()[1]+1 < iend)
        iend = d_localRange.getValue()[1]+1;

    Real limit = this->d_maxForce.getValue();
    limit *= limit; // squared

    Real stiff = this->d_stiffness.getValue();
    Real damp = this->d_damping.getValue();
    DPos planeN = d_planeNormal.getValue();

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(p1[i])*planeN-d_planeD.getValue();
        if (d_bilateral.getValue() || d<0 )
        {
            Real forceIntensity = -stiff*d;
            Real dampingIntensity = -damp*d;
            DPos force = planeN*forceIntensity - DataTypes::getDPos(v1[i])*dampingIntensity;

            Real amplitude = force.norm2();
            if(limit>0 && amplitude > limit)
                force *= sqrt(limit / amplitude);

            Deriv tmpF;
            DataTypes::setDPos(tmpF, force);
            f1[i] += tmpF;
            this->m_contacts.push_back(i);
        }
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    sofa::helper::WriteAccessor< core::objectmodel::Data< VecDeriv > > df1 = df;
    sofa::helper::ReadAccessor< core::objectmodel::Data< VecDeriv > > dx1 = dx;

    df1.resize(dx1.size());
    const Real fact = (Real)(-this->d_stiffness.getValue() * sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue()));
    DPos planeN = d_planeNormal.getValue();

    for (unsigned int i=0; i<this->m_contacts.size(); i++)
    {
        unsigned int p = this->m_contacts[i];
        assert(p<dx1.size());
        DataTypes::setDPos(df1[p], DataTypes::getDPos(df1[p]) + planeN * (fact * (DataTypes::getDPos(dx1[p]) * planeN)));
    }
}

template<class DataTypes>
void PlaneForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    const Real fact = (Real)(-this->d_stiffness.getValue()*sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue()));
    Deriv normal;
    DataTypes::setDPos(normal, d_planeNormal.getValue());
    const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = mref.matrix;
    unsigned int offset = mref.offset;

    for (unsigned int i=0; i<this->m_contacts.size(); i++)
    {
        unsigned int p = this->m_contacts[i];
        for (sofa::Index l=0; l<Deriv::total_size; ++l)
            for (sofa::Index c=0; c<Deriv::total_size; ++c)
            {
                SReal coef = normal[l] * fact * normal[c];
                mat->add(offset + p*Deriv::total_size + l, offset + p*Deriv::total_size + c, coef);
            }
    }
}

template <class DataTypes>
void PlaneForceField<DataTypes>::buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix)
{
    if (!this->isComponentStateValid())
    {
        return;
    }

    Deriv normal;
    DataTypes::setDPos(normal, d_planeNormal.getValue());
    const auto localMatrix = -this->d_stiffness.getValue() * sofa::type::dyad(normal, normal);

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (const auto& contact : m_contacts)
    {
        const auto locationInGlobalMatrix = contact * Deriv::total_size;
        dfdx(locationInGlobalMatrix, locationInGlobalMatrix) += localMatrix;
    }
}

template <class DataTypes>
void PlaneForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void PlaneForceField<DataTypes>::updateStiffness( const VecCoord& vx )
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    helper::ReadAccessor<VecCoord> x = vx;

    this->m_contacts.clear();

    unsigned int ibegin = 0;
    unsigned int iend = x.size();

    if (d_localRange.getValue()[0] >= 0)
        ibegin = d_localRange.getValue()[0];

    if (d_localRange.getValue()[1] >= 0 && (unsigned int)d_localRange.getValue()[1]+1 < iend)
        iend = d_localRange.getValue()[1]+1;

    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(x[i])*d_planeNormal.getValue()-d_planeD.getValue();
        if (d<0)
            this->m_contacts.push_back(i);
    }
}


// Rotate the plane. Note that the rotation is only applied on the 3 first coordinates
template<class DataTypes>
void PlaneForceField<DataTypes>::rotate( Deriv axe, Real angle )
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    const type::Vec3d axe3d { DataTypes::getDPos(axe) };
    type::Vec3d normal3d { d_planeNormal.getValue() };
    type::Vec3d v = normal3d.cross(axe3d);
    if (v.norm2() < 1.0e-10) return;
    v.normalize();
    v = normal3d * cos ( angle ) + v * sin ( angle );
    *d_planeNormal.beginEdit() = v;
    d_planeNormal.endEdit();
}


template<class DataTypes>
void PlaneForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(this->d_componentState.getValue() != ComponentState::Valid)
        return ;

    if (!vparams->displayFlags().getShowForceFields() || !d_drawIsEnabled.getValue())
        return;

    drawPlane(vparams);
}


template<class DataTypes>
void PlaneForceField<DataTypes>::drawPlane(const core::visual::VisualParams* vparams,float size)
{
    if(!vparams->displayFlags().getShowForceFields())
        return;
    if (size == 0.0f)
        size = (float)d_drawSize.getValue();

    helper::ReadAccessor<VecCoord> p1 = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    type::Vec3 normal{type::NOINIT}, v1{ type::NOINIT }, v2{ type::NOINIT };
    get3DFrameFromDPosNormal<DataTypes>(d_planeNormal.getValue(), v1, v2, normal);

    const type::Vec3d center = normal*d_planeD.getValue();
    type::Vec3d corners[4];
    corners[0] = center-v1*size-v2*size;
    corners[1] = center+v1*size-v2*size;
    corners[2] = center+v1*size+v2*size;
    corners[3] = center-v1*size+v2*size;

    std::vector< type::Vec3 > points;

    points.push_back(corners[0]);
    points.push_back(corners[1]);
    points.push_back(corners[2]);

    points.push_back(corners[0]);
    points.push_back(corners[2]);
    points.push_back(corners[3]);
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    vparams->drawTool()->setPolygonMode(2,false); //Cull Front face

    vparams->drawTool()->drawTriangles(points, sofa::type::RGBAColor(d_drawColor.getValue()[0],d_drawColor.getValue()[1],d_drawColor.getValue()[2],0.5));
    vparams->drawTool()->setPolygonMode(0,false); //No Culling

    std::vector< type::Vec3 > pointsLine;

    // lines for points penetrating the plane
    unsigned int ibegin = 0;
    unsigned int iend = p1.size();

    if (d_localRange.getValue()[0] >= 0)
        ibegin = d_localRange.getValue()[0];

    if (d_localRange.getValue()[1] >= 0 && (unsigned int)d_localRange.getValue()[1]+1 < iend)
        iend = d_localRange.getValue()[1]+1;

    type::Vec3 point1,point2;
    for (unsigned int i=ibegin; i<iend; i++)
    {
        Real d = DataTypes::getCPos(p1[i])*d_planeNormal.getValue()-d_planeD.getValue();
        CPos p2 = DataTypes::getCPos(p1[i]);
        p2 += d_planeNormal.getValue()*(-d);
        if (d<0)
        {
            point1 = DataTypes::getCPos(p1[i]);
            point2 = p2;
            pointsLine.push_back(point1);
            pointsLine.push_back(point2);
        }
    }
    vparams->drawTool()->drawLines(pointsLine, 1, sofa::type::RGBAColor(1,0,0,1));

}

template <class DataTypes>
void PlaneForceField<DataTypes>::computeBBox(const core::ExecParams * params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if (onlyVisible && !d_drawIsEnabled.getValue())
        return;

    constexpr SReal max_real = std::numeric_limits<SReal>::max();
    constexpr SReal min_real = std::numeric_limits<SReal>::lowest();
    SReal maxBBox[3] = {min_real,min_real,min_real};
    SReal minBBox[3] = {max_real,max_real,max_real};

    const SReal size = d_drawSize.getValue();

    type::Vec3 normal{}, v1{}, v2{};
    get3DFrameFromDPosNormal<DataTypes>(d_planeNormal.getValue(), v1, v2, normal);

    const type::Vec3& center = normal*d_planeD.getValue();
    type::Vec3 corners[4];
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
    this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox,maxBBox));
}

} // namespace sofa::component::mechanicalload
