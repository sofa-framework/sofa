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

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/solidmechanics/spring/FrameSpringForceField.h>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/RGBAColor.h>
#include <cassert>
#include <iostream>


namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
FrameSpringForceField<DataTypes>::FrameSpringForceField()
    : FrameSpringForceField(nullptr, nullptr)
{
}

template<class DataTypes>
FrameSpringForceField<DataTypes>::FrameSpringForceField ( MechanicalState* object1, MechanicalState* object2 )
    : Inherit ( object1, object2 )
    , springs ( initData ( &springs,"spring","pairs of indices, stiffness, damping, rest length" ) )
    , showLawfulTorsion ( initData ( &showLawfulTorsion, false, "show lawful Torsion", "dislpay the lawful part of the joint rotation" ) )
    , showExtraTorsion ( initData ( &showExtraTorsion, false, "show illicit Torsion", "dislpay the illicit part of the joint rotation" ) )
{
}

template <class DataTypes>
void FrameSpringForceField<DataTypes>::init()
{
    this->Inherit::init();
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::addSpringForce ( SReal& /*potentialEnergy*/, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int , const Spring& spring )
{
    int a = spring.m1;
    int b = spring.m2;

    Mat Mr01, Mr10, Mr02, Mr20;
    p1[a].writeRotationMatrix ( Mr01 );
    const bool canInvert1 = type::invertMatrix ( Mr10, Mr01 );
    assert(canInvert1);
    SOFA_UNUSED(canInvert1);
    p2[b].writeRotationMatrix ( Mr02 );
    const bool canInvert2 = type::invertMatrix ( Mr20, Mr02 );
    assert(canInvert2);
    SOFA_UNUSED(canInvert2);

    Deriv Vp1p2 = v2[b] - v1[a];

    VecN damping ( spring.kd, spring.kd, spring.kd );
    VecN kst ( spring.stiffnessTrans, spring.stiffnessTrans, spring.stiffnessTrans );
    VecN ksr ( spring.stiffnessRot, spring.stiffnessRot, spring.stiffnessRot );

    //store the referential of the spring (p1) to use it in addSpringDForce()
    springRef[a] = p1[a];

    VecN fT = kst.linearProduct( ( p2[b].getCenter() + Mr02 * ( spring.vec2)) - ( p1[a].getCenter() + Mr01 * ( spring.vec1))) + damping.linearProduct ( getVCenter(Vp1p2));
    VecN fR = ksr.linearProduct( ( p1[a].getOrientation().inverse() * p2[b].getOrientation()).quatToRotationVector());  // Use of quatToRotationVector instead of toEulerVector:
                                                                                                                        // this is done to keep the old behavior (before the
                                                                                                                        // correction of the toEulerVector  function). If the
                                                                                                                        // purpose was to obtain the Eulerian vector and not the
                                                                                                                        // rotation vector please use the following line instead
    //VecN fR = ksr.linearProduct( ( p1[a].getOrientation().inverse() * p2[b].getOrientation()).toEulerVector());

    VecN C1 = fR + cross( Mr01 * ( spring.vec1), fT) + damping.linearProduct ( getVOrientation(Vp1p2) );
    VecN C2 = fR + cross( Mr02 * ( spring.vec2), fT) + damping.linearProduct ( -getVOrientation(Vp1p2) );

    f1[a] += Deriv ( fT, C1);
    f2[b] -= Deriv ( fT, C2);
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::addSpringDForce ( VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int , const Spring& spring )
{
    const int a = spring.m1;
    const int b = spring.m2;
    const Deriv Mdx1dx2 = dx2[b] - dx1[a];

    Mat Mr01, Mr10;
    springRef[a].writeRotationMatrix ( Mr01 );
    const bool canInvert = type::invertMatrix ( Mr10, Mr01 );
    assert(canInvert);
    SOFA_UNUSED(canInvert);

    VecN kst ( spring.stiffnessTrans, spring.stiffnessTrans, spring.stiffnessTrans );
    VecN ksr ( spring.stiffnessRot, spring.stiffnessRot, spring.stiffnessRot );

    //compute directional force
    VecN df0 = Mr01 * ( kst.linearProduct ( Mr10*getVCenter(Mdx1dx2) ) );
    //compute rotational force
    VecN dR0 = Mr01 * ( ksr.linearProduct ( Mr10* getVOrientation(Mdx1dx2) ) );

    const Deriv dforce ( df0, dR0 );

    f1[a] += dforce;
    f2[b] -= dforce;
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{

    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();


    springRef.resize ( x1.size() );
    f1.resize ( x1.size() );
    f2.resize ( x2.size() );
    m_potentialEnergy = 0;
    const sofa::type::vector<Spring>& springsVec = springs.getValue();
    for ( unsigned int i=0; i<springsVec.size(); i++ )
    {
        this->addSpringForce ( m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springsVec[i] );
    }

    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();


    df1.resize ( dx1.size() );
    df2.resize ( dx2.size() );

    const sofa::type::vector<Spring>& springsVec = springs.getValue();
    for ( unsigned int i=0; i<springsVec.size(); i++ )
    {
        this->addSpringDForce ( df1,dx1,df2,dx2, i, springsVec[i] );
    }

    data_df1.endEdit();
    data_df2.endEdit();
}

template <class DataTypes>
void FrameSpringForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if ( ! ( ( this->mstate1 == this->mstate2 ) ?vparams->displayFlags().getShowForceFields() :vparams->displayFlags().getShowInteractionForceFields() ) ) return;
    const VecCoord& p1 =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 =this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();

    std::vector<sofa::type::Vec3> vertices;
    std::vector<sofa::type::RGBAColor> colors;

    const bool external = ( this->mstate1!=this->mstate2 );
    const type::vector<Spring>& springs = this->springs.getValue();

    for ( unsigned int i=0; i<springs.size(); i++ )
    {
        const double restLength = (springs[i].vec1.norm() + springs[i].vec2.norm());
        Real d = ( p2[springs[i].m2].getCenter()-p1[springs[i].m1].getCenter()).norm();
        if ( external )
        {
            if ( d < restLength *0.9999 )
            {
                colors.push_back(sofa::type::RGBAColor::red());
                colors.push_back(sofa::type::RGBAColor::red());
            }
            else
            {
                colors.push_back(sofa::type::RGBAColor::green());
                colors.push_back(sofa::type::RGBAColor::green());
            }
        }
        else
        {
            if ( d < restLength *0.9999 )
            {
                colors.push_back(sofa::type::RGBAColor(1,0.5, 0,1));
                colors.push_back(sofa::type::RGBAColor(1,0.5, 0,1));
            }
            else
            {
                colors.push_back(sofa::type::RGBAColor(0,1,0.5,1));
                colors.push_back(sofa::type::RGBAColor(0,1,0.5,1));
            }

        }

        vertices.push_back( p1[springs[i].m1].getCenter() );
        vertices.push_back( p2[springs[i].m2].getCenter() );
    }

    vparams->drawTool()->drawLines(vertices, 1, colors);


}


template<class DataTypes>
void FrameSpringForceField<DataTypes>::clear ( int reserve )
{
    type::vector<Spring>& springs = *this->springs.beginEdit();
    springs.clear();
    if ( reserve ) springs.reserve ( reserve );
    this->springs.endEdit();
}

template<class DataTypes>
void FrameSpringForceField<DataTypes>::addSpring ( const Spring& s )
{
    springs.beginEdit()->push_back ( s );
    springs.endEdit();
}


template<class DataTypes>
void FrameSpringForceField<DataTypes>::addSpring ( int m1, int m2, Real softKst, Real softKsr, Real kd )
{
    Spring s ( m1,m2,softKst,softKsr, kd );
    //TODO// Init vec1 et vec2. Encore mieux a la creation du ressort mais il manque les positions des DOFs...
    //const MechanicalState<DataTypes> obj1 = *(getMState1());
    //const MechanicalState<DataTypes> obj2 = *(getMState2());

    springs.beginEdit()->push_back ( s );
    springs.endEdit();
}

} // namespace sofa::component::solidmechanics::spring
