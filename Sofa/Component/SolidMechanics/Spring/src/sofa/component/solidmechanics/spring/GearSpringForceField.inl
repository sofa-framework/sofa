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

#include <sofa/component/solidmechanics/spring/GearSpringForceField.h>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/RGBAColor.h>

#include <cassert>
#include <iostream>
#include <fstream>



namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
GearSpring<DataTypes>::GearSpring()
    : GearSpring(0, 0, 0, 0)
{
}

template<class DataTypes>
GearSpring<DataTypes>::GearSpring(unsigned int m1, unsigned int m2, unsigned int p1, unsigned int p2)
    : GearSpring(m1, m2, p1, p2, 10000, 10000, 10000, 10000, 1)
{
}

template<class DataTypes>
GearSpring<DataTypes>::GearSpring(unsigned int m1, unsigned  int m2, unsigned int p1, unsigned int p2, Real hardKst, Real softKsr, Real hardKsr, Real kd, Real ratio)
    : m1(m1)
    , m2(m2)
    , p1(p1)
    , p2(p2)
    , previousAngle1(0)
    , previousAngle2(0)
    , angle1(0)
    , angle2(0)
    , kd(kd)
    , hardStiffnessTrans(hardKst)
    , softStiffnessRot(softKsr)
    , hardStiffnessRot(hardKsr)
    , Ratio(ratio)
{
    freeAxis = sofa::type::Vec<2,unsigned int>(0,0);
}

template<class DataTypes>
GearSpringForceField<DataTypes>::GearSpringForceField(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
    , outfile(nullptr)
    , springs(initData(&springs,"spring","pairs of indices, stiffness, damping"))
    , f_filename( initData(&f_filename, "filename", "output file name"))
    , f_period( initData(&f_period, (Real)0.0, "period", "period between outputs"))
    , f_reinit( initData(&f_reinit, false, "reinit", "flag enabling reinitialization of the output file at each timestep"))
    , lastTime((Real)0.0)
    , showFactorSize(initData(&showFactorSize, (Real)1.0, "showFactorSize", "modify the size of the debug information of a given factor" ))
{
}

template<class DataTypes>
GearSpringForceField<DataTypes>::GearSpringForceField()
    : GearSpringForceField(nullptr, nullptr)
{
}

template<class DataTypes>
GearSpringForceField<DataTypes>::~GearSpringForceField()
{
    if (outfile) 	  delete outfile;
}


template <class DataTypes>
void GearSpringForceField<DataTypes>::init()
{
    this->Inherit::init();

    const std::string& filename = f_filename.getFullPath();
    if (!filename.empty())
    {
        outfile = new std::ofstream(filename.c_str());
        if( !outfile->is_open() )
        {
            msg_error() << "Creating file " << filename << " failed.";
            delete outfile;
            outfile = nullptr;
        }
    }

}

template <class DataTypes>
void GearSpringForceField<DataTypes>::reinit()
{
    const VecCoord& x1=this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2=this->mstate2->read(core::ConstVecCoordId::position())->getValue();
    sofa::type::vector<Spring> &springsVector=*(springs.beginEdit());
    for (unsigned int i=0; i<springs.getValue().size(); ++i)
    {
        Spring &s=springsVector[i];
        if(s.p1==s.m1) { s.angle1 = s.previousAngle1 = 0.0; s.ini1 = x1[s.p1]; }
        else s.angle1 = s.previousAngle1 = getAngleAroundAxis(x1[s.p1],x1[s.m1],s.freeAxis[0]);
        if(s.p2==s.m2) { s.angle2 = s.previousAngle2 = 0.0; s.ini2 = x2[s.p2]; }
        else s.angle2 = s.previousAngle2 = getAngleAroundAxis(x2[s.p2],x2[s.m2],s.freeAxis[1]);
    }
    springs.endEdit();
}


template <class DataTypes>
void GearSpringForceField<DataTypes>::bwdInit()
{
    reinit();
}

template<class DataTypes>
void GearSpringForceField<DataTypes>::addSpringForce( SReal& /*potentialEnergy*/, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int , /*const*/ Spring& spring)
{
    const Coord *cc1 = &p1[spring.m1]  , *cc2 = &p2[spring.m2] , *cp1 , *cp2;
    Deriv dv1 = -v1[spring.m1] , dv2 = -v2[spring.m2] ;

    if(spring.m1==spring.p1) cp1 = &spring.ini1; else { cp1 = &p1[spring.p1]; dv1 += v1[spring.p1];	spring.ini1 = *cp1;	}
    if(spring.m2==spring.p2) cp2 = &spring.ini2; else { cp2 = &p2[spring.p2]; dv2 += v2[spring.p2];	spring.ini2 = *cp2;	}

    // pivot
    // force
    Vector  fT1 = ((*cp1).getCenter() - (*cc1).getCenter())*spring.hardStiffnessTrans + getVCenter(dv1)*spring.kd,
            fT2 = ((*cp2).getCenter() - (*cc2).getCenter())*spring.hardStiffnessTrans + getVCenter(dv2)*spring.kd;

    // torque
    Vector fR1 , fR2;
    getVectorAngle(*cc1,*cp1,spring.freeAxis[0],fR1);		fR1 = fR1*spring.hardStiffnessRot + getVOrientation(dv1)*spring.kd,
            getVectorAngle(*cc2,*cp2,spring.freeAxis[1],fR2);		fR2 = fR2*spring.hardStiffnessRot + getVOrientation(dv2)*spring.kd;

    // add force
    const Deriv force1(fT1, fR1 );
    f1[spring.m1] += force1; 
    if(spring.m1!=spring.p1) 
    {	
        f1[spring.p1] -= force1; 
    }
    const Deriv force2(fT2, fR2 );
    f2[spring.m2] += force2; 
    if(spring.m2!=spring.p2) 
    { 	
        f2[spring.p2] -= force2; 
    }

    // gear
    // get gear rotation axis in global coord
    Mat M1, M2;
    cc1->writeRotationMatrix(M1);
    cc2->writeRotationMatrix(M2);

    Vector axis1,axis2;
    for(unsigned int i=0; i<axis1.size(); ++i)
    {
        axis1[i]=M1[i][spring.freeAxis[0]];
        axis2[i]=M2[i][spring.freeAxis[1]];
    }

    // compute 1D forces using updated angles around gear rotation axis
    Real newAngle1 = getAngleAroundAxis(*cp1,*cc1,spring.freeAxis[0]),
            newAngle2 = getAngleAroundAxis(*cp2,*cc2,spring.freeAxis[1]);

    while(newAngle1 - spring.previousAngle1 > M_PI) newAngle1 -= M_2_PI;
    while(newAngle1 - spring.previousAngle1 < -M_PI) newAngle1 += M_2_PI;
    while(newAngle2 - spring.previousAngle2 > M_PI) newAngle2 -= M_2_PI;
    while(newAngle2 - spring.previousAngle2 < -M_PI) newAngle2 += M_2_PI;

    spring.angle1 += newAngle1 - spring.previousAngle1; spring.previousAngle1 = newAngle1;
    spring.angle2 += newAngle2 - spring.previousAngle2; spring.previousAngle2 = newAngle2;

    Real f = ( - spring.angle1 * spring.Ratio - spring.angle2 ) * spring.softStiffnessRot; // force1 = - force2  at contact point

    // convert 1D force into torques
    getVOrientation(f1[spring.m1]) += axis1*f;
    getVOrientation(f2[spring.m2]) += axis2*f/spring.Ratio;

    // write output file
    if (outfile)
    {
        if(f_reinit.getValue())  outfile->seekp(std::ios::beg);

        SReal time = this->getContext()->getTime();
        if (time >= (lastTime + f_period.getValue()))
        {
            lastTime += f_period.getValue();
            (*outfile) << "T= "<< time << "\n";
            (*outfile) << "  Angles= " << spring.angle1 << " , " << spring.angle2 << "\n";
            (*outfile) << "  F= " << f << "\n";

            if(f_reinit.getValue()) (*outfile) << "\n\n\n\n\n";

            outfile->flush();
        }
    }

}



template<class DataTypes>
void GearSpringForceField<DataTypes>::addSpringDForce(VecDeriv& f1, const VecDeriv& dx1, VecDeriv& f2, const VecDeriv& dx2, int , /*const*/ Spring& spring, Real kFactor)
{
    Deriv d1 = -dx1[spring.m1] , d2 = -dx2[spring.m2] ;

    // pivots
    if(spring.m1!=spring.p1) d1 += dx1[spring.p1];
    if(spring.m2!=spring.p2) d2 += dx2[spring.p2];

    Vector  dfT1 = getVCenter(d1)*spring.hardStiffnessTrans,
            dfT2 = getVCenter(d2)*spring.hardStiffnessTrans;

    Vector	KR1(spring.hardStiffnessRot,spring.hardStiffnessRot,spring.hardStiffnessRot),
            KR2(spring.hardStiffnessRot,spring.hardStiffnessRot,spring.hardStiffnessRot);

    KR1[spring.freeAxis[0]] = 0;
    KR2[spring.freeAxis[1]] = 0;

    Vector  dfR1 = spring.ini1.rotate(KR1.linearProduct(spring.ini1.inverseRotate(getVOrientation(d1)))),
            dfR2 = spring.ini2.rotate(KR2.linearProduct(spring.ini2.inverseRotate(getVOrientation(d2))));

    const Deriv dforce1(dfT1,dfR1),
            dforce2(dfT2,dfR2);

    f1[spring.m1] += dforce1 * kFactor; if(spring.m1!=spring.p1) f1[spring.p1] -= dforce1 * kFactor;
    f2[spring.m2] += dforce2 * kFactor; if(spring.m2!=spring.p2) f2[spring.p2] -= dforce2 * kFactor;

    // gear
    Real dangle1 = spring.ini1.inverseRotate(getVOrientation(dx1[spring.m1]))[spring.freeAxis[0]];
    Real dangle2 = spring.ini2.inverseRotate(getVOrientation(dx2[spring.m2]))[spring.freeAxis[1]];

    KR1 = KR2 = Vector(0,0,0);
    KR1[spring.freeAxis[0]] = - spring.softStiffnessRot * ( dangle1 * spring.Ratio + dangle2 );
    KR2[spring.freeAxis[1]] = - spring.softStiffnessRot * ( dangle1 + dangle2 / spring.Ratio );

    getVOrientation(f1[spring.m1]) += spring.ini1.rotate(KR1) * kFactor;
    getVOrientation(f2[spring.m2]) += spring.ini2.rotate(KR2) * kFactor;

}

template<class DataTypes>
void GearSpringForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{

    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    type::vector<Spring>& springs = *this->springs.beginEdit();

    f1.resize(x1.size());
    f2.resize(x2.size());
    m_potentialEnergy = 0;
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringForce(m_potentialEnergy,f1,x1,v1,f2,x2,v2, i, springs[i]);
    }
    this->springs.endEdit();

    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void GearSpringForceField<DataTypes>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();


    df1.resize(dx1.size());
    df2.resize(dx2.size());

    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    //const type::vector<Spring>& springs = this->springs.getValue();
    type::vector<Spring>& springs = *this->springs.beginEdit();
    for (unsigned int i=0; i<springs.size(); i++)
    {
        this->addSpringDForce(df1, dx1, df2, dx2, i, springs[i], kFactor);
    }
    this->springs.endEdit();

    data_df1.endEdit();
    data_df2.endEdit();
}

template <class DataTypes>
void GearSpringForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void GearSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields())) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& p1 =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& p2 =this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    vparams->drawTool()->disableLighting();
    constexpr const sofa::type::RGBAColor& color = sofa::type::RGBAColor::yellow();
    const type::vector<Spring>& springs = this->springs.getValue();

    float radius = showFactorSize.getValue() / 15; //see helper/gl/Cylinder.cpp

    for (unsigned int i=0; i<springs.size(); i++)
    {
        if(springs[i].freeAxis[0] == 0)
        {
            typename DataTypes::CPos vec = Vector((Real)(1.0*showFactorSize.getValue()), 0, 0);

            typename DataTypes::CPos v0 = p1[springs[i].m1].getCenter();
            typename DataTypes::CPos v1 = p1[springs[i].m1].getOrientation().rotate(vec) + v0;

            vparams->drawTool()->drawCylinder(v0, v1 , radius, color);
        }
        if(springs[i].freeAxis[0] == 1)
        {
            typename DataTypes::CPos vec = Vector(0, (Real)(1.0*showFactorSize.getValue()), 0.0);

            typename DataTypes::CPos v0 = p1[springs[i].m1].getCenter();
            typename DataTypes::CPos v1 = p1[springs[i].m1].getOrientation().rotate(vec) + v0;

            vparams->drawTool()->drawCylinder(v0, v1, radius, color);
        }
        if(springs[i].freeAxis[0] == 2)
        {
            typename DataTypes::CPos vec = Vector(0, 0, (Real)(1.0*showFactorSize.getValue()));

            typename DataTypes::CPos v0 = p1[springs[i].m1].getCenter();
            typename DataTypes::CPos v1 = p1[springs[i].m1].getOrientation().rotate(vec) + v0;

            vparams->drawTool()->drawCylinder(v0, v1, radius, color);
        }

        if(springs[i].freeAxis[1] == 0)
        {
            typename DataTypes::CPos vec = Vector((Real)(1.0*showFactorSize.getValue()), 0, 0);

            typename DataTypes::CPos v0 = p2[springs[i].m2].getCenter();
            typename DataTypes::CPos v1 = p2[springs[i].m2].getOrientation().rotate(vec) + v0;
            
            vparams->drawTool()->drawCylinder(v0, v1, radius, color);
        }
        if(springs[i].freeAxis[1] == 1)
        {
            typename DataTypes::CPos vec = Vector(0, (Real)(1.0*showFactorSize.getValue()), 0.0);

            typename DataTypes::CPos v0 = p2[springs[i].m2].getCenter();
            typename DataTypes::CPos v1 = p2[springs[i].m2].getOrientation().rotate(vec) + v0;

            vparams->drawTool()->drawCylinder(v0, v1, radius, color);
        }
        if(springs[i].freeAxis[1] == 2)
        {
            typename DataTypes::CPos vec = Vector(0, 0, (Real)(1.0*showFactorSize.getValue()));

            typename DataTypes::CPos v0 = p2[springs[i].m2].getCenter();
            typename DataTypes::CPos v1 = p2[springs[i].m2].getOrientation().rotate(vec) + v0;

            vparams->drawTool()->drawCylinder(v0, v1, radius, color);
        }
    }


}

} // namespace sofa::component::solidmechanics::spring
