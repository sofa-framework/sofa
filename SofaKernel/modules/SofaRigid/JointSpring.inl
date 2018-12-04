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
#ifndef SOFA_JOINTSPRING_INL
#define SOFA_JOINTSPRING_INL

#include <SofaRigid/JointSpring.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
JointSpring<DataTypes>::JointSpring(int m1 , int m2,
                                    Real softKst, Real hardKst , Real softKsr , Real hardKsr , Real blocKsr,
                                    Real axmin , Real axmax , Real aymin , Real aymax , Real azmin , Real azmax,
                                    Real kd):
                                      m1(m1), m2(m2), kd(kd)
                                    , torsion(0,0,0), lawfulTorsion(0,0,0), KT(0,0,0) , KR(0,0,0)
                                    , softStiffnessTrans(softKst), hardStiffnessTrans(hardKst), softStiffnessRot(softKsr), hardStiffnessRot(hardKsr), blocStiffnessRot(blocKsr)
                                    , needToInitializeTrans(true), needToInitializeRot(true)
{
    limitAngles = sofa::defaulttype::Vec<6,Real>(axmin,axmax,aymin,aymax,azmin,azmax);
    freeMovements = sofa::defaulttype::Vec<6,bool>(false, false, false, true, true, true);
    for (unsigned int i=0; i<3; i++)
    {
        if(limitAngles[2*i]==limitAngles[2*i+1])
            freeMovements[3+i] = false;
    }
    initTrans = Vector(0,0,0);
    initRot = defaulttype::Quat(0,0,0,1);
}

template<class DataTypes>
std::istream& operator >> ( std::istream& in, JointSpring<DataTypes>& s )
{
    //default joint is a free rotation joint --> translation is bloqued, rotation is free
    s.freeMovements.set(false, false, false, true, true, true);
    s.initTrans.set(0,0,0);
    s.initRot.set(0,0,0,1);
    s.blocStiffnessRot = 0.0;
    //by default no angle limitation is set (bi values for initialisation)
    s.limitAngles.set(-100000., 100000., -100000., 100000., -100000., 100000.);

    bool initTransFound=false;

    std::string str;
    in>>str;
    if(str == "BEGIN_SPRING")
    {
        in>>s.m1>>s.m2; //read references
        in>>str;
        while(str != "END_SPRING")
        {
            if(str == "FREE_AXIS")
                in>>s.freeMovements;
            else if(str == "KS_T")
                in>>s.softStiffnessTrans>>s.hardStiffnessTrans;
            else if(str == "KS_R")
                in>>s.softStiffnessRot>>s.hardStiffnessRot;
            else if(str == "KS_B")
                in>>s.blocStiffnessRot;
            else if(str == "KD")
                in>>s.kd;
            else if(str == "R_LIM_X")
                in>>s.limitAngles[0]>>s.limitAngles[1];
            else if(str == "R_LIM_Y")
                in>>s.limitAngles[2]>>s.limitAngles[3];
            else if(str == "R_LIM_Z")
                in>>s.limitAngles[4]>>s.limitAngles[5];
            else if(str == "REST_T")
            {
                in>>s.initTrans;
                initTransFound=true;
            }
            else if(str == "REST_R")
            {
                in>>s.initRot;
            }
            else
            {
                msg_error("JointSprintForceField")<<"Unknown Attribute while parsing '"<<str<<"'" ;
                return in;
            }

            in>>str;
        }
    }
}

template<class DataTypes>
std::ostream& operator << ( std::ostream& out, const JointSpring<DataTypes>& s )
{
    out<<"BEGIN_SPRING  "<<s.m1<<" "<<s.m2<<"  ";

    if (s.freeMovements[0]!=false || s.freeMovements[1]!=false || s.freeMovements[2]!=false || s.freeMovements[3]!=true || s.freeMovements[4]!=true || s.freeMovements[5]!=true)
        out<<"FREE_AXIS "<<s.freeMovements<<"  ";
    if (s.softStiffnessTrans != 0.0 || s.hardStiffnessTrans != 10000.0)
        out<<"KS_T "<<s.softStiffnessTrans<<" "<<s.hardStiffnessTrans<<"  ";
    if (s.softStiffnessRot != 0.0 || s.hardStiffnessRot != 10000.0)
        out<<"KS_R "<<s.softStiffnessRot<<" "<<s.hardStiffnessRot<<"  ";
    if (s.blocStiffnessRot != s.hardStiffnessRot/100)
        out<<"KS_B "<<s.blocStiffnessRot<<"  ";
    if (s.kd != 0.0)
        out<<"KD "<<s.kd<<"  ";
    if (s.limitAngles[0]!=-100000 || s.limitAngles[1] != 100000)
        out<<"R_LIM_X "<<s.limitAngles[0]<<" "<<s.limitAngles[1]<<"  ";
    if (s.limitAngles[2]!=-100000 || s.limitAngles[3] != 100000)
        out<<"R_LIM_Y "<<s.limitAngles[2]<<" "<<s.limitAngles[3]<<"  ";
    if (s.limitAngles[4]!=-100000 || s.limitAngles[5] != 100000)
        out<<"R_LIM_Z "<<s.limitAngles[4]<<" "<<s.limitAngles[5]<<"  ";
    if (s.initTrans!= typename JointSpring<DataTypes>::Vector(0,0,0))
        out<<"REST_T "<<s.initTrans<<"  ";
    if (s.initRot[3]!= 1)
        out<<"REST_R "<<s.initRot<<"  ";

    out<<"END_SPRING"<<std::endl;
    return out;
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_JOINTSPRING_INL */

