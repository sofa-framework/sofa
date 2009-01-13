/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
//
// C++ Implementation : MechanicalStateController
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//
#define SOFA_COMPONENT_CONTROLLER_MECHANICALSTATECONTROLLER_CPP
#include <sofa/component/controller/MechanicalStateController.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MechanicalStateController)

// Register in the Factory
int MechanicalStateControllerClass = core::RegisterObject("")
//.add< MechanicalStateController<Vec3dTypes> >()
//.add< MechanicalStateController<Vec3fTypes> >()
//.add< MechanicalStateController<Vec2dTypes> >()
//.add< MechanicalStateController<Vec2fTypes> >()
        .add< MechanicalStateController<Vec1dTypes> >()
        .add< MechanicalStateController<Vec1fTypes> >()
        .add< MechanicalStateController<Rigid3dTypes> >()
        .add< MechanicalStateController<Rigid3fTypes> >()
//.add< MechanicalStateController<Rigid2dTypes> >()
//.add< MechanicalStateController<Rigid2fTypes> >()
        ;


template <>
void MechanicalStateController<Vec1dTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;


    //sout<<" applyController() : omni "<< omni << "  buttonOmni " <<buttonOmni<<sendl;

    if(omni)
    {
        if(mState)
        {

            if(buttonOmni)
            {
                if ((*mState->getX0())[0].x() < -0.001)
                    (*mState->getX0())[0].x() += 0.05;
                else
                    (*mState->getX0())[0].x() =  -0.001;

                if ((*mState->getX0())[1].x() > 0.001)
                    (*mState->getX0())[1].x() -= 0.05;
                else
                    (*mState->getX0())[1].x() = 0.001;
            }
            else
            {
                //sout<<"mouseMode==Release"<<sendl;

                if ((*mState->getX0())[0].x() > -0.7)
                    (*mState->getX0())[0].x() -= 0.05;
                else
                    (*mState->getX0())[0].x() = -0.7;

                if ((*mState->getX0())[1].x() < 0.7)
                    (*mState->getX0())[1].x() += 0.05;
                else
                    (*mState->getX0())[1].x() = 0.7;

            }


        }

    }
    else
    {
        //if (mState)
        //{
        //	if (mouseMode==BtLeft || mouseMode==BtRight)
        //	{
        //			//sout<<"mouseMode==BtLeft"<<sendl;

        //			if ((*mState->getX0())[0].x() < -0.01)
        //				(*mState->getX0())[0].x() += 0.01;
        //			else
        //				(*mState->getX0())[0].x() =  -0.01;
        //
        //			if ((*mState->getX0())[1].x() > 0.01)
        //				(*mState->getX0())[1].x() -= 0.01;
        //			else
        //				(*mState->getX0())[1].x() = 0.01;

        //	}
        //	else
        //	{
        //			//sout<<"mouseMode==Release"<<sendl;

        //			if ((*mState->getX0())[0].x() > -0.7)
        //				(*mState->getX0())[0].x() -= 0.01;
        //			else
        //				(*mState->getX0())[0].x() = -0.7;
        //
        //			if ((*mState->getX0())[1].x() < 0.7)
        //				(*mState->getX0())[1].x() += 0.01;
        //			else
        //				(*mState->getX0())[1].x() = 0.7;

        //	}
        //}
    }



    //	//sofa::simulation::tree::GNode *node = static_cast<sofa::simulation::tree::GNode*> (this->getContext());
    //	//sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
    //	//sofa::simulation::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);
    //}
};

template <>
void MechanicalStateController<Vec1fTypes>::applyController()
{
    using sofa::defaulttype::Quat;
    using sofa::defaulttype::Vec;


    //sout<<" applyController() : omni "<< omni << "  buttonOmni " <<buttonOmni<<sendl;

    if(omni)
    {
        if(mState)
        {

            if(buttonOmni)
            {
                if ((*mState->getX0())[0].x() < -0.001f)
                    (*mState->getX0())[0].x() += 0.05f;
                else
                    (*mState->getX0())[0].x() =  -0.001f;

                if ((*mState->getX0())[1].x() > 0.001f)
                    (*mState->getX0())[1].x() -= 0.05f;
                else
                    (*mState->getX0())[1].x() = 0.001f;
            }
            else
            {
                //sout<<"mouseMode==Release"<<sendl;

                if ((*mState->getX0())[0].x() > -0.7f)
                    (*mState->getX0())[0].x() -= 0.05f;
                else
                    (*mState->getX0())[0].x() = -0.7f;

                if ((*mState->getX0())[1].x() < 0.7f)
                    (*mState->getX0())[1].x() += 0.05f;
                else
                    (*mState->getX0())[1].x() = 0.7f;

            }


        }

    }
    else
    {
        //if (mState)
        //{
        //	if (mouseMode==BtLeft || mouseMode==BtRight)
        //	{
        //			//sout<<"mouseMode==BtLeft"<<sendl;

        //			if ((*mState->getX0())[0].x() < -0.01f)
        //				(*mState->getX0())[0].x() += 0.01f;
        //			else
        //				(*mState->getX0())[0].x() =  -0.01f;
        //
        //			if ((*mState->getX0())[1].x() > 0.01f)
        //				(*mState->getX0())[1].x() -= 0.01f;
        //			else
        //				(*mState->getX0())[1].x() = 0.01f;

        //	}
        //	else
        //	{
        //			//sout<<"mouseMode==Release"<<sendl;

        //			if ((*mState->getX0())[0].x() > -0.7f)
        //				(*mState->getX0())[0].x() -= 0.01f;
        //			else
        //				(*mState->getX0())[0].x() = -0.7f;
        //
        //			if ((*mState->getX0())[1].x() < 0.7f)
        //				(*mState->getX0())[1].x() += 0.01f;
        //			else
        //				(*mState->getX0())[1].x() = 0.7f;

        //	}
        //}
    }



    //	//sofa::simulation::tree::GNode *node = static_cast<sofa::simulation::tree::GNode*> (this->getContext());
    //	//sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor; mechaVisitor.execute(node);
    //	//sofa::simulation::UpdateMappingVisitor updateVisitor; updateVisitor.execute(node);
    //}
};

//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec3dTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec3fTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec2dTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec2fTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec1dTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Vec1fTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid3dTypes>;
template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid3fTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid2dTypes>;
//template class SOFA_COMPONENT_CONTROLLER_API MechanicalStateController<Rigid2fTypes>;

} // namespace controller

} // namespace component

} // namespace sofa
