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
#define SOFA_COMPONENT_CONTROLLER_ARTRACKCONTROLLER_CPP
#include "ARTrackController.inl"
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

SOFA_DECL_CLASS(ARTrackController)
SOFA_DECL_CLASS(ARTrackVirtualTimeController)

int ARTrackVirtualTimeControllerClass = core::RegisterObject("Provides ARTtrack control on the time of a BVHController.")
        .add< ARTrackVirtualTimeController >();

// Register in the Factory
int ARTrackControllerClass = core::RegisterObject("Provides ARTrack user control on a Mechanical State.")
#ifndef SOFA_FLOAT
        .add< ARTrackController<Vec1dTypes> >()
        .add< ARTrackController<Vec3dTypes> >()
        .add< ARTrackController<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ARTrackController<Vec1fTypes> >()
        .add< ARTrackController<Vec3fTypes> >()
        .add< ARTrackController<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ARTrackController<defaulttype::Vec1dTypes>;
template class ARTrackController<defaulttype::Vec3dTypes>;
template class ARTrackController<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ARTrackController<defaulttype::Vec1fTypes>;
template class ARTrackController<defaulttype::Vec3fTypes>;
template class ARTrackController<defaulttype::Rigid3fTypes>;
#endif


ARTrackVirtualTimeController::ARTrackVirtualTimeController()
    : virtualTime( initData(&virtualTime, 0.001, "virtualTime", "Time found for the BVH") )
    , step1(initData(&step1, 0.01, "step1", "time at initial position" ))
    , step2(initData(&step2, 0.6, "step2", "time at intermediate position" ))
    , step3(initData(&step3, 1.1, "step3", "time at final position" ))
    , maxMotion( initData(&maxMotion, 30.0, "maxMotion", "Displacement amplitude")  )
{

    mousePosX = 0;
    mousePosY = 0;

    ARTrackMotion = 0;
    resetBool = false;

}

void ARTrackVirtualTimeController::init()
{
    mousePosX = 0;
    mousePosY = 0;
    TotalMouseDisplacement =0.0;
    backward = false;
    this->virtualTime.setValue(step1.getValue());

    std::cout<<" ARTrackVirtualTimeController::init()"<<std::endl;

    resetBool=true;

}

void ARTrackVirtualTimeController::reinit()
{
    init();




}



void ARTrackVirtualTimeController::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event))
    {
        sofa::core::objectmodel::ARTrackEvent *aev = dynamic_cast<sofa::core::objectmodel::ARTrackEvent *>(event);
        onARTrackEvent(aev);
    }

    if (dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event))
    {
        sofa::core::objectmodel::MouseEvent *mev = dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event);
        onMouseEvent(mev);
    }
}

void ARTrackVirtualTimeController::onMouseEvent(core::objectmodel::MouseEvent *mev)
{

    std::cout<<"Mouse Event Detected : x ="<<mev->getPosX()<<" y="<<mev->getPosY()<<std::endl;
    mousePosX += mev->getPosX();
    mousePosY += mev->getPosY();





    mouseWheel = 0;
    applyController();


}
void ARTrackVirtualTimeController::onARTrackEvent(core::objectmodel::ARTrackEvent *aev)
{

    //if(f_printLog.getValue())
    //std::cout<<"onARTrackEvent: aev pos :"<<aev->getPosition()<<" aev quat :"<<aev->getOrientation()<<std::endl;


    if(resetBool)
    {
        ARTrackResetPos = - aev->getPosition().z();
        resetBool = false;

    }
    ARTrackMotion = - aev->getPosition().z() - ARTrackResetPos;



    //ARTrackMotion+=1.0;

    std::cout<<"ARTrackMotion = "<<ARTrackMotion<<std::endl;

    applyController();

}
void ARTrackVirtualTimeController::applyController()
{

    //int dx = mousePosX;
    //int dy = mousePosY;
    //mousePosX = 0;
    //mousePosY = 0;


    if (!backward)
    {
        if(ARTrackMotion > 0.0)
        {
            TotalMouseDisplacement = ARTrackMotion;
        }
        else
            TotalMouseDisplacement = 0.0;
    }
    else
    {
        if(ARTrackMotion - ARTrackIntermediatePos > 0.0)
            TotalMouseDisplacement = 0.0;
        else
            TotalMouseDisplacement = ARTrackIntermediatePos - ARTrackMotion;

    }


    std::cout<<"TotalMouseDisplacement"<<TotalMouseDisplacement<<std::endl;


    if (TotalMouseDisplacement > maxMotion.getValue() && !backward )
    {
        backward = true;
        ARTrackIntermediatePos = ARTrackMotion;
        TotalMouseDisplacement = 0;

    }
    if (TotalMouseDisplacement > maxMotion.getValue() && backward )
    {
        this->virtualTime.setValue(step3.getValue());
        return;

    }
    //TotalMouseDisplacement += sqrt((double)(dx*dx + dy*dy)) + (double)mouseWheel ;
    double alpha = TotalMouseDisplacement/maxMotion.getValue();

    std::cout<<" alpha = "<<alpha<<std::endl;
    if (!backward)
    {

        double time = step1.getValue() + alpha * (step2.getValue() - step1.getValue());
        this->virtualTime.setValue(time);

    }
    else
    {
        double time = step2.getValue() + alpha * (step3.getValue() - step2.getValue());
        this->virtualTime.setValue(time);
    }




}


} // namespace controller

} // namespace component

} // namespace sofa
