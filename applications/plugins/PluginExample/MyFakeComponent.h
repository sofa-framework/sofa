/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_BEHAVIORMODEL_MyFakeComponent_H
#define SOFA_COMPONENT_BEHAVIORMODEL_MyFakeComponent_H


#include <sofa/core/BehaviorModel.h>


#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#else
#ifdef SOFA_BUILD_DYNAMICLIBEXAMPLE
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_DYNAMICLIBEXAMPLEAPI SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_DYNAMICLIBEXAMPLEAPI SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif




namespace sofa
{

namespace component
{

namespace behaviormodel
{


class  MyFakeComponent : public sofa::core::BehaviorModel
{
public:
    SOFA_CLASS(MyFakeComponent,sofa::core::BehaviorModel);
    MyFakeComponent();
    ~MyFakeComponent();

    virtual void init();

    virtual void reinit();

    virtual void updatePosition(double dt);


protected:

    Data<unsigned> customUnsignedData;
    Data<unsigned> regularUnsignedData;
private:

};


}

}

}



#endif
