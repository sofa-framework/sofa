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
#ifndef PLUGINEXAMPLE_MYBEHAVIORMODEL_H
#define PLUGINEXAMPLE_MYBEHAVIORMODEL_H

#include <PluginExample/config.h>
#include <sofa/core/BehaviorModel.h>


namespace sofa
{

namespace component
{

namespace behaviormodel
{


/**
 * This BehaviorModel does nothing but contain a custom data widget.
 */
class MyBehaviorModel : public sofa::core::BehaviorModel
{

public:
    SOFA_CLASS(MyBehaviorModel, sofa::core::BehaviorModel);

protected:
    MyBehaviorModel();
    ~MyBehaviorModel();

public:
    virtual void init();
    virtual void reinit();
    virtual void updatePosition(double dt);

protected:
    Data<unsigned> customUnsignedData; ///< Example of unsigned data with custom widget
    Data<unsigned> regularUnsignedData; ///< Example of unsigned data with standard widget
};


} // namespace behaviormodel

} // namespace component

} // namespace sofa


#endif // PLUGINEXAMPLE_MYBEHAVIORMODEL_H
