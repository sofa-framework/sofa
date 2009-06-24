/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_ARTRACKEVENT_H
#define SOFA_CORE_OBJECTMODEL_ARTRACKEVENT_H

#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/system/config.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

using namespace sofa::defaulttype;

/**
 * @brief This event notifies about ARTrack device interaction.
 */
class ARTrackEvent : public sofa::core::objectmodel::Event
{
public:

    /**
     * @brief Constructor.
     */
    ARTrackEvent(const Vector3& position, const Quat& orientation, const sofa::helper::fixed_array<double,3>& angles);

    /**
     * @brief Destructor.
     */
    virtual ~ARTrackEvent() {}

    const Vector3 getPosition() const;
    const Quat getOrientation() const;
    const sofa::helper::fixed_array<double,3> getAngles() const;

private:
    Vector3 m_position; ///< ARTrack coordinates in a Vec3d type.
    Quat m_orientation; ///< ARTrack orientation.
    sofa::helper::fixed_array<double,3> m_angles; ///< ARTrack finger angles.
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
