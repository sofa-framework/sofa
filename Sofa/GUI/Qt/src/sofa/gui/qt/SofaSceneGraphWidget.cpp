/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaSceneGraphWidget.h"

namespace sofa::gui::qt
{

void SofaSceneGraphWidget::setViewToDirty()
{
    if(!m_isLocked)
        return;

    if(m_isDirty)
        return;

    m_isDirty = true;
    emit dirtynessChanged(m_isDirty);
}

bool SofaSceneGraphWidget::isDirty()
{
    return m_isDirty;
}

bool SofaSceneGraphWidget::isLocked()
{
    return m_isLocked;
}

void SofaSceneGraphWidget::lock()
{
    if(m_isLocked)
        return;

    m_isLocked = true;
    emit lockingChanged(m_isLocked);
}

void SofaSceneGraphWidget::unLock()
{
    if(!m_isLocked)
        return;

    m_isLocked = false;

    if(m_isDirty)
        update();

    emit lockingChanged(m_isLocked);
}

} //namespace sofa::gui::qt
