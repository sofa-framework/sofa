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
#include <sofa/gui/qt/QtMessageRedirection.h>
#include <sofa/config.h>
#include <sofa/helper/logging/Messaging.h>

void sofa::gui::qt::redirectQtMessages(QtMsgType type,
                                       const QMessageLogContext& context,
                                       const QString& msg)
{
    SOFA_UNUSED(context);
    const QByteArray localMsg = msg.toLocal8Bit();
    switch (type)
    {
        case QtDebugMsg:
            msg_info("Qt") << localMsg.constData();
            break;
        case QtInfoMsg:
            msg_info("Qt") << localMsg.constData();
            break;
        case QtWarningMsg:
            msg_warning("Qt") << localMsg.constData();
            break;
        case QtCriticalMsg:
            msg_error("Qt") << localMsg.constData();
            break;
        case QtFatalMsg:
            msg_fatal("Qt") << localMsg.constData();
            break;
    }
}
