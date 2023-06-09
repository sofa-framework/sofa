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
#include "AboutSOFADialog.h"
#include <sofa/gui/common/BaseGUI.h>

#include <QPushButton>
#include <QDesktopServices>
#include <QUrl>

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository;

namespace sofa::gui::qt
{

AboutSOFADialog::AboutSOFADialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi(this);
    connect(buttonOk, &QPushButton::clicked, this, &AboutSOFADialog::clickSupportUs);

    std::string file = "icons/AboutSOFA.png";
    if (DataRepository.findFile(file))
    {
        const QPixmap pix(QPixmap::fromImage(QImage(DataRepository.getFile ( file ).c_str())));
        label_2->setPixmap(pix);
    }
}

void AboutSOFADialog::clickSupportUs()
{
    QDesktopServices::openUrl(QUrl("https://www.sofa-framework.org/consortium/support-us/"));
}


} // namespace sofa::gui::qt
