/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_QT_DISPLAYLINKWIDGET_H
#define SOFA_GUI_QT_DISPLAYLINKWIDGET_H

#include "LinkWidget.h"
#include "ModifyObject.h"

#include <QWidget>
#include <QLineEdit>
#include <QTextEdit>
#include <QGroupBox>
#include <QSlider>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace gui
{

namespace qt
{

class LinkWidget;
class QDisplayLinkInfoWidget;
struct ModifyObjectFlags;

class QDisplayLinkWidget : public QGroupBox
{
    Q_OBJECT
public:
    QDisplayLinkWidget(QWidget* parent,
            core::objectmodel::BaseLink* link,
            const ModifyObjectFlags& flags);
    unsigned int getNumWidgets() const { return numWidgets_;};

public slots:
    void UpdateLink();              //QWidgets ---> BaseLink
    void UpdateWidgets();           //BaseLink ---> QWidget
signals:
    void WidgetDirty(bool);
    void WidgetUpdate();
    void LinkUpdate();
    void LinkOwnerDirty(bool);

protected:
    static QIcon& RefreshIcon()
    {
        static QIcon icon;
        if(icon.isNull())
        {
            std::string filename = "textures/refresh.png";
            sofa::helper::system::DataRepository.findFile(filename);
            icon = QIcon(filename.c_str());
        }
        return icon;
    }

protected:
    core::objectmodel::BaseLink* link_;
    QDisplayLinkInfoWidget*  linkinfowidget_;
    LinkWidget* linkwidget_;
    unsigned int numWidgets_;
    QHBoxLayout* gridLayout_;

};



class QLinkSimpleEdit : public LinkWidget
{
    Q_OBJECT
    typedef enum QEditType { TEXTEDIT, LINEEDIT } QEditType;
    typedef union QEditWidgetPtr
    {
        QLineEdit* lineEdit;
        QTextEdit* textEdit;
    } QEditWidgetPtr;

    typedef struct QSimpleEdit
    {
        QEditType type;
        QEditWidgetPtr widget;
    } QSimpleEdit;
public :
    QLinkSimpleEdit(QWidget*, const char* name, core::objectmodel::BaseLink*);
    virtual unsigned int numColumnWidget() {return 3;}
    virtual unsigned int sizeWidget() {return 1;}
    virtual bool createWidgets();
protected:
    virtual void readFromLink();
    virtual void writeToLink();
    QSimpleEdit innerWidget_;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_QT_DISPLAYLINKWIDGET_H
