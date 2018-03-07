/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_GUI_QT_LINKWIDGET_H
#define SOFA_GUI_QT_LINKWIDGET_H


#include "SofaGUIQt.h"
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/Factory.h>


#include <QDialog>
#include <QLineEdit>
#include <QTableWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QRadioButton>
#include <QButtonGroup>



namespace sofa
{

namespace gui
{

namespace qt
{

/**
 *\brief Abstract Interface of a qwidget which allows to edit a link.
 */
class SOFA_SOFAGUIQT_API LinkWidget : public QWidget
{
    Q_OBJECT
public:
    //
    // Factory related code
    //

    struct CreatorArgument
    {
        std::string name;
        core::objectmodel::BaseLink* link;
        QWidget* parent;
        bool readOnly;
    };

    static LinkWidget *CreateLinkWidget(const LinkWidget::CreatorArgument &dwarg);


public slots:
    /// Checks that widget has been edited
    /// emit LinkOwnerDirty in case the name field has been modified
    void updateLinkValue()
    {
        if(dirty)
        {
            const bool hasOwner = baseLink->getOwnerBase();
            std::string previousName;
            if ( hasOwner ) previousName = baseLink->getOwnerBase()->getName();
            writeToLink();
            updateVisibility();
            if(hasOwner && baseLink->getOwnerBase()->getName() != previousName)
            {
                emit LinkOwnerDirty(true);
            }
        }

        dirty = false;
        counter = baseLink->getCounter();

    }
    /// First checks that the widget is not currently being edited
    /// checks that the link has changed since the last time the widget
    /// has read the link value.
    /// ultimately read the link value.
    void updateWidgetValue()
    {
        if(!dirty)
        {
            if(counter != baseLink->getCounter())
            {
                readFromLink();
                this->update();
            }
        }


    }
    /// You call this slot anytime you want to specify that the widget
    /// value is out of sync with the underlying link value.
	void setWidgetDirty()
	{
		setWidgetDirty(true);
	}

    void setWidgetDirty(bool b)
    {
        dirty = b;
        emit WidgetDirty(b);
    }
signals:
    /// Emitted each time setWidgetDirty is called. You can also emit
    /// it if you want to tell the widget value is out of sync with
    /// the underlying link value.
    void WidgetDirty(bool );
    /// Currently this signal is used to reflect the changes of the
    /// component name in the sofaListview.
    void LinkOwnerDirty(bool );

	void LinkBeingChanged();
public:
    typedef core::objectmodel::BaseLink MyLink;

    LinkWidget(QWidget* parent,const char* /*name*/, MyLink* l) :
        QWidget(parent /*,name */), baseLink(l), dirty(false), counter(-1)
    {
    }
    virtual ~LinkWidget() {}

    inline virtual void setLink( MyLink* d)
    {
        baseLink = d;
        readFromLink();
    }


    /// BaseLink pointer accessor function.
    const core::objectmodel::BaseLink* getBaseLink() const { return baseLink; }
    core::objectmodel::BaseLink* getBaseLink() { return baseLink; }

    void updateVisibility()
    {
        //parentWidget()->setShown(baseLink->isDisplayed());
    }
    bool isDirty() { return dirty; }

    /// The implementation of this method holds the widget creation and the signal / slot
    /// connections.
    virtual bool createWidgets() = 0;
    /// Helper method to give a size.
    virtual unsigned int sizeWidget() {return 1;}
    /// Helper method for colum.
    virtual unsigned int numColumnWidget() {return 3;}

protected:
    /// The implementation of this method tells how the widget reads the value of the link.
    virtual void readFromLink() = 0;
    /// The implementation of this methods needs to tell how the widget can write its value
    /// in the link
    virtual void writeToLink() = 0;

    core::objectmodel::BaseLink* baseLink;
    bool dirty;
    int counter;
};



/// Widget used to display the name of a Link
class QDisplayLinkInfoWidget: public QWidget
{
    Q_OBJECT
public:
    QDisplayLinkInfoWidget(QWidget* parent, const std::string& helper, core::objectmodel::BaseLink* l, bool modifiable);
public slots:
    unsigned int getNumLines() const { return numLines_;}
protected:
    void formatHelperString(const std::string& helper, std::string& final_text);
    static unsigned int numLines(const std::string& str);
    core::objectmodel::BaseLink* link;
    unsigned int numLines_;
};

} // namespace qt

} // namespace gui

} // namespace sofa

#endif // SOFA_GUI_QT_LINKWIDGET_H
