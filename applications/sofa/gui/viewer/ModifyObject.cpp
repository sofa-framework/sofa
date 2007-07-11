/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/

#include "ModifyObject.h"
#include <iostream>

#ifdef QT_MODULE_QT3SUPPORT
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <Q3GroupBox>
#else
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qspinbox.h>
#include <qcheckbox.h>
#include <qlayout.h>
#include <qgroupbox.h>
#endif


#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include "WFloatLineEdit.h"

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif

namespace sofa
{

namespace gui
{

namespace guiviewer
{

using namespace  sofa::defaulttype;

#ifndef QT_MODULE_QT3SUPPORT
typedef QGroupBox Q3GroupBox;
#endif


ModifyObject::ModifyObject( QWidget*  , const char*, bool, Qt::WFlags )
{
    node = NULL;
    list_Object = NULL;
}

ModifyObject::~ModifyObject()
{
}

//Set the default file
void ModifyObject::setNode(core::objectmodel::Base* node_clicked, Q3ListViewItem* item_clicked)
{
    node = node_clicked;
    item = item_clicked;

    dialogLayout = new QVBoxLayout( this, 0, 1, "dialogLayout");
    // displayWidget

    if (node)
    {

        //All the pointers to the QObjects will be kept in memory in list_Object
        list_Object= new std::list< QObject *>();


        const std::map< std::string, core::objectmodel::FieldBase* >& fields = node->getFields();

        int i=0;

        for( std::map< std::string, core::objectmodel::FieldBase* >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {

            //For each element, we create a layout
            std::ostringstream oss;
            oss << "itemLayout_" << i;
            Q3GroupBox *lineLayout = NULL;;
            // The label
            //QLabel *label = new QLabel(QString((*it).first.c_str()), lineLayout,0);
            //label->setGeometry( 10, i*25+5, 200, 20 );

            const std::string& fieldname = (*it).second->getValueTypeString();
            if( fieldname=="bool" )
            {
                //Remove from the dialog window everything about showing collision models, visual models...
                //Don't have any effect if the scene is animated: the root will erase the value.
                std::string name((*it).first);
                name.resize(4);
                if (name == "show") continue;

                std::string box_name(oss.str());
                lineLayout = new Q3GroupBox(this, QString(box_name.c_str()));
                lineLayout->setColumns(4);
                lineLayout->setTitle(QString((*it).first.c_str()));

                if( strcmp((*it).second->help,"TODO") )new QLabel((*it).second->help, lineLayout);

                // the bool line edit
                QCheckBox* checkBox = new QCheckBox(lineLayout);
                list_Object->push_back( (QObject *) checkBox);

                //checkBox->setGeometry( 205, i*25+5, 170, 20 );

                if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
                {
                    checkBox->setChecked(ff->getValue());
                    connect( checkBox, SIGNAL( toggled(bool) ), this, SLOT( changeValue() ) );
                }
            }
            else
            {
                std::string box_name(oss.str());
                lineLayout = new Q3GroupBox(this, QString(box_name.c_str()));
                lineLayout->setColumns(4);
                lineLayout->setTitle(QString((*it).first.c_str()));

                if( strcmp((*it).second->help,"TODO") )new QLabel((*it).second->help, lineLayout);

                if( fieldname=="int")
                {
                    QSpinBox* spinBox = new QSpinBox((int)INT_MIN,(int)INT_MAX,1,lineLayout);
                    list_Object->push_back( (QObject *) spinBox);

                    if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
                    {
                        spinBox->setValue(ff->getValue());
                        connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                    }
                }
                else if( fieldname=="unsigned int")
                {
                    QSpinBox* spinBox = new QSpinBox((int)0,(int)INT_MAX,1,lineLayout);
                    list_Object->push_back( (QObject *) spinBox);

                    if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
                    {
                        spinBox->setValue(ff->getValue());
                        connect( spinBox, SIGNAL( valueChanged(int) ), this, SLOT( changeValue() ) );
                    }
                }
                else if( fieldname=="float" || fieldname=="double" )
                {

                    WFloatLineEdit* editSFFloat = new WFloatLineEdit( lineLayout, "editSFFloat" );
                    list_Object->push_back( (QObject *) editSFFloat);

                    editSFFloat->setMinFloatValue( (float)-INFINITY );
                    editSFFloat->setMaxFloatValue( (float)INFINITY );


                    if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                    {
                        editSFFloat->setFloatValue(ff->getValue());
                        connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                    }
                    else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                    {
                        editSFFloat->setFloatValue(ff->getValue());
                        connect( editSFFloat, SIGNAL( textChanged(const QString &) ), this, SLOT( changeValue() ) );
                    }

                }
                else if( fieldname=="string" )
                {

                    QLineEdit* lineEdit = new QLineEdit(lineLayout);
                    list_Object->push_back( (QObject *) lineEdit);

                    if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
                    {
                        lineEdit->setText(QString(ff->getValue().c_str()));
                        connect( lineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }

                }
                else if( fieldname=="Vec3f"        || fieldname=="Vec3d" ||
                        fieldname=="Vec<3,float>" || fieldname=="Vec<3,double>" )
                {

                    WFloatLineEdit* editSFFloatX = new WFloatLineEdit( lineLayout, "editSFFloatX" );
                    list_Object->push_back( (QObject *) editSFFloatX);

                    editSFFloatX->setMinFloatValue( (float)-INFINITY );
                    editSFFloatX->setMaxFloatValue( (float)INFINITY );

                    WFloatLineEdit* editSFFloatY = new WFloatLineEdit( lineLayout, "editSFFloatY" );
                    list_Object->push_back( (QObject *) editSFFloatY);

                    editSFFloatY->setMinFloatValue( (float)-INFINITY );
                    editSFFloatY->setMaxFloatValue( (float)INFINITY );

                    WFloatLineEdit* editSFFloatZ = new WFloatLineEdit( lineLayout, "editSFFloatZ" );
                    list_Object->push_back( (QObject *) editSFFloatZ);

                    editSFFloatZ->setMinFloatValue( (float)-INFINITY );
                    editSFFloatZ->setMaxFloatValue( (float)INFINITY );



                    if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);
                        editSFFloatY->setFloatValue(ff->getValue()[1]);
                        editSFFloatZ->setFloatValue(ff->getValue()[2]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatY, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatZ, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }
                    else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);
                        editSFFloatY->setFloatValue(ff->getValue()[1]);
                        editSFFloatZ->setFloatValue(ff->getValue()[2]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatY, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatZ, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }

                }
                else if( fieldname=="Vec2f" || fieldname=="Vec2d" )
                {

                    WFloatLineEdit* editSFFloatX = new WFloatLineEdit( lineLayout, "editSFFloatX" );
                    list_Object->push_back( (QObject *) editSFFloatX);

                    editSFFloatX->setMinFloatValue( (float)-INFINITY );
                    editSFFloatX->setMaxFloatValue( (float)INFINITY );

                    WFloatLineEdit* editSFFloatY = new WFloatLineEdit( lineLayout, "editSFFloatY" );
                    list_Object->push_back( (QObject *) editSFFloatY);
                    editSFFloatY->setMinFloatValue( (float)-INFINITY );
                    editSFFloatY->setMaxFloatValue( (float)INFINITY );

                    if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);
                        editSFFloatY->setFloatValue(ff->getValue()[1]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatY, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }
                    else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                    {
                        editSFFloatX->setFloatValue(ff->getValue()[0]);
                        editSFFloatY->setFloatValue(ff->getValue()[1]);

                        connect( editSFFloatX, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                        connect( editSFFloatY, SIGNAL( textChanged(const QString&) ), this, SLOT( changeValue() ) );
                    }

                }
                else
                    std::cerr<<"RealGUI.cpp: UNKNOWN GUI FIELD TYPE : "<<fieldname<<"   --> add a new GUIField"<<std::endl;
            }

            ++i;
            if (lineLayout != NULL)
                dialogLayout->addWidget( lineLayout );
        }


        //Adding buttons at the bottom of the dialog
        QHBoxLayout *lineLayout = new QHBoxLayout( 0, 0, 6, "Button Layout");

        buttonUpdate = new QPushButton( this, "buttonUpdate" );
        lineLayout->addWidget(buttonUpdate);
        buttonUpdate->setText("&Update");
        buttonUpdate->setEnabled(false);

        QSpacerItem *Horizontal_Spacing = new QSpacerItem( 20, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
        lineLayout->addItem( Horizontal_Spacing );

        QPushButton *buttonOk = new QPushButton( this, "buttonOk" );
        lineLayout->addWidget(buttonOk);
        buttonOk->setText( tr( "&OK" ) );

        QPushButton *buttonCancel = new QPushButton( this, "buttonCancel" );
        lineLayout->addWidget(buttonCancel);
        buttonCancel->setText( tr( "&Cancel" ) );

        dialogLayout->addLayout( lineLayout );



        //Signals and slots connections
        connect( buttonUpdate,   SIGNAL( clicked() ), this, SLOT( updateValues() ) );
        connect( buttonOk,       SIGNAL( clicked() ), this, SLOT( closeDialog() ) );
        connect( buttonCancel,   SIGNAL( clicked() ), this, SLOT( reject() ) );

        //Title of the Dialog
        setCaption((node->getTypeName()+"::"+node->getName()).data());

        resize( QSize(553, 130).expandedTo(minimumSizeHint()) );
    }


}

void ModifyObject::changeValue()
{
    if (buttonUpdate == NULL) return;
    buttonUpdate->setEnabled(true);
}

void ModifyObject::closeDialog()
{
    updateValues();
    emit(accept());
}

void ModifyObject::updateValues()
{
    if (buttonUpdate == NULL) return;

    //Make the update of all the values
    if (node && list_Object != NULL)
    {

        std::list< QObject *>::iterator list_it=list_Object->begin();

        const std::map< std::string, core::objectmodel::FieldBase* >& fields = node->getFields();
        int i=0;

        for( std::map< std::string, core::objectmodel::FieldBase* >::const_iterator it = fields.begin(); it!=fields.end(); ++it)
        {
            const std::string& fieldname = (*it).second->getValueTypeString();
            if( fieldname=="int")
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;


                if( DataField<int> * ff = dynamic_cast< DataField<int> * >( (*it).second )  )
                {
                    ff->setValue(atoi(spinBox->text()));
                }
            }
            else if( fieldname=="unsigned int")
            {

                QSpinBox* spinBox = dynamic_cast< QSpinBox *> ( (*list_it) ); list_it++;


                if( DataField<unsigned int> * ff = dynamic_cast< DataField<unsigned int> * >( (*it).second )  )
                {
                    ff->setValue(atoi(spinBox->text()));
                }
            }
            else if( fieldname=="float" || fieldname=="double" )
            {

                WFloatLineEdit* editSFFloat = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;


                if( DataField<float> * ff = dynamic_cast< DataField<float> * >( (*it).second )  )
                {
                    ff->setValue(editSFFloat->getFloatValue());
                }
                else if(DataField<double> * ff = dynamic_cast< DataField<double> * >( (*it).second )  )
                {
                    ff->setValue((double) editSFFloat->getFloatValue());
                }

            }
            else if( fieldname=="bool" )
            {
                std::string name((*it).first);
                name.resize(4);
                if (name == "show") continue;

                // the bool line edit
                QCheckBox* checkBox = dynamic_cast< QCheckBox *> ( (*list_it) ); list_it++;


                if( DataField<bool> * ff = dynamic_cast< DataField<bool> * >( (*it).second )  )
                {
                    ff->setValue(checkBox->isOn());
                }

            }
            else if( fieldname=="string" )
            {

                QLineEdit* lineEdit = dynamic_cast< QLineEdit *> ( (*list_it) ); list_it++;


                if( DataField<std::string> * ff = dynamic_cast< DataField<std::string> * >( (*it).second )  )
                {

#ifdef QT_MODULE_QT3SUPPORT
                    std::string value(lineEdit->text());
                    ff->setValue(value.c_str());
#else
                    ff->setValue(lineEdit->text());
#endif
                    if ((*it).first == std::string("name")) item->setText(0,lineEdit->text());
                }

            }
            else if( fieldname=="Vec3f"        || fieldname=="Vec3d" ||
                    fieldname=="Vec<3,float>" || fieldname=="Vec<3,double>")
            {

                WFloatLineEdit* editSFFloatX = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                WFloatLineEdit* editSFFloatY = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                WFloatLineEdit* editSFFloatZ = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;

                if( DataField<Vec3f> * ff = dynamic_cast< DataField<Vec3f> * >( (*it).second )  )
                {
                    Vec<3, float> value ( editSFFloatX->getFloatValue() ,
                            editSFFloatY->getFloatValue() ,
                            editSFFloatZ->getFloatValue() );
                    ff->setValue(value);
                }
                else if(DataField<Vec3d> * ff = dynamic_cast< DataField<Vec3d> * >( (*it).second )  )
                {
                    Vec<3, double> value ((double) editSFFloatX->getFloatValue() ,
                            (double) editSFFloatY->getFloatValue() ,
                            (double) editSFFloatZ->getFloatValue() );
                    ff->setValue(value);
                }

            }
            else if( fieldname=="Vec2f" || fieldname=="Vec2d" )
            {

                WFloatLineEdit* editSFFloatX = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;
                WFloatLineEdit* editSFFloatY = dynamic_cast< WFloatLineEdit *> ( (*list_it) ); list_it++;

                if( DataField<Vec2f> * ff = dynamic_cast< DataField<Vec2f> * >( (*it).second )  )
                {
                    Vec<2, float> value (editSFFloatX->getFloatValue(),
                            editSFFloatY->getFloatValue());
                    ff->setValue(value);
                }
                else if(DataField<Vec2d> * ff = dynamic_cast< DataField<Vec2d> * >( (*it).second )  )
                {
                    Vec<2, double> value ( (double) editSFFloatX->getFloatValue(),
                            (double) editSFFloatY->getFloatValue());
                    ff->setValue(value);
                }

            }
            else
                std::cerr<<"RealGUI.cpp: UNKNOWN GUI FIELD TYPE : "<<fieldname<<"   --> add a new GUIField"<<std::endl;

            ++i;
        }

    }
    buttonUpdate->setEnabled(false);

}


} // namespace qt

} // namespace gui

} // namespace sofa

