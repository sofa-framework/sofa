/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_QT_VIEWER_QTOGRE_QOGRELIGHTWIDGET
#define SOFA_GUI_QT_VIEWER_QTOGRE_QOGRELIGHTWIDGET

#include <iostream>

#include <Ogre.h>

#include <sofa/gui/qt/WDoubleLineEdit.h>
#ifdef SOFA_QT4
#include <QWidget>
#include <Q3GroupBox>
#include <QCheckBox>
#include <QGridLayout>
#else
#include <qgroupbox.h>
#include <qcheckbox.h>
#include <qlayout.h>
typedef QGroupBox Q3GroupBox;
#endif

#define SIZE_ENTRY 50

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

class QOgreLightWidget: public Q3GroupBox
{
    Q_OBJECT
public:
    QOgreLightWidget(Ogre::SceneManager* s, QWidget * p = 0,  std::string lightName=std::string(),const char * name = 0 );
    ~QOgreLightWidget();


    Q3GroupBox *getWidget() {return this;}

    std::string getName() const {return name;}
    virtual std::string getType() const =0;

    virtual void save(std::ofstream &out);

    bool getDirty() const {return dirty;}

    void restoreLight(std::string name);

    //Update the parameters of the light
    virtual void updateLight();
    //Update the GUI with the current parameters of the light
    virtual void updateInfo();

public slots:
    void setDirty() {dirty=true; emit(isDirty());}
    void setClean() {dirty=false;}



signals:
    void isDirty();
protected:
    virtual void addLight()=0;

    //Name of the light in Ogre
    std::string name;
    Ogre::SceneManager* mSceneMgr;
    QWidget *parent;

    //indicate if the parameters have been modified
    bool dirty;

    //Widget
    QWidget *global;
    QGridLayout *globalLayout;

    //Parameters
    QCheckBox *castShadows;
    WDoubleLineEdit *diffuse[3];
    WDoubleLineEdit *specular[3];
};

class QOgreDirectionalLightWidget: public QOgreLightWidget
{
public :
    QOgreDirectionalLightWidget(Ogre::SceneManager* s, QWidget *p=0, std::string lightName=std::string(), const char * name = 0);
    ~QOgreDirectionalLightWidget();
    std::string getType() const {return std::string("directional");}
    void save(std::ofstream &out);

    void updateLight();
    void updateInfo();
protected:
    void addLight();

    WDoubleLineEdit *direction[3];
};


class QOgrePointLightWidget: public QOgreLightWidget
{
public :
    QOgrePointLightWidget(Ogre::SceneManager* s, QWidget *p=0, std::string lightName=std::string(), const char * name = 0);
    ~QOgrePointLightWidget();
    std::string getType() const {return std::string("point");}
    void save(std::ofstream &out);

    void updateLight();
    void updateInfo();
protected:
    void addLight();
    WDoubleLineEdit *position[3];
};


class QOgreSpotLightWidget: public QOgreLightWidget
{
public :
    QOgreSpotLightWidget(Ogre::SceneManager* s, QWidget *p=0, std::string lightName=std::string(), const char * name = 0);
    ~QOgreSpotLightWidget();
    std::string getType() const {return std::string("spot");}
    void save(std::ofstream &out);

    void updateLight();
    void updateInfo();
protected:
    void addLight();
    WDoubleLineEdit *direction[3];
    WDoubleLineEdit *position[3];
    WDoubleLineEdit *range[2];
};

}
}
}
}

#endif
