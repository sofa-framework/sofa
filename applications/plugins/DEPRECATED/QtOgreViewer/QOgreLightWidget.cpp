/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "QOgreLightWidget.h"


#ifdef SOFA_QT4
#include <QLabel>
#else
#include <qlabel.h>
#endif

#if !defined(INFINITY)
#define INFINITY 9.0e10
#endif


namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{


QOgreLightWidget::QOgreLightWidget(Ogre::SceneManager* s, QWidget * p, std::string lightName, const char * n ): Q3GroupBox(n,p), parent(p)
{
    mSceneMgr = s;

    dirty=false;

    int countLight=0;

    if (lightName.empty())
    {
        std::ostringstream sName;
        sName << "Light" << ++countLight;
        while (s->hasLight(sName.str()))
        {
            sName.str("");
            sName << "Light" << ++countLight;
        }
        name = sName.str();
    }
    else
        name=lightName;



    setTitle(QString(name.c_str()));
    setColumns(4);
    global = new QWidget(this);
    globalLayout = new QGridLayout(global);


    //Shadows
    globalLayout->addWidget(new QLabel(QString("castShadows"), global),0,0);
    globalLayout->addWidget(castShadows = new QCheckBox(global),0,1);
    connect(castShadows, SIGNAL( toggled(bool) ), this, SLOT( setDirty() ) );
    //Diffuse Colour
    globalLayout->addWidget(new QLabel(QString("Diffuse"), global),1,0);
    diffuse[0] = new WDoubleLineEdit(global,"diffuseR");
    globalLayout->addWidget(diffuse[0],1,1);
    diffuse[0]->setMinValue( 0.0f);
    diffuse[0]->setMaxValue( 1.0f);
    connect( diffuse[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    diffuse[1] = new WDoubleLineEdit(global,"diffuseG");
    globalLayout->addWidget(diffuse[1],1,2);
    diffuse[1]->setMinValue( 0.0f);
    diffuse[1]->setMaxValue( 1.0f);
    connect( diffuse[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    diffuse[2] = new WDoubleLineEdit(global,"diffuseB");
    globalLayout->addWidget(diffuse[2],1,3);
    diffuse[2]->setMinValue( 0.0f);
    diffuse[2]->setMaxValue( 1.0f);
    connect( diffuse[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    //Specular Colour
    globalLayout->addWidget(new QLabel(QString("Specular"), global), 2,0);
    specular[0] = new WDoubleLineEdit(global,"specularR");
    globalLayout->addWidget(specular[0],2,1);
    specular[0]->setMinValue( 0.0f);
    specular[0]->setMaxValue( 1.0f);
    connect( specular[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    specular[1] = new WDoubleLineEdit(global,"specularG");
    globalLayout->addWidget(specular[1],2,2);
    specular[1]->setMinValue( 0.0f);
    specular[1]->setMaxValue( 1.0f);
    connect( specular[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    specular[2] = new WDoubleLineEdit(global,"specularB");
    globalLayout->addWidget(specular[2],2,3);
    specular[2]->setMinValue( 0.0f);
    specular[2]->setMaxValue( 1.0f);
    connect( specular[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    castShadows->setChecked(true);
    diffuse[0]->setValue(1); diffuse[1]->setValue(1); diffuse[2]->setValue(1);
    specular[0]->setValue(1); specular[1]->setValue(1); specular[2]->setValue(1);
}

QOgreLightWidget::~QOgreLightWidget()
{
    if (mSceneMgr->hasLight(name))
    {
        mSceneMgr->destroyLight(name);
        delete castShadows;
        for (unsigned int i=0; i<3; ++i)
        {
            delete diffuse[i];
            delete specular[i];
        }
    }
}

void QOgreLightWidget::save(std::ofstream &out)
{
    out << "\t\t\t\t<colourDiffuse ";
    out << "r=\"" << diffuse[0]->getValue() << "\" ";
    out << "g=\"" << diffuse[1]->getValue() << "\" ";
    out << "b=\"" << diffuse[2]->getValue() << "\" />\n";


    out << "\t\t\t\t<colourSpecular ";
    out << "r=\"" << specular[0]->getValue() << "\" ";
    out << "g=\"" << specular[1]->getValue() << "\" ";
    out << "b=\"" << specular[2]->getValue() << "\" />\n";
}

void QOgreLightWidget::restoreLight(std::string n)
{
    if (mSceneMgr->hasLight(n))
    {
        name = n;
        setTitle(QString(n.c_str()));
        updateInfo();
    }
}

void QOgreLightWidget::updateLight()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    l->setCastShadows(castShadows->isChecked());
    l->setDiffuseColour(diffuse[0]->Value(),diffuse[1]->Value(),diffuse[2]->Value());
    l->setSpecularColour(specular[0]->Value(),specular[1]->Value(),specular[2]->Value());
}
void QOgreLightWidget::updateInfo()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    castShadows->setChecked(l->getCastShadows());
    const Ogre::ColourValue &d = l->getDiffuseColour();
    const Ogre::ColourValue &s = l->getSpecularColour();
    for (unsigned int i=0; i<3; ++i)
    {
        diffuse[i]->setValue(d[i]);
        specular[i]->setValue(s[i]);
    }
}


QOgreDirectionalLightWidget::QOgreDirectionalLightWidget(Ogre::SceneManager* s, QWidget *p, std::string lightName,  const char * name):QOgreLightWidget(s,p,lightName,name)
{
    //Direction
    globalLayout->addWidget(new QLabel(QString("Direction"), global),3,0);
    direction[0] = new WDoubleLineEdit(global,"directionX");
    globalLayout->addWidget(direction[0],3,1);
    direction[0]->setMinValue( (double)-INFINITY);
    direction[0]->setMaxValue( (double) INFINITY);
    connect( direction[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    direction[1] = new WDoubleLineEdit(global,"directionY");
    globalLayout->addWidget(direction[1],3,2);
    direction[1]->setMinValue( (double)-INFINITY);
    direction[1]->setMaxValue( (double) INFINITY);
    connect( direction[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    direction[2] = new WDoubleLineEdit(global,"directionZ");
    globalLayout->addWidget(direction[2],3,3);
    direction[2]->setMinValue( (double)-INFINITY);
    direction[2]->setMaxValue( (double) INFINITY);
    connect( direction[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    direction[0]->setValue(0); direction[1]->setValue(-1); direction[2]->setValue(0);

    addLight();
}

QOgreDirectionalLightWidget::~QOgreDirectionalLightWidget()
{
    for (unsigned int i=0; i<3; ++i)
    {
        delete direction[i];
    }
}

void QOgreDirectionalLightWidget::save(std::ofstream &out)
{
    out << "\t\t\t<light name=\"" << name << "\" type=\"directional\" castShadows=\"" ;
    if (castShadows->isChecked()) out << "true";
    else                          out << "false";
    out << "\">\n";

    out << "\t\t\t\t<normal ";
    out << "x=\"" << direction[0]->getValue() << "\" ";
    out << "y=\"" << direction[1]->getValue() << "\" ";
    out << "z=\"" << direction[2]->getValue() << "\" />\n";

    QOgreLightWidget::save(out);

    out << "\t\t\t</light>\n";
}

void QOgreDirectionalLightWidget::addLight()
{
    if (mSceneMgr->hasLight(name))
    {
        updateInfo();
    }
    else
    {
        Ogre::Light *light = mSceneMgr->createLight(name);
        light->setType(Ogre::Light::LT_DIRECTIONAL);
        updateLight();
    }
}


void QOgreDirectionalLightWidget::updateLight()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateLight();
    l->setDirection(direction[0]->Value(),direction[1]->Value(),direction[2]->Value());
}

void QOgreDirectionalLightWidget::updateInfo()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateInfo();
    const Ogre::Vector3 &n = l->getDirection();
    for (unsigned int i=0; i<3; ++i) direction[i]->setValue(n[i]);
}


QOgrePointLightWidget::QOgrePointLightWidget(Ogre::SceneManager* s, QWidget *p, std::string lightName,const char * name):QOgreLightWidget(s,p,lightName,name)
{
    //Position
    globalLayout->addWidget(new QLabel(QString("Position"), global),3,0);
    position[0] = new WDoubleLineEdit(global,"positionX");
    globalLayout->addWidget(position[0],3,1);
    position[0]->setMinValue( (double)-INFINITY);
    position[0]->setMaxValue( (double) INFINITY);
    connect( position[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    position[1] = new WDoubleLineEdit(global,"positionY");
    globalLayout->addWidget(position[1],3,2);
    position[1]->setMinValue( (double)-INFINITY);
    position[1]->setMaxValue( (double) INFINITY);
    connect( position[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    position[2] = new WDoubleLineEdit(global,"positionZ");
    globalLayout->addWidget(position[2],3,3);
    position[2]->setMinValue( (double)-INFINITY);
    position[2]->setMaxValue( (double) INFINITY);
    connect( position[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    position[0]->setValue(0); position[1]->setValue(50); position[2]->setValue(0);

    addLight();
}

QOgrePointLightWidget::~QOgrePointLightWidget()
{
    for (unsigned int i=0; i<3; ++i)
    {
        delete position[i];
    }
}

void QOgrePointLightWidget::save(std::ofstream &out)
{
    out << "\t\t\t<light name=\"" << name << "\" type=\"point\" castShadows=\"" ;
    if (castShadows->isChecked()) out << "true";
    else                          out << "false";
    out << "\">\n";

    out << "\t\t\t\t<position ";
    out << "x=\"" << position[0]->getValue() << "\" ";
    out << "y=\"" << position[1]->getValue() << "\" ";
    out << "z=\"" << position[2]->getValue() << "\" />\n";

    QOgreLightWidget::save(out);

    out << "\t\t\t</light>\n";
}

void QOgrePointLightWidget::addLight()
{
    if (mSceneMgr->hasLight(name))
    {
        updateInfo();
    }
    else
    {
        Ogre::Light *light = mSceneMgr->createLight(name);
        light->setType(Ogre::Light::LT_POINT);
        updateLight();
    }
}


void QOgrePointLightWidget::updateLight()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateLight();
    l->setPosition(position[0]->Value(),position[1]->Value(),position[2]->Value());
}

void QOgrePointLightWidget::updateInfo()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateInfo();
    const Ogre::Vector3 &n = l->getPosition();
    for (unsigned int i=0; i<3; ++i) position[i]->setValue(n[i]);
}


QOgreSpotLightWidget::QOgreSpotLightWidget(Ogre::SceneManager* s, QWidget *p, std::string lightName, const char * name):QOgreLightWidget(s,p,lightName,name)
{
    //Position
    globalLayout->addWidget(new QLabel(QString("Position"), global),3,0);
    position[0] = new WDoubleLineEdit(global,"positionX");
    globalLayout->addWidget(position[0],3,1);
    position[0]->setMinValue( (double)-INFINITY);
    position[0]->setMaxValue( (double) INFINITY);
    connect( position[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    position[1] = new WDoubleLineEdit(global,"positionY");
    globalLayout->addWidget(position[1],3,2);
    position[1]->setMinValue( (double)-INFINITY);
    position[1]->setMaxValue( (double) INFINITY);
    connect( position[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    position[2] = new WDoubleLineEdit(global,"positionZ");
    globalLayout->addWidget(position[2],3,3);
    position[2]->setMinValue( (double)-INFINITY);
    position[2]->setMaxValue( (double) INFINITY);
    connect( position[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    //Direction
    globalLayout->addWidget(new QLabel(QString("Direction"), global),4,0);
    direction[0] = new WDoubleLineEdit(global,"directionX");
    globalLayout->addWidget(direction[0],4,1);
    direction[0]->setMinValue( (double)-INFINITY);
    direction[0]->setMaxValue( (double) INFINITY);
    connect( direction[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    direction[1] = new WDoubleLineEdit(global,"directionY");
    globalLayout->addWidget(direction[1],4,2);
    direction[1]->setMinValue( (double)-INFINITY);
    direction[1]->setMaxValue( (double) INFINITY);
    connect( direction[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );
    direction[2] = new WDoubleLineEdit(global,"directionZ");
    globalLayout->addWidget(direction[2],4,3);
    direction[2]->setMinValue( (double)-INFINITY);
    direction[2]->setMaxValue( (double) INFINITY);
    connect( direction[2], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    //Range
    globalLayout->addWidget(new QLabel(QString("Range In"), global),5,0);
    range[0] = new WDoubleLineEdit(global,"rangeIN");  range[0]->setMaximumWidth(SIZE_ENTRY);
    globalLayout->addWidget(range[0],5,1);
    range[0]->setMinValue( 0.0f);
    range[0]->setMaxValue( 360.0f);
    connect( range[0], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    globalLayout->addWidget(new QLabel(QString("Out"), global),5,2);
    range[1] = new WDoubleLineEdit(global,"rangeOUT"); range[1]->setMaximumWidth(SIZE_ENTRY);
    globalLayout->addWidget(range[1],5,3);
    range[1]->setMinValue( 0.0f);
    range[1]->setMaxValue( 360.0f);
    connect( range[1], SIGNAL( returnPressed() ), this, SLOT( setDirty() ) );

    position[0]->setValue(0); position[1]->setValue(50); position[2]->setValue(0);
    direction[0]->setValue(0); direction[1]->setValue(-1); direction[2]->setValue(0);
    range[0]->setValue(30.0f);
    range[1]->setValue(50.0f);


    addLight();
}

QOgreSpotLightWidget::~QOgreSpotLightWidget()
{
    for (unsigned int i=0; i<3; ++i)
    {
        delete position[i];
    }
}

void QOgreSpotLightWidget::save(std::ofstream &out)
{
    out << "\t\t\t<light name=\"" << name << "\" type=\"spot\" castShadows=\"" ;
    if (castShadows->isChecked()) out << "true";
    else                          out << "false";
    out << "\">\n";

    out << "\t\t\t\t<position ";
    out << "x=\"" << position[0]->getValue() << "\" ";
    out << "y=\"" << position[1]->getValue() << "\" ";
    out << "z=\"" << position[2]->getValue() << "\" />\n";

    out << "\t\t\t\t<normal ";
    out << "x=\"" << direction[0]->getValue() << "\" ";
    out << "y=\"" << direction[1]->getValue() << "\" ";
    out << "z=\"" << direction[2]->getValue() << "\" />\n";

    out << "\t\t\t\t<lightRange ";
    out << "inner=\""  << range[0]->getValue() << "\" ";
    out << "outer=\""  << range[1]->getValue() << "\" ";
    out << "fallof=\"" << 1.0 << "\" />\n";

    QOgreLightWidget::save(out);

    out << "\t\t\t</light>\n";
}

void QOgreSpotLightWidget::addLight()
{
    if (mSceneMgr->hasLight(name))
    {
        updateInfo();
    }
    else
    {
        Ogre::Light *light = mSceneMgr->createLight(name);
        light->setType(Ogre::Light::LT_SPOTLIGHT);
        updateLight();
    }
}


void QOgreSpotLightWidget::updateLight()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateLight();
    l->setPosition(position[0]->Value(),position[1]->Value(),position[2]->Value());
    l->setDirection(direction[0]->Value(),direction[1]->Value(),direction[2]->Value());
    Ogre::Radian in (Ogre::Degree(range[0]->Value()));
    Ogre::Radian out(Ogre::Degree(range[1]->Value()));
    l->setSpotlightRange(in,out);
}

void QOgreSpotLightWidget::updateInfo()
{
    Ogre::Light *l = mSceneMgr->getLight(name);
    QOgreLightWidget::updateInfo();
    const Ogre::Vector3 &n = l->getPosition();
    const Ogre::Vector3 &d = l->getDirection();
    for (unsigned int i=0; i<3; ++i)
    {
        position[i]->setValue(n[i]);
        direction[i]->setValue(d[i]);
    }
    Ogre::Degree in(Ogre::Radian(l->getSpotlightInnerAngle()));
    Ogre::Degree out(Ogre::Radian(l->getSpotlightOuterAngle()));
    range[0]->setValue(in.valueDegrees());
    range[1]->setValue(out.valueDegrees());
}

}
}
}
}

