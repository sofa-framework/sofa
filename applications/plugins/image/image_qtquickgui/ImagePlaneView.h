/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEVIEW_H
#define SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEVIEW_H

#include "ImageQtQuickGUI.h"

#include <QQuickPaintedItem>
#include <QImage>

namespace sofa
{

namespace qtquick
{

class ImagePlaneModel;

class SOFA_IMAGE_QTQUICKGUI_API ImagePlaneView : public QQuickPaintedItem
{
    Q_OBJECT

public:
    ImagePlaneView(QQuickItem* parent = 0);

public:
    void paint(QPainter* painter);

public slots:
    void update();

public:
    Q_PROPERTY(sofa::qtquick::ImagePlaneModel* imagePlaneModel READ imagePlaneModel WRITE setImagePlaneModel NOTIFY imagePlaneModelChanged)
    Q_PROPERTY(int axis READ axis WRITE setAxis NOTIFY axisChanged)
    Q_PROPERTY(int index READ index WRITE setIndex NOTIFY indexChanged)
    Q_PROPERTY(int length READ length NOTIFY lengthChanged)

public:
    sofa::qtquick::ImagePlaneModel* imagePlaneModel() const {return myImagePlaneModel;}
    int axis() const {return myAxis;}
    int index() const {return myIndex;}
    int length() const {return myLength;}

protected:
    void setImagePlaneModel(sofa::qtquick::ImagePlaneModel* imagePlaneModel);
    void setAxis(int axis);
    void setIndex(int index);
    void setLength(int length);

signals:
    void imagePlaneModelChanged();
    void lengthChanged();
    void axisChanged();
    void indexChanged();

private:
    ImagePlaneModel*                    myImagePlaneModel;

    int                                 myAxis;
    int                                 myIndex;
    QImage                              myImage;
    int                                 myLength;

};

}

}

#endif // SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEVIEW_H
