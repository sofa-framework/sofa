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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEMODEL_H
#define SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEMODEL_H

#include "ImageQtQuickGUI.h"
#include <Scene.h>

#include "../ImageTypes.h"

namespace sofa
{

namespace qtquick
{

class BaseImagePlaneWrapper
{
public:
    virtual ~BaseImagePlaneWrapper()
    {

    }

    virtual cimg_library::CImg<unsigned char> retrieveSlice(int index, int axis) const = 0;
    virtual cimg_library::CImg<unsigned char> retrieveSlicedModels(int index, int axis) const = 0;
    virtual int length(int axis) const = 0;

};

template<class T>
class ImagePlaneWrapper : public BaseImagePlaneWrapper
{
public:
    ImagePlaneWrapper(const sofa::defaulttype::ImagePlane<T>& imagePlane) : BaseImagePlaneWrapper(),
        myImagePlane(imagePlane)
    {

    }

    cimg_library::CImg<unsigned char> retrieveSlice(int index, int axis) const
    {
        return convertToUC(myImagePlane.get_slice((unsigned int) index, (unsigned int) axis));
    }

    cimg_library::CImg<unsigned char> retrieveSlicedModels(int index, int axis) const
    {
        return convertToUC(myImagePlane.get_slicedModels((unsigned int) index, (unsigned int) axis));
    }

    int length(int axis) const
    {
        return myImagePlane.getDimensions()[axis];
    }

private:
    const sofa::defaulttype::ImagePlane<T>&     myImagePlane;

};

class SOFA_IMAGE_QTQUICKGUI_API ImagePlaneModel : public QObject
{
    Q_OBJECT

public:
    ImagePlaneModel(QObject* parent = 0);

public:
    cimg_library::CImg<unsigned char> retrieveSlice(int index, int axis) const;
    cimg_library::CImg<unsigned char> retrieveSlicedModels(int index, int axis) const;
    int length(int axis) const;

public:
    Q_PROPERTY(sofa::qtquick::SceneData* sceneData READ sceneData WRITE setSceneData NOTIFY sceneDataChanged)

public:
    SceneData* sceneData() const                {return mySceneData;}
    BaseImagePlaneWrapper* imagePlane() const;

protected:
    void setSceneData(sofa::qtquick::SceneData* sceneData);
    void setImagePlane(BaseImagePlaneWrapper* imagePlane);

signals:
    void sceneDataChanged();
    void imagePlaneChanged();

private slots:
    void handleSceneDataChange();

private:
    sofa::qtquick::SceneData*       mySceneData;
    mutable BaseImagePlaneWrapper*  myImagePlane;

};

}

}

#endif // SOFA_IMAGE_QTQUICKGUI_IMAGEPLANEMODEL_H
