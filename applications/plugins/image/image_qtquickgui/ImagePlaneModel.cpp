#include <QDebug>
#include "ImagePlaneModel.h"

using cimg_library::CImg;

namespace sofa
{

namespace qtquick
{

using namespace sofa::defaulttype;

ImagePlaneModel::ImagePlaneModel(QObject* parent) : QObject(parent),
    mySceneData(0),
    myImagePlane(0)
{
    connect(this, &ImagePlaneModel::sceneDataChanged, this, &ImagePlaneModel::handleSceneDataChange);
}

cimg_library::CImg<unsigned char> ImagePlaneModel::retrieveSlice(int index, int axis) const
{
    if(!imagePlane())
        return cimg_library::CImg<unsigned char>();

    return myImagePlane->retrieveSlice(index, axis);
}

cimg_library::CImg<unsigned char> ImagePlaneModel::retrieveSlicedModels(int index, int axis) const
{
    if(!imagePlane())
        return cimg_library::CImg<unsigned char>();

    return myImagePlane->retrieveSlicedModels(index, axis);
}

int ImagePlaneModel::length(int axis) const
{
    if(!imagePlane() || axis < 0 || axis > 5)
        return 0;

    return myImagePlane->length(axis);
}

BaseImagePlaneWrapper* ImagePlaneModel::imagePlane() const
{
    const BaseData* data = mySceneData->data();
    if(!data)
        myImagePlane = 0;

    return myImagePlane;
}

void ImagePlaneModel::setSceneData(SceneData* sceneData)
{
    if(sceneData == mySceneData)
        return;

    mySceneData = sceneData;

    sceneDataChanged();
}

void ImagePlaneModel::setImagePlane(BaseImagePlaneWrapper *imagePlane)
{
    if(imagePlane == myImagePlane)
        return;

    myImagePlane = imagePlane;

    imagePlaneChanged();
}

void ImagePlaneModel::handleSceneDataChange()
{
    delete myImagePlane;
    setImagePlane(0);

    if(!mySceneData)
        return;

    const BaseData* data = mySceneData->data();
    if(!data)
        return;

    QString type = QString::fromStdString(data->getValueTypeString());

    if(0 == type.compare("ImagePlane<char>"))
        setImagePlane(new ImagePlaneWrapper<char>(*(ImagePlane<char>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<unsigned char>"))
        setImagePlane(new ImagePlaneWrapper<unsigned char>(*(ImagePlane<unsigned char>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<int>"))
        setImagePlane(new ImagePlaneWrapper<int>(*(ImagePlane<int>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<unsigned int>"))
        setImagePlane(new ImagePlaneWrapper<unsigned int>(*(ImagePlane<unsigned int>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<short>"))
        setImagePlane(new ImagePlaneWrapper<short>(*(ImagePlane<short>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<unsigned short>"))
        setImagePlane(new ImagePlaneWrapper<unsigned short>(*(ImagePlane<unsigned short>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<long>"))
        setImagePlane(new ImagePlaneWrapper<long>(*(ImagePlane<long>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<unsigned long>"))
        setImagePlane(new ImagePlaneWrapper<unsigned long>(*(ImagePlane<unsigned long>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<float>"))
        setImagePlane(new ImagePlaneWrapper<float>(*(ImagePlane<float>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<double>"))
        setImagePlane(new ImagePlaneWrapper<double>(*(ImagePlane<double>*) data->getValueVoidPtr()));
    else if(0 == type.compare("ImagePlane<bool>"))
        setImagePlane(new ImagePlaneWrapper<bool>(*(ImagePlane<bool>*) data->getValueVoidPtr()));
    else
        qWarning() << "Type unknown";
}

}

}
