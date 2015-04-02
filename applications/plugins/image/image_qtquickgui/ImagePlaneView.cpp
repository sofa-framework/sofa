#include "ImagePlaneView.h"
#include "ImagePlaneModel.h"

#include <QPainter>
#include <qmath.h>

#include "../ImageTypes.h"

using cimg_library::CImg;

namespace sofa
{

namespace qtquick
{

using namespace sofa::defaulttype;

ImagePlaneView::ImagePlaneView(QQuickItem* parent) : QQuickPaintedItem(parent),
    myImagePlaneModel(0),
    myAxis(0),
    myIndex(0),
    myImage(),
    myLength(0)
{
    connect(this, &ImagePlaneView::imagePlaneModelChanged,  this, &ImagePlaneView::update);
    connect(this, &ImagePlaneView::axisChanged,             this, &ImagePlaneView::update);
    connect(this, &ImagePlaneView::indexChanged,            this, &ImagePlaneView::update);
}

void ImagePlaneView::paint(QPainter* painter)
{
    QSize size(myImage.size());
    size.scale(width(), height(), Qt::AspectRatioMode::KeepAspectRatio);

    double scaleRatio = 1.0;
    if(qFloor(width()) == size.width())
        scaleRatio = width() / myImage.width();
    else
        scaleRatio = height() / myImage.height();

    painter->translate((width()  - size.width() ) * 0.5,
                       (height() - size.height()) * 0.5);

    painter->scale(scaleRatio, scaleRatio);

    painter->drawImage(0, 0, myImage);
}

void ImagePlaneView::update()
{
    if(!myImagePlaneModel || -1 == myIndex || myAxis < 0 || myAxis > 5)
    {
        myImage = QImage();
        return;
    }

    setLength(myImagePlaneModel->length(myAxis));

    const cimg_library::CImg<unsigned char>& slice = myImagePlaneModel->retrieveSlice(myIndex, myAxis);
    if(slice.width() != myImage.width() || slice.height() != myImage.height())
    {
        myImage = QImage(slice.width(), slice.height(), QImage::Format_RGB32);
        setImplicitWidth(slice.width());
        setImplicitHeight(slice.height());
    }

    if(1 == slice.spectrum())
    {
        for(int y = 0; y < myImage.height(); y++)
            for(int x = 0; x < myImage.width(); x++)
                myImage.setPixel(x, y, qRgb(slice(x, y, 0, 0), slice(x, y, 0, 0), slice(x, y, 0, 0)));
    }
    else if(2 == slice.spectrum())
    {
        for(int y = 0; y < myImage.height(); y++)
            for(int x = 0; x < myImage.width(); x++)
                myImage.setPixel(x, y, qRgb(slice(x, y, 0, 0), slice(x, y, 0, 1), slice(x, y, 0, 0)));
    }
    else
    {
        for(int y = 0; y < myImage.height(); y++)
            for(int x = 0; x < myImage.width(); x++)
                myImage.setPixel(x, y, qRgb(slice(x, y, 0, 0), slice(x, y, 0, 1), slice(x, y, 0, 2)));
    }

    CImg<unsigned char> slicedModels = myImagePlaneModel->retrieveSlicedModels(myIndex, myAxis);
    if(slicedModels)
        for(int y = 0; y < myImage.height(); y++)
            for(int x = 0; x < myImage.width(); x++)
                if(slicedModels(x, y, 0, 0) || slicedModels(x, y, 0, 1) || slicedModels(x, y, 0, 2))
                    myImage.setPixel(x, y, qRgb(slicedModels(x, y, 0, 0), slicedModels(x, y, 0, 1), slicedModels(x, y, 0, 2)));

    // display selected pixels on the image plane
//    for(int i = 0; i < points.size(); ++i)
//    {
//        Coord point = points[i];
//        if(myAxis==0 && myIndex == point.x())
//            myImage.setPixel(QPoint(point.z(), point.y()), qRgb(255, 0, 0));

//        else if(myAxis==1 && myIndex == point.y())
//            myImage.setPixel(QPoint(point.x(), point.z()), qRgb(255, 0, 0));

//        else if (myAxis==2 && myIndex == point.z())
//            myImage.setPixel(QPoint(point.x(), point.y()), qRgb(255, 0, 0));
//    }

    QQuickItem::update();
}

void ImagePlaneView::setImagePlaneModel(ImagePlaneModel* imagePlaneModel)
{
    if(imagePlaneModel == myImagePlaneModel)
        return;

    myImagePlaneModel = imagePlaneModel;

    imagePlaneModelChanged();
}

void ImagePlaneView::setAxis(int axis)
{
    if(axis == myAxis)
        return;

    myAxis = axis;

    axisChanged();
}

void ImagePlaneView::setIndex(int index)
{
    if(index == myIndex)
        return;

    myIndex = index;

    indexChanged();
}

void ImagePlaneView::setLength(int length)
{
    if(length == myLength)
        return;

    myLength = length;

    lengthChanged();
}

}

}
