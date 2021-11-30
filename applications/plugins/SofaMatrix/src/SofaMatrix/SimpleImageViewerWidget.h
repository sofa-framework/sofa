/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <SofaMatrix/config.h>

#include <sofa/gui/qt/SimpleDataWidget.h>
#include <image/image_gui/ImagePlaneWidget.h>

namespace sofa::gui::qt
{


class SimpleImageViewerGraphScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit SimpleImageViewerGraphScene(QImage* im, QObject *parent = nullptr)
    : QGraphicsScene(parent)
    , m_image(im)
    {
        assert(im);
        this->setSceneRect(0,0, m_image->width(), m_image->height());
    }

    void drawBackground(QPainter *painter, const QRectF &rect) override
    {
        QGraphicsScene::drawBackground(painter, rect);
        if(m_image)
        {
            painter->drawImage(this->sceneRect(),*m_image);
        }
    }

private:
    const QImage *m_image;
};

class BaseSimpleImageViewerGraphWidget : public QGraphicsView
{
    Q_OBJECT

public:
    using QGraphicsView::QGraphicsView;
};

template<class SimpleBitmapType>
class SimpleImageViewerGraphWidget : public BaseSimpleImageViewerGraphWidget
{
public:
    SimpleImageViewerGraphWidget(QWidget * parent, const SimpleBitmapType& plane)
    : BaseSimpleImageViewerGraphWidget(parent)
    , m_imageplane(&plane)
    {
        readFromData(plane);
        const QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setSizePolicy(sizePolicy);
        QVBoxLayout* vbox = new QVBoxLayout();
        vbox->addStretch();
        this->setLayout(vbox);
        
        m_scene = new SimpleImageViewerGraphScene(&m_image, this);
        this->setScene(m_scene);
        fitInView();
        if (m_scene)
            m_scene->update();
    }

    void readFromData(const SimpleBitmapType& plane)
    {
        m_imageplane = &plane;

        if (auto planeImage = m_imageplane->getImage())
        {
            if (planeImage->getCImgList().size())
            {
                const auto& cimg = planeImage->getCImg();
                if (cimg.size())
                {
                    const auto w = cimg.width();
                    const auto h = cimg.height();

                    m_image = QImage(w,h, QImage::Format_RGB32);

                    for( int y=0; y < m_image.height(); y++)
                    {
                        for( int x=0; x < m_image.width(); x++)
                        {
                            auto grey = cimg(x,y);
                            m_image.setPixel(x, y, qRgb(grey, grey, grey));
                        }
                    }

                    return;
                }
            }
        }

        m_image = QImage(1,1, QImage::Format_RGB32);
    }

    ~SimpleImageViewerGraphWidget() override {}
protected:

    void fitInView()
    {
        this->QGraphicsView::fitInView(QRectF(0,0,m_image.width(), m_image.height()), Qt::KeepAspectRatioByExpanding);
    }

    void resizeEvent ( QResizeEvent* /*resizeevent*/) override  { this->fitInView(); }

    const SimpleBitmapType* m_imageplane { nullptr };
    QImage m_image;
    SimpleImageViewerGraphScene * m_scene { nullptr };
};

class BaseSimpleImageViewerWidgetContainer: public QObject
{
    Q_OBJECT;
public:
    using QObject::QObject;
};

template<class SimpleBitmapType>
class SimpleImageViewerWidgetContainer: public BaseSimpleImageViewerWidgetContainer
{
public:
    using BaseSimpleImageViewerWidgetContainer::BaseSimpleImageViewerWidgetContainer;

    typedef SimpleImageViewerGraphWidget<SimpleBitmapType> Graph;
    Graph* m_graph { nullptr };

    typedef QHBoxLayout Layout;
    Layout* container_layout { nullptr };

    void writeToData(SimpleBitmapType& d)
    {
    }

    void readFromData(const SimpleBitmapType& plane)
    {
        if (m_graph)
        {
            m_graph->readFromData(plane);
        }
    }

    void setReadOnly(bool /*readOnly*/) { }

    bool createLayout( DataWidget* parent )
    {
        if( parent->layout() != nullptr || container_layout != nullptr ) return false;
        container_layout = new Layout(parent);
        return true;
    }

    bool createLayout( QLayout* layout)
    {
        if ( container_layout != nullptr )
            return false;
        container_layout = new Layout();
        layout->addItem(container_layout);
        return true;
    }

    bool createWidgets(DataWidget* parent, const SimpleBitmapType& plane, bool /*readOnly*/)
    {
        m_graph = new Graph(parent, plane);
        return true;
    }

    void insertWidgets()
    {
        if(m_graph)
            container_layout->addWidget(m_graph);
    }
};


template<class SimpleBitmapType>
class SOFA_SOFAMATRIX_API SimpleImageViewerWidget : public SimpleDataWidget<SimpleBitmapType, SimpleImageViewerWidgetContainer<SimpleBitmapType> >
{
    typedef SimpleDataWidget<SimpleBitmapType, SimpleImageViewerWidgetContainer<SimpleBitmapType> > Inherit;
    typedef sofa::core::objectmodel::Data<SimpleBitmapType> MyData;

public:
    SimpleImageViewerWidget(QWidget* parent,const char* name, MyData* d) : Inherit(parent,name,d)
    {
        this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    unsigned int sizeWidget() override {return 1;}
    unsigned int numColumnWidget() override {return 1;}

};

} // namespace sofa::gui::qt
