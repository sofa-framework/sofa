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
#include <SofaMatrix/BaseMatrixImageViewerWidget.h>

namespace sofa::gui::qt
{

// Add the widget in the DataWidgetFactory with the key 'matrixbitmap', so that a Data< type::BaseMatrixProxy > can call setWidget("matrixbitmap")
helper::Creator<DataWidgetFactory, BaseMatrixImageViewerWidget > DWClass_imageViewerWidget("matrixbitmap",true);

BaseMatrixImageViewerGraphScene::BaseMatrixImageViewerGraphScene(QImage* im, QObject* parent): QGraphicsScene(parent)
    , m_image(im)
{
    assert(im);
    this->setSceneRect(0,0, m_image->width(), m_image->height());
}

void BaseMatrixImageViewerGraphScene::drawBackground(QPainter* painter, const QRectF& rect)
{
    QGraphicsScene::drawBackground(painter, rect);
    if(m_image)
    {
        painter->drawImage(this->sceneRect(),*m_image);
    }
}

BaseMatrixImageViewerGraphWidget::BaseMatrixImageViewerGraphWidget(QWidget* parent, const type::BaseMatrixImageProxy& plane)
    : QGraphicsView(parent)
    , m_imageplane(&plane)
{
    readFromData(plane);
    const QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setSizePolicy(sizePolicy);
    QVBoxLayout* vbox = new QVBoxLayout();
    vbox->addStretch();
    this->setLayout(vbox);

    m_scene = new BaseMatrixImageViewerGraphScene(&m_image, this);
    this->setScene(m_scene);
    fitInView();
    if (m_scene)
        m_scene->update();
}

void BaseMatrixImageViewerGraphWidget::readFromData(const type::BaseMatrixImageProxy& proxy)
{
    m_imageplane = &proxy;

    if (const linearalgebra::BaseMatrix* matrix = proxy.getMatrix())
    {
        const auto w = matrix->rows();
        const auto h = matrix->cols();

        if (m_image.width() != w || m_image.height() != h)
        {
            m_image = QImage(w,h, QImage::Format_Mono);

            constexpr QRgb white = qRgb(255, 255, 255);
            constexpr QRgb black = qRgb(0, 0, 0);

            // definition of a color table where the first color is white and the second is black
            m_image.setColor(0, white);
            m_image.setColor(1, black);
        }

        for( int y = 0; y < h; ++y)
        {
            for( int x = 0; x < w; ++x)
            {
                // matrix element is first converted to a bool (zero or non-zero), then  converted to a color table index
                m_image.setPixel(x, y, static_cast<uint>(static_cast<bool>(matrix->element(x, y))));
            }
        }
    }
    else
    {
        m_image = QImage(1, 1, QImage::Format_Mono);
    }
}

void BaseMatrixImageViewerGraphWidget::fitInView()
{
    this->QGraphicsView::fitInView(QRectF(0,0,m_image.width(), m_image.height()), Qt::KeepAspectRatioByExpanding);
}

void BaseMatrixImageViewerGraphWidget::resizeEvent(QResizeEvent* resize_event)
{
    SOFA_UNUSED(resize_event);
    this->fitInView();
}

void BaseMatrixImageViewerWidgetContainer::writeToData(type::BaseMatrixImageProxy& d)
{
    SOFA_UNUSED(d);
}

void BaseMatrixImageViewerWidgetContainer::readFromData(const type::BaseMatrixImageProxy& proxy) const
{
    if (m_graph)
    {
        m_graph->readFromData(proxy);
    }
}

void BaseMatrixImageViewerWidgetContainer::setReadOnly(bool)
{ }

bool BaseMatrixImageViewerWidgetContainer::createLayout(DataWidget* parent)
{
    if( parent->layout() != nullptr || container_layout != nullptr ) return false;
    container_layout = new Layout(parent);
    return true;
}

bool BaseMatrixImageViewerWidgetContainer::createLayout(QLayout* layout)
{
    if ( container_layout != nullptr )
        return false;
    container_layout = new Layout();
    layout->addItem(container_layout);
    return true;
}

bool BaseMatrixImageViewerWidgetContainer::createWidgets(DataWidget* parent, const type::BaseMatrixImageProxy& proxy, bool)
{
    m_graph = new Graph(parent, proxy);
    return true;
}

void BaseMatrixImageViewerWidgetContainer::insertWidgets() const
{
    if(container_layout && m_graph)
    {
        container_layout->addWidget(m_graph);
    }
}

BaseMatrixImageViewerWidget::BaseMatrixImageViewerWidget(QWidget* parent, const char* name, MyData* d)
    : Inherit(parent,name,d)
{
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // the two following lines avoid to generate the image multiple times by time step
    dirty = false;
    counter = baseData->getCounter();
}

void BaseMatrixImageViewerWidget::readFromData()
{
    SimpleDataWidget<type::BaseMatrixImageProxy, BaseMatrixImageViewerWidgetContainer>::readFromData();

    // the two following lines avoid to generate the image multiple times by time step
    dirty = false;
    counter = baseData->getCounter();
}
} // namespace sofa::gui::qt
