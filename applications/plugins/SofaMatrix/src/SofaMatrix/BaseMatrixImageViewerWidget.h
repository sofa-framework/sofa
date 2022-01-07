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

#include <QGraphicsView>
#include <sofa/gui/qt/SimpleDataWidget.h>

#include <SofaMatrix/BaseMatrixImageProxy.h>

namespace sofa::gui::qt
{

/// A QGraphicsScene showing a QImage
class BaseMatrixImageViewerGraphScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit BaseMatrixImageViewerGraphScene(QImage* im, QObject *parent = nullptr);

    void drawBackground(QPainter *painter, const QRectF &rect) override;

private:
    const QImage *m_image;
};

/// A QGraphicsView showing a scene containing the image to display
/// The image is generated from the BaseMatrix in this class
class BaseMatrixImageViewerGraphWidget : public QGraphicsView
{
    Q_OBJECT
public:
    BaseMatrixImageViewerGraphWidget(QWidget * parent, const type::BaseMatrixImageProxy& plane);

    // Generation of the QImage from the BaseMatrix
    void readFromData(const type::BaseMatrixImageProxy& proxy);

    ~BaseMatrixImageViewerGraphWidget() override {}
protected:

    void fitInView();

    void resizeEvent ( QResizeEvent* /*resizeevent*/) override;

    const type::BaseMatrixImageProxy* m_imageplane { nullptr };
    QImage m_image;
    BaseMatrixImageViewerGraphScene* m_scene { nullptr };
};

/// Qt definition of the widget, compatible with SimpleDataWidget
class BaseMatrixImageViewerWidgetContainer: public QObject
{
    Q_OBJECT;
public:
    typedef BaseMatrixImageViewerGraphWidget Graph;
    Graph* m_graph { nullptr };

    typedef QHBoxLayout Layout;
    Layout* container_layout { nullptr };

    void writeToData(type::BaseMatrixImageProxy& d);

    void readFromData(const type::BaseMatrixImageProxy& proxy) const;

    void setReadOnly(bool /*readOnly*/);

    bool createLayout( DataWidget* parent );

    bool createLayout( QLayout* layout);

    bool createWidgets(DataWidget* parent, const type::BaseMatrixImageProxy& proxy, bool /*readOnly*/);

    void insertWidgets() const;
};

/// A Qt Data widget to display a BaseMatrix (from a BaseMatrixProxy) as a bitmap in the GUI
class BaseMatrixImageViewerWidget : public SimpleDataWidget<type::BaseMatrixImageProxy, BaseMatrixImageViewerWidgetContainer>
{
    using Inherit = SimpleDataWidget<type::BaseMatrixImageProxy, BaseMatrixImageViewerWidgetContainer>;
    using MyData = sofa::core::objectmodel::Data<type::BaseMatrixImageProxy>;

public:
    BaseMatrixImageViewerWidget(QWidget* parent,const char* name, MyData* d);
    unsigned int sizeWidget() override {return 1;}
    unsigned int numColumnWidget() override {return 1;}

    void readFromData() override;
};

} // namespace sofa::gui::qt
