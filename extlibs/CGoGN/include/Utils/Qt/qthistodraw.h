/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __QT_HISTO_DRAW__
#define __QT_HISTO_DRAW__


#include <QWidget>
#include <QPainter>
#include <QMouseEvent>

#include "Utils/Qt/qtpopup.h"
#include "Algo/Histogram/histogram.h"

namespace CGoGN
{

namespace Utils
{

namespace QT
{


class RenderHistogram : public QWidget
{
    Q_OBJECT

	static const int m_frameWidth = 10;


	const Algo::Histogram::Histogram& m_histo;
    std::vector<Geom::Vec3f> m_qcolors;

    unsigned int m_max;
	int m_h;
	int m_w;
	unsigned int m_l;

	std::vector<int> m_vals_ax_left;

	int m_axl_nbd;
	bool m_drawHisto;
	bool m_drawQuantiles;
	bool m_histoFront;
	float m_opacity;
	bool m_drawAxis;

	/// draw all in painter
	void draw(QPainter& painter);

	/// compute left axis value
	void axeVals();

public:

	static const unsigned int NONE = 0xffffffff;

	/**
	 * constructor
	 * @param parent parent widget
	 * @param histo histogram to draw
	 */
    RenderHistogram(QWidget* parent, Algo::Histogram::Histogram& histo, bool drawAxis=true );

    /// minimum size
    virtual QSize minimumSizeHint() const;

    /// size at launch
    virtual QSize sizeHint() const;

    /**
     * set color table for quantiles drawing
     * @param colors vector of colors
     */
    void setQuantilesColors(const std::vector<Geom::Vec3f>& colors);

   /***
    * svg export
    * @param filename file name of svg file
    */
    void svgExport(const std::string& filename);

    /**
     * set histo position (front or back)
     * @param hf if true histo is in front of the quantille
     */
    void setHistoPosition(bool hf);

    /**
     * define if histogram is drawn
     */
    void setHistoDraw(bool d);

    /**
     * define if quantille is drawn
     */
    void setQuantilesDraw(bool d);

    /**
	 * get bool value that indicate drawing of histogram
	 */
    bool getHistoDraw();

    /**
	 * get bool value that indicate drawing of quantille
	 */
    bool getQuantilesDraw();

    /**
	 * get opacity value
	 */
    float getOpacity();

    /**
     * define the opacity if the two graphs are drawn
     */
    void setOpacity(float op);

    /**
     * update drawing
     */
    void update();

    signals:
    /**
     * emitted signal when a column of histogram is clicked
     * @param i column of histo (NONE if none)
     * @param j column of quantileq (NONE if none)
     */
    void clicked(unsigned int, unsigned int);



protected:

    /// draw the histogram in painter widget
    void drawHisto(QPainter& painter);

    /// draw the quatilles in painter widget
    void drawQuantiles(QPainter& painter);

    /// functinn calles when widget need to be redraw
   void paintEvent(QPaintEvent *event);

   /// draw the histogram in painter widget
   void mousePressEvent(QMouseEvent* event);

};

}
}
}


#endif
