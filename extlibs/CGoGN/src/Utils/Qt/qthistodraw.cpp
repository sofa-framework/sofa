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

#include <QtSvg/QSvgGenerator>
#include <QPen>

#include "Utils/Qt/qthistodraw.h"

namespace CGoGN
{

namespace Utils
{

namespace QT
{


RenderHistogram::RenderHistogram(QWidget* parent, Algo::Histogram::Histogram& histo, bool drawAxis )
:QWidget(parent),m_histo(histo), m_drawHisto(true), m_drawQuantiles(false), m_histoFront(true),m_opacity(0.5f),m_drawAxis(drawAxis)
{
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}


void RenderHistogram::axeVals()
{
	int m = m_histo.getMaxBar();
	m_vals_ax_left.clear();
	m_vals_ax_left.reserve(4);
	m_axl_nbd = floor(log10(double(m)));

	int x = int (floor(m / pow(10.0,m_axl_nbd)));
	int y = int(pow(10.0,m_axl_nbd));

	switch(x)
	{
	case 9:
	case 8:
		m_vals_ax_left.push_back(2*y);m_vals_ax_left.push_back(4*y);m_vals_ax_left.push_back(6*y);m_vals_ax_left.push_back(8*y);
		break;
	case 7:
	case 6:
		m_vals_ax_left.push_back(2*y);m_vals_ax_left.push_back(4*y);m_vals_ax_left.push_back(6*y);
		break;
	case 5:
	case 4:
		m_vals_ax_left.push_back(y);m_vals_ax_left.push_back(2*y);m_vals_ax_left.push_back(3*y);m_vals_ax_left.push_back(4*y);
		break;
	case 3:
		m_vals_ax_left.push_back(y);m_vals_ax_left.push_back(2*y);m_vals_ax_left.push_back(3*y);
		break;
	case 2:
		m_vals_ax_left.push_back(y/2);m_vals_ax_left.push_back(y);m_vals_ax_left.push_back(3*y/2);m_vals_ax_left.push_back(2*y);m_vals_ax_left.push_back(5*y/2);
		break;
	case 1:
		{
			m_vals_ax_left.push_back(y/4);m_vals_ax_left.push_back(y/2);m_vals_ax_left.push_back(3*y/4);m_vals_ax_left.push_back(y);
			int yy = y + y/4;
			while (yy <= m)
			{
				m_vals_ax_left.push_back(yy);
				yy += y/4;
			}
		}
		break;
	}
	// store number of digits
	m_axl_nbd++;
}


QSize RenderHistogram::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize RenderHistogram::sizeHint() const
{
    return QSize(400, 200);
}

void RenderHistogram::draw(QPainter& painter)
{
	QRect bb = painter.viewport();

	axeVals();

	unsigned int sz_bottonAxis = 50;

	m_h = bb.height()- (2*m_frameWidth);

	if (m_drawAxis)
		m_h -= sz_bottonAxis;

	m_w = bb.width()- (2*m_frameWidth + 8*m_axl_nbd);

	if (m_histoFront)
	{
		painter.setOpacity(1.0);
		if (m_drawQuantiles)
		{
			drawQuantiles(painter);
			painter.setOpacity(m_opacity);
		}
		if (m_drawHisto)
			drawHisto(painter);
	}
	else
	{
		painter.setOpacity(1.0);
		if (m_drawHisto)
		{
			drawHisto(painter);
			painter.setOpacity(m_opacity);
		}
		if (m_drawQuantiles)
			drawQuantiles(painter);
	}

}

void RenderHistogram::drawHisto(QPainter& painter)
{
	unsigned int nbmax = m_histo.getMaxBar();

	int widthAxl = 8*m_axl_nbd;
	const std::vector<unsigned int>& pop = m_histo.getPopulation();
	m_l = m_w/pop.size();

	qreal op = painter.opacity();
	painter.setOpacity(1.0);

	if (m_drawAxis)
	{
		// draw axes:
		axeVals();

		for (unsigned int i = 0; i< m_vals_ax_left.size(); ++i)
		{
			int h = m_h+m_frameWidth - (m_vals_ax_left[i]*m_h)/nbmax;
			painter.setPen(QColor(100,100,100));
			painter.drawLine(5+widthAxl, h ,widthAxl+m_w+m_frameWidth, h);
			painter.setPen(QColor(0,0,0));
			std::ostringstream oss(std::ostringstream::out);
			oss << m_vals_ax_left[i];
			std::string sv =  oss.str();
			const char* ptr =sv.c_str();
			QString qsv(ptr);

			painter.drawText (1,h-6,widthAxl,15,Qt::AlignRight,qsv);
		}

		painter.setPen(QColor(0,0,0));
		QFont qf = painter.font();
		if (qf.pointSize() > int(m_l))
		{
			qf.setPointSize( m_l-2);
			painter.setFont(qf);
		}
		double lb = (m_histo.getMax() - m_histo.getMin())/pop.size();
		for (unsigned int i=0; i<=pop.size(); ++i)
		{
			double val =  m_histo.getMin() + lb*i;
			std::ostringstream oss(std::ostringstream::out);
			oss <<val;
			std::string sv = oss.str();
			const char* ptr = sv.c_str();
			QString qsv(ptr);
			painter.save();
			painter.translate(widthAxl+m_frameWidth+i*m_l, m_h+ m_frameWidth+8);
			painter.rotate(60);
			painter.drawText (0,0,qsv);
			painter.restore();
		}
	}

	painter.drawLine(widthAxl+m_frameWidth, m_h+m_frameWidth ,widthAxl+m_w+m_frameWidth, m_h+m_frameWidth);
	painter.drawLine(widthAxl+m_frameWidth, m_frameWidth ,widthAxl+m_frameWidth, m_h+m_frameWidth);


	painter.setOpacity(op);


	for (unsigned int i = 0; i< pop.size(); ++i)
	{
		int v = (pop[i]*m_h)/nbmax;

		QRect rect(widthAxl+m_frameWidth+i*m_l, m_h+m_frameWidth-v, m_l, v);
		Geom::Vec3f col = m_histo.colorMap().colorIndex(i);
		painter.setBrush(QColor(255*col[0],255*col[1],255*col[2]));
		painter.drawRect(rect);
	}
}

void RenderHistogram::drawQuantiles(QPainter& painter)
{
	/// vector of intervals
	const std::vector<double>& interv = m_histo.getQuantilesIntervals();
	const std::vector<double>& pop = m_histo.getQuantilesHeights();

	if (interv.empty())
		return;

	int widthAxl = 8*m_axl_nbd;

	painter.setPen(QColor(0,0,0));
	double nbmax = m_histo.getMaxQBar();

	double lw = (m_histo.getMax() - m_histo.getMin()) / double(m_w);
	painter.setBrush(QBrush(QColor(0,0,0)));
	for (unsigned int i = 0; i< pop.size(); ++i)
	{
		double i0 = interv[i];
		if (i0 < m_histo.getMin())
			i0 = m_histo.getMin();
		double i1 = interv[i+1];
		if (i1 < m_histo.getMin())
			i1 = m_histo.getMin();
		if (i1 > m_histo.getMax())
			i1 = m_histo.getMax();

		double x0 =	(i0 - m_histo.getMin()) / lw;
		double xw =	(i1 - i0) / lw;
		int v = (pop[i]*m_h)/nbmax;
		QRect rect(widthAxl+m_frameWidth+int(x0), m_h+m_frameWidth-v, int(xw), v);

		if (i < m_qcolors.size())
		{
			Geom::Vec3f& col = m_qcolors[i];
			painter.setBrush(QBrush(QColor(255*col[0],255*col[1],255*col[2])));
		}
		painter.drawRect(rect);
	}
}


void RenderHistogram::paintEvent(QPaintEvent* /*ev*/)
{
	QPainter painter;
	painter.begin(this);
	draw(painter);
	painter.end();
}


void RenderHistogram::svgExport(const std::string& filename)
{
	 QSvgGenerator generator;
	 generator.setFileName(QString(filename.c_str()));


	 // use screen drawing size

	 generator.setSize(QSize(this->width(),this->height()));
	 generator.setViewBox(QRect(0, 0, this->width(),this->height()));
	 generator.setTitle(tr("Histogram"));
	 generator.setDescription(tr("An SVG histogram created by CGoGN."));

	QPainter painter;
	painter.begin(&generator);

	draw(painter);
 	painter.end();
}

void RenderHistogram::mousePressEvent(QMouseEvent* event)
{
	int x = NONE;
	int widthAxl = 8*m_axl_nbd;

	const std::vector<unsigned int>& pop = m_histo.getPopulation();
	if (m_drawHisto)
	{
		x = (event->x()- widthAxl - m_frameWidth-2)/m_l;
		if ((x>=0) && (x<int(pop.size())))
		{
			int v = (pop[x]*m_h)/m_histo.getMaxBar();
			int y = event->y();
			if ((y<(m_h+m_frameWidth-v)) || (y>m_h+(2*m_frameWidth)))
			{
				x=-1;
			}
		}
	}

	const std::vector<double>& interv = m_histo.getQuantilesIntervals();
	int xx = NONE;
	if ( m_drawQuantiles && !interv.empty())
	{
		double lw = (m_histo.getMax() - m_histo.getMin()) / double(m_w);
		double xd = m_histo.getMin() + double(event->x()- widthAxl - m_frameWidth -2) * lw;

		xx = 0;
		while (xx<int(interv.size()) && (xd > interv[xx]))
			xx++;

		if (xx < int(interv.size()))
		{
			xx--; // back from [1,n] to [0,n-1]

			const std::vector<double>& popq = m_histo.getQuantilesHeights();
			if ((xx>=0) && (xx<int(popq.size())))
			{
				double v = (popq[xx]*m_h)/m_histo.getMaxQBar();
				int y = event->y();
				if ((y<(m_h+m_frameWidth-v)) || (y>m_h+(2*m_frameWidth)))
					xx=-1;
			}
		}
		else
		{
			xx = -1;
		}
	}

	if ((x>=0) || (xx>=0))
	{
		emit clicked(x,xx);
	}
}


void RenderHistogram::setHistoPosition(bool hf)
{
	m_histoFront=hf;
}

void RenderHistogram::setHistoDraw(bool d)
{
	m_drawHisto=d;
}

void RenderHistogram::setQuantilesDraw(bool d)
{
	m_drawQuantiles=d;
}

void RenderHistogram::setOpacity(float op)
{
	m_opacity = op;
}


bool RenderHistogram::getHistoDraw()
{
	return m_drawHisto;
}

bool RenderHistogram::getQuantilesDraw()
{
	return m_drawQuantiles;
}

float RenderHistogram::getOpacity()
{
	return m_opacity;
}



void RenderHistogram::setQuantilesColors(const std::vector<Geom::Vec3f>& colors)
{
	m_qcolors.assign(colors.begin(), colors.end());
}

void RenderHistogram::update()
{
	paintEvent(NULL);
}

}
}
}
