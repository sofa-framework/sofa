/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/* -------------------------------------------------------- */
#ifndef __WFLOATLINEEDIT__
#define __WFLOATLINEEDIT__
/* -------------------------------------------------------- */
#include <qvalidator.h>
#include <qlineedit.h>

#ifdef SOFA_QT4
#include <QKeyEvent>
#endif
/* -------------------------------------------------------- */


class WFloatLineEdit : public QLineEdit
{
    Q_OBJECT
    Q_PROPERTY( float minFloatValue READ minFloatValue WRITE setMinFloatValue )
    Q_PROPERTY( float maxFloatValue READ maxFloatValue WRITE setMaxFloatValue )
    Q_PROPERTY( float floatValue    READ floatValue    WRITE setFloatValue )
    Q_PROPERTY( int   intValue      READ intValue	     WRITE setIntValue )

protected:

    int               m_iPercent;
    float             m_fMinValue;
    float             m_fMaxValue;
    bool              m_bFirst;
    mutable float     m_fValue;
    QDoubleValidator *m_DblValid;
    double            m_bInternal;

    void              checkValue();
    virtual void      keyPressEvent(QKeyEvent *);
public:

    WFloatLineEdit(QWidget *parent,const char *name);

    float   minFloatValue() const { return (m_fMinValue);}
    float   getMinFloatValue() { emit(returnPressed()); return minFloatValue();}
    void    setMinFloatValue(float f) {m_fMinValue=f; m_DblValid->setBottom(m_fMinValue); }


    float   maxFloatValue() const { return (m_fMaxValue);}
    float   getMaxFloatValue() { emit(returnPressed()); return maxFloatValue();}
    void    setMaxFloatValue(float f) {m_fMaxValue=f; m_DblValid->setTop(m_fMaxValue); }

    float   floatValue() const { return (m_fValue);}
    float   getFloatValue() { emit(returnPressed()); return floatValue();}
    void    setFloatValue(float f);

    int     intValue() const { return static_cast<int>(m_fValue);}
    int     getIntValue() { emit(returnPressed()); return intValue();}
    void	  setIntValue(int f);

    int     valuePercent();

    //Return the value displayed: WARNING!! NO VALIDATION IS MADE!
    float   getFloatDisplayedValue() {return text().toFloat();};
    int     getIntDisplayedValue() {return static_cast<int>(text().toFloat());};

signals:

    void floatValueChanged(float);
    void valuePercentChanged(int);

protected slots:

    void slotCalcFloatValue(const QString&);
    void slotCalcFloatValue(float);
    void slotReturnPressed();

public slots:

    void setValuePercent(int p);

};
/* -------------------------------------------------------- */
#endif
/* -------------------------------------------------------- */


