/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/* -------------------------------------------------------- */
#ifndef __WDOUBLELINEEDIT__
#define __WDOUBLELINEEDIT__
/* -------------------------------------------------------- */
#include "SofaGUIQt.h"
#include <qvalidator.h>
#include <qlineedit.h>

#include <QKeyEvent>

/* -------------------------------------------------------- */


class SOFA_SOFAGUIQT_API WDoubleLineEdit : public QLineEdit
{
    Q_OBJECT
    Q_PROPERTY( double minValue READ minValue WRITE setMinValue )
    Q_PROPERTY( double maxValue READ maxValue WRITE setMaxValue )
    Q_PROPERTY( double Value    READ Value    WRITE setValue )
    Q_PROPERTY( int    intValue READ intValue WRITE setIntValue )

protected:

    int               m_iPercent;
    double            m_fMinValue;
    double            m_fMaxValue;
    bool              m_bFirst;
    mutable double    m_fValue;
    QDoubleValidator *m_DblValid;
    double            m_bInternal;

    void              checkValue();
    virtual void      keyPressEvent(QKeyEvent *);
public:

    WDoubleLineEdit(QWidget *parent,const char *name);

    double  minValue() const { return (m_fMinValue);}
    double  getMinValue() { emit(returnPressed()); return minValue();}
    void    setMinValue(double f) {m_fMinValue=f; m_DblValid->setBottom(m_fMinValue); }


    double  maxValue() const { return (m_fMaxValue);}
    double  getMaxValue() { emit(returnPressed()); return maxValue();}
    void    setMaxValue(double f) {m_fMaxValue=f; m_DblValid->setTop(m_fMaxValue); }

    double  Value() const { return (m_fValue);}
    double  getValue() { emit(returnPressed()); return Value();}
    void    setValue(double f);

    int     intValue() const { return static_cast<int>(m_fValue);}
    int     getIntValue() { emit(returnPressed()); return intValue();}
    void	  setIntValue(int f);

    int     valuePercent();

    //Return the value displayed: WARNING!! NO VALIDATION IS MADE!
    double  getDisplayedValue() {return text().toDouble();}
    int     getIntDisplayedValue() {return static_cast<int>(text().toDouble());}

signals:

    void ValueChanged(double);
    void valuePercentChanged(int);

protected slots:

    void slotCalcValue(const QString&);
    void slotCalcValue(double);
    void slotReturnPressed();

public slots:

    void setValuePercent(int p);

};
/* -------------------------------------------------------- */
#endif
/* -------------------------------------------------------- */
