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
#include <math.h>
#include "WDoubleLineEdit.h"
#include <iostream>
using std::cerr; using std::endl;
/* -------------------------------------------------------- */

WDoubleLineEdit::WDoubleLineEdit(QWidget *parent,const char *name) : QLineEdit(parent /*,name */)
{
    setObjectName(name);

    m_iPercent=-1;
    m_fMinValue=0.0;
    m_fMaxValue=1.0;
    m_fValue=0.0;
    m_bFirst=true;
    m_DblValid=new QDoubleValidator(m_fMinValue,m_fMaxValue,20,this);
    setValidator(m_DblValid);

    connect(this,SIGNAL(returnPressed()),
            this,SLOT  (slotReturnPressed()));

    m_bInternal=false;
    //validateAndSet(QString("%1").arg(m_fValue),0,0,0);
    this->setText(QString("%1").arg(m_fValue));
    this->setCursorPosition(0);
    this->setSelection(0, 0);
}
/* -------------------------------------------------------- */
void WDoubleLineEdit::slotReturnPressed()
{
    //cerr<<"WDoubleLineEdit::slotReturnPressed"<<endl;
    m_bInternal=true;

    slotCalcValue(text().toDouble());

}
/* -------------------------------------------------------- */
void WDoubleLineEdit::slotCalcValue(double f)
{
    int    p;

    //cerr << "WDoubleLineEdit::slotCalcValue" << endl;
    if (f < m_fMinValue)
        f=m_fMinValue;
    else if (f > m_fMaxValue)
        f=m_fMaxValue;
    if (f != m_fValue || m_bFirst)
    {
        m_bFirst=false;
        m_fValue=f;
        //cerr << "WDoubleLineEdit::slotCalcValue m_fValue = " << m_fValue << endl;
        emit (ValueChanged(f));
        p=(int)(100.0*(f - m_fMinValue)/(m_fMaxValue - m_fMinValue));
        if (p != m_iPercent)
        {
//      cerr << "m_iPercent = " << m_iPercent << endl;
            emit (valuePercentChanged(p));
            m_iPercent=p;
        }
        update();
    }
//    validateAndSet(QString("%1").arg(m_fValue),0,0,0);
    this->setText(QString("%1").arg(m_fValue));
    this->setCursorPosition(0);
    this->setSelection(0, 0);
}
/* -------------------------------------------------------- */
void WDoubleLineEdit::slotCalcValue(const QString& s)
{
    slotCalcValue(s.toDouble());
}
/* -------------------------------------------------------- */
void WDoubleLineEdit::setValue(double f)
{
    m_bInternal=true;
    slotCalcValue(f);
}

void WDoubleLineEdit::setIntValue(int f)
{
    setValue(static_cast<double>(f));
}
/* -------------------------------------------------------- */
void WDoubleLineEdit::setValuePercent(int p)
{
    if (!m_bInternal)
        setValue(m_fMinValue + (m_fMaxValue - m_fMinValue)*((double)p)/99.0);
    else
        m_bInternal=false;
}
/* -------------------------------------------------------- */
int WDoubleLineEdit::valuePercent()
{
    return ((int)(99.0*(m_fValue - m_fMinValue)/(m_fMaxValue - m_fMinValue)));
}
/* -------------------------------------------------------- */
void WDoubleLineEdit::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
    {
//        validateAndSet(QString("%1").arg(m_fValue),0,0,0);
        this->setText(QString("%1").arg(m_fValue));
        this->setCursorPosition(0);
        this->setSelection(0, 0);
    }
    else
        QLineEdit::keyPressEvent(e);
}
/* -------------------------------------------------------- */
