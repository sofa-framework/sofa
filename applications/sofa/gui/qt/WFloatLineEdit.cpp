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
#include <math.h>
#include "WFloatLineEdit.h"
#include <iostream>
using std::cerr; using std::endl;
/* -------------------------------------------------------- */

WFloatLineEdit::WFloatLineEdit(QWidget *parent,const char *name) : QLineEdit(parent,name)
{
    m_iPercent=-1;
    m_fMinValue=0.0;
    m_fMaxValue=1.0;
    m_fValue=0.0;
    m_bFirst=true;
    m_DblValid=new QDoubleValidator(m_fMinValue,m_fMaxValue,10,this);
    setValidator(m_DblValid);

    connect(this,SIGNAL(returnPressed()),
            this,SLOT  (slotReturnPressed()));

    m_bInternal=false;
    validateAndSet(QString("%1").arg(m_fValue),0,0,0);
}
/* -------------------------------------------------------- */
void WFloatLineEdit::slotReturnPressed()
{
    //cerr<<"WFloatLineEdit::slotReturnPressed"<<endl;
    m_bInternal=true;

    slotCalcValue(text().toFloat());

}
/* -------------------------------------------------------- */
void WFloatLineEdit::slotCalcValue(float f)
{
    int    p;

    //cerr << "WFloatLineEdit::slotCalcValue" << endl;
    if (f < m_fMinValue)
        f=m_fMinValue;
    else if (f > m_fMaxValue)
        f=m_fMaxValue;
    if (f != m_fValue || m_bFirst)
    {
        m_bFirst=false;
        m_fValue=f;
        //cerr << "WFloatLineEdit::slotCalcValue m_fValue = " << m_fValue << endl;
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
    validateAndSet(QString("%1").arg(m_fValue),0,0,0);
}
/* -------------------------------------------------------- */
void WFloatLineEdit::slotCalcValue(const QString& s)
{
    slotCalcValue(s.toFloat());
}
/* -------------------------------------------------------- */
void WFloatLineEdit::setValue(float f)
{
    m_bInternal=true;
    slotCalcValue(f);
}

void WFloatLineEdit::setIntValue(int f)
{
    setValue(static_cast<float>(f));
}
/* -------------------------------------------------------- */
void WFloatLineEdit::setValuePercent(int p)
{
    if (!m_bInternal)
        setValue(m_fMinValue + (m_fMaxValue - m_fMinValue)*((double)p)/99.0);
    else
        m_bInternal=false;
}
/* -------------------------------------------------------- */
int WFloatLineEdit::valuePercent()
{
    return ((int)(99.0*(m_fValue - m_fMinValue)/(m_fMaxValue - m_fMinValue)));
}
/* -------------------------------------------------------- */
void WFloatLineEdit::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        validateAndSet(QString("%1").arg(m_fValue),0,0,0);
    else
        QLineEdit::keyPressEvent(e);
}
/* -------------------------------------------------------- */

