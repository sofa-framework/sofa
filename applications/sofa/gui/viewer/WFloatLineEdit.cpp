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
    m_DblValid=new QDoubleValidator(m_fMinValue,m_fMaxValue,6,this);
    setValidator(m_DblValid);
    connect(this,SIGNAL(returnPressed()),
            this,SLOT(slotReturnPressed()));
    m_bInternal=false;
    validateAndSet(QString("%1").arg(m_fValue),0,0,0);
}
/* -------------------------------------------------------- */
void WFloatLineEdit::slotReturnPressed()
{
    //cerr<<"WFloatLineEdit::slotReturnPressed"<<endl;
    m_bInternal=true;
    slotCalcFloatValue(text().toFloat());
}
/* -------------------------------------------------------- */
void WFloatLineEdit::slotCalcFloatValue(float f)
{
    int    p;

    //cerr << "WFloatLineEdit::slotCalcFloatValue" << endl;
    if (f < m_fMinValue)
        f=m_fMinValue;
    else if (f > m_fMaxValue)
        f=m_fMaxValue;
    if (f != m_fValue || m_bFirst)
    {
        m_bFirst=false;
        m_fValue=f;
        //cerr << "WFloatLineEdit::slotCalcFloatValue m_fValue = " << m_fValue << endl;
        emit (floatValueChanged(f));
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
void WFloatLineEdit::slotCalcFloatValue(const QString& s)
{
    slotCalcFloatValue(s.toFloat());
}
/* -------------------------------------------------------- */
void WFloatLineEdit::setFloatValue(float f)
{
    m_bInternal=true;
    slotCalcFloatValue(f);
}

void WFloatLineEdit::setIntValue(int f)
{
    setFloatValue(static_cast<float>(f));
}
/* -------------------------------------------------------- */
void WFloatLineEdit::setValuePercent(int p)
{
    if (!m_bInternal)
        setFloatValue(m_fMinValue + (m_fMaxValue -
                m_fMinValue)*((double)p)/99.0);
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

