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


