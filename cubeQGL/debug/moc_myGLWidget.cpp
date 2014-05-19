/****************************************************************************
** Meta object code from reading C++ file 'myGLWidget.h'
**
** Created: Fri 9. May 13:34:36 2014
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../myGLWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'myGLWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_myGLWidget[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_myGLWidget[] = {
    "myGLWidget\0\0timeOutSlot()\0"
};

const QMetaObject myGLWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_myGLWidget,
      qt_meta_data_myGLWidget, 0 }
};

const QMetaObject *myGLWidget::metaObject() const
{
    return &staticMetaObject;
}

void *myGLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_myGLWidget))
        return static_cast<void*>(const_cast< myGLWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int myGLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: timeOutSlot(); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
