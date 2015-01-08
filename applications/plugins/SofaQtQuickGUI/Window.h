#ifndef WINDOW_H
#define WINDOW_H

#include <QQuickWindow>
#include <QApplication>

class Window : public QQuickWindow
{
    Q_OBJECT

public:
    explicit Window(QWindow* parent = 0);
    ~Window();

public:
	Q_PROPERTY(int overrideCursorShape READ overrideCursorShape WRITE setOverrideCursorShape NOTIFY overrideCursorShapeChanged);

public:
	int overrideCursorShape() const {return QApplication::overrideCursor() ? QApplication::overrideCursor()->shape() : Qt::ArrowCursor;}
	void setOverrideCursorShape(int newCursorShape);
	
signals:
	void overrideCursorShapeChanged();

};

#endif // WINDOW_H
