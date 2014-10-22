import QtQuick 2.0
import Viewer 1.0

Viewer {
    id: base

    Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: base.update()
	}
}
