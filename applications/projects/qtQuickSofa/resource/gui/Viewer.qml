import QtQuick 2.0
import Viewer 1.0

Viewer {
    id: root

    Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: root.update()
    }
}
