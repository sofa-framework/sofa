import QtQuick 2.0
import QtQuick.Controls 1.1
import Viewer 1.0
import Scene 1.0

Viewer {
    id: root

    Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: root.update()
    }

    BusyIndicator {
        anchors.centerIn: parent
        width: 100
        height: width
        running: scene.status == Scene.Loading
    }
}
