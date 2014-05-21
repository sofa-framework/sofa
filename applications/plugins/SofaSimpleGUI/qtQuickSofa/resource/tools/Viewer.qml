import QtQuick 2.0
import Viewer 1.0

Viewer {
    id: base

    SequentialAnimation on t {
        NumberAnimation { to: 1; duration: 2500; easing.type: Easing.InQuad }
        NumberAnimation { to: 0; duration: 2500; easing.type: Easing.OutQuad }
        loops: Animation.Infinite
        running: true
    }
}
