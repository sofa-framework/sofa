import QtQuick 2.0
import Camera 1.0

Camera {
    id: root

    onZoomFactorChanged: {
        smoothedZoomFactor = zoomFactor;
    }

    Behavior on smoothedZoomFactor {
        enabled: root.smoothZoom;
        NumberAnimation {
            duration: 200
            easing.type: Easing.OutQuad
        }
    }
}
