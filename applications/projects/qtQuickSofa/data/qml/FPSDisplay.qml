import QtQuick 2.0
import QtQuick.Controls 1.0

Label {
    id: root
    text: "FPS: " + timer.fps

    Timer {
        id: timer
        running: true
        repeat: true
        interval: 1

        property int frameCount: 0
        property int previousTime: 0
        property int fps: 0
        onTriggered: {
            frameCount++;
            var currentTime = new Date().getSeconds();

            if(currentTime > previousTime)
            {
                previousTime = currentTime;
                fps = frameCount;
                frameCount = 0;
            }
            else
            {
                previousTime = currentTime;
            }
        }
    }
}
