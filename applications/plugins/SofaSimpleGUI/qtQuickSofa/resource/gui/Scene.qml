import QtQuick 2.0
import Scene 1.0

Scene {
    id: base
	/*
    property var timer: Timer {
        interval: 16
        running: true
        repeat: true
        onTriggered: base.step()
    }
	*/

    Component.onCompleted: {
        base.open("C:/MyFiles/Sofa/applications/plugins/SofaSimpleGUI/examples/oneTet.scn");
    }
}
