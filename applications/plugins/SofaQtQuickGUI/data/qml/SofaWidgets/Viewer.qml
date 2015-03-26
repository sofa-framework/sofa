import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.1
import SofaBasics 1.0
import Viewer 1.0
import Scene 1.0
import "qrc:/SofaCommon/SofaToolsScript.js" as SofaToolsScript

Viewer {
    id: root
    clip: true
	
	Action{
		shortcut: "F5"
		onTriggered: root.viewAll()
	}

    Timer {
        running: true
        repeat: true
        interval: 16
        onTriggered: root.update() // TODO: warning, does not work with multithreaded render loop
    }

    BusyIndicator {
        id: busyIndicator
        anchors.centerIn: parent
        width: 100
        height: width
        running: scene ? scene.status === Scene.Loading : false;
    }

    Component {
        id: cameraComponent

        Camera {

        }
    }

    onScenePathChanged: {
        if(camera)
            camera.destroy();

        camera = cameraComponent.createObject(root);
    }

    Image {
        id: handIcon
        source: "qrc:/icon/hand.png"
        visible: scene ? scene.pickingInteractor.picking : false
        antialiasing: true

        Connections {
            target: scene ? scene.pickingInteractor : null
            onPositionChanged: {
                var position = root.mapFromWorld(scene.pickingInteractor.position)
                if(position.z > 0.0 && position.z < 1.0) {
                    handIcon.x = position.x - 6;
                    handIcon.y = position.y - 2;
                }
            }
        }
    }

    Component.onCompleted: {
        if(scene)
            sceneChanged(scene);
    }

    onSceneChanged: {
        if(scene)
            actor.init();
    }

    property alias actor: actor
    MouseArea {
        anchors.fill: parent
        focus: true
        acceptedButtons: Qt.AllButtons
        propagateComposedEvents: true

        Actor {
            id: actor

            property var previousX: -1
            property var previousY: -1

            property real moveSpeed: 0.00133
            property real turnSpeed: 20.0
            property real zoomSpeed: 1.0

            function init() {
                addMousePressedMapping (Qt.LeftButton, function(mouse) {
                    var nearPosition = root.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 0.0));
                    var farPosition = root.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 1.0));
                    if(scene.pickingInteractor.pick(nearPosition, farPosition.minus(nearPosition))) {
                        var z = root.computeDepth(scene.pickingInteractor.pickedPointPosition());
                        var position = root.projectOnViewPlane(nearPosition, z);
                        scene.pickingInteractor.position = position;

                        setMouseMoveMapping(function(mouse) {
                            var nearPosition = root.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 0.0));
                            var farPosition = root.mapToWorld(Qt.vector3d(mouse.x + 0.5, mouse.y + 0.5, 1.0));
                            var z = root.computeDepth(scene.pickingInteractor.pickedPointPosition());
                            var position = root.projectOnViewPlane(nearPosition, z);
                            scene.pickingInteractor.position = position;
                        });
                    }
                });

                addMouseReleasedMapping(Qt.LeftButton, function(mouse) {
                    scene.pickingInteractor.release();

                    setMouseMoveMapping(null);
                });

                addMousePressedMapping (Qt.RightButton, function(mouse) {
                    previousX = mouse.x;
                    previousY = mouse.y;

                    crosshairGizmo.visible = true;
                    //circleGizmo.visible = true;
                    SofaToolsScript.Tools.overrideCursorShape = Qt.ClosedHandCursor;

                    setMouseMoveMapping(function(mouse) {
                        if(!camera)
                            return;

                        var angleAroundX = 0.0;
                        var angleAroundY = 0.0;
                        var angleAroundZ = 0.0;

                        if(Qt.ControlModifier & mouse.modifiers) {
                            angleAroundZ = (previousX - mouse.x) / 180.0 * Math.PI * turnSpeed;
                        } else {
                            angleAroundX = (previousY - mouse.y) / 180.0 * Math.PI * turnSpeed;
                            angleAroundY = (previousX - mouse.x) / 180.0 * Math.PI * turnSpeed;
                        }

                        camera.turn(angleAroundX, angleAroundY, angleAroundZ);

                        previousX = mouse.x;
                        previousY = mouse.y;
                    });
                });

                addMouseReleasedMapping(Qt.RightButton, function(mouse) {
                    setMouseMoveMapping(null);

                    SofaToolsScript.Tools.overrideCursorShape = 0;
                    //circleGizmo.visible = false;
                    crosshairGizmo.visible = false;
                });

                addMousePressedMapping (Qt.MiddleButton, function(mouse) {
                    previousX = mouse.x;
                    previousY = mouse.y;

                    crosshairGizmo.visible = true;
                    SofaToolsScript.Tools.overrideCursorShape = Qt.ClosedHandCursor;

                    setMouseMoveMapping(function(mouse) {
                        if(!camera)
                            return;

                        var screenToScene = camera.target().minus(camera.eye()).length();

                        var moveX = (mouse.x - previousX) * screenToScene * moveSpeed;
                        var moveY = (mouse.y - previousY) * screenToScene * moveSpeed;
                        camera.move(-moveX, moveY, 0.0);

                        previousX = mouse.x;
                        previousY = mouse.y;
                    });
                });

                addMouseReleasedMapping(Qt.MiddleButton, function(mouse) {
                    setMouseMoveMapping(null);

                    SofaToolsScript.Tools.overrideCursorShape = 0;
                    crosshairGizmo.visible = false;
                });

                setMouseWheelMapping(function(wheel) {
                    if(!camera)
                        return;

                    if(0 === wheel.angleDelta.y)
                        return;

                    var boundary = 2.0;
                    var factor = Math.max(-boundary, Math.min(wheel.angleDelta.y / 120.0, boundary)) / boundary;
                    if(factor < 0.0) {
                        factor = 1.0 + 0.5 * factor;
                        factor /= zoomSpeed;
                    }
                    else {
                        factor = 1.0 + factor;
                        factor *= zoomSpeed;
                    }

                    camera.zoom(factor);

                    wheel.accepted = true;
                });
            }
        }

        onClicked: {
            if(!activeFocus)
                focus = true;

            actor.mouseClicked(mouse);
        }

        onDoubleClicked: {
            if(!activeFocus)
                focus = true;

            actor.mouseDoubleClicked(mouse);
        }

        onPressed: {
            if(!activeFocus)
                focus = true;

            actor.mousePressed(mouse);
        }

        onReleased: {
            actor.mouseReleased(mouse);
        }

        onWheel: {
            actor.mouseWheel(wheel);
        }

        onPositionChanged: {
            actor.mouseMove(mouse);
        }

        Keys.onPressed: {
            if(event.isAutoRepeat) {
                event.accepted = true;
                return;
            }

            if(scene)
                scene.keyPressed(event);

            actor.keyPressed(event);

            event.accepted = true;
        }

        Keys.onReleased: {
            if(event.isAutoRepeat) {
                event.accepted = true;
                return;
            }

            if(scene)
                scene.keyReleased(event);

            actor.keyReleased(event);

            event.accepted = true;
        }
    }

    Item {
        id: crosshairGizmo
        anchors.centerIn: parent
        visible: false

        opacity: 0.75
        property color color: "red"
        property real size: Math.min(root.width, root.height) / 20.0
        property real thickness: 1

        Rectangle {
            anchors.centerIn: parent
            color: crosshairGizmo.color
            width: crosshairGizmo.size
            height: crosshairGizmo.thickness
        }

        Rectangle {
            anchors.centerIn: parent
            color: crosshairGizmo.color
            width: crosshairGizmo.thickness
            height: crosshairGizmo.size
        }
    }

    /*Item {
        id: circleGizmo
        anchors.centerIn: parent
        visible: false

        opacity: 0.75
        property color color: "red"
        property real size: Math.min(root.width, root.height) / 2.0
        property real thickness: 1

        Rectangle {
            anchors.centerIn: parent
            color: "transparent"
            border.color: circleGizmo.color
            border.width: circleGizmo.thickness
            width: circleGizmo.size
            height: width
            radius: width / 2.0
        }
    }*/

    Rectangle {
        id: toolPanel
        color: "lightgrey"
        anchors.top: toolPanelSwitch.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.topMargin: -6
        anchors.bottomMargin: 20
        anchors.rightMargin: -radius
        width: 250
        radius: 5
        visible: false
        opacity: 0.9
        layer.enabled: true

        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.AllButtons
            onWheel: wheel.accepted = true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: toolPanel.radius / 2
                anchors.rightMargin: anchors.margins - toolPanel.anchors.rightMargin
                spacing: 2

                Text {
                    Layout.fillWidth: true
                    text: "Viewer parameters"
                    font.bold: true
                    color: "darkblue"
                }

                Flickable {
                    id: flickable
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    Column {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        spacing: 5

                        GroupBox {
                            id: visualPanel
                            implicitWidth: parent.width
                            title: "Visual"

                            RowLayout {
                                anchors.fill: parent
                                spacing: 0

                                Label {
                                    Layout.fillWidth: true
                                    text: "Wireframe"
                                }

                                Switch {
                                    Layout.fillWidth: true
                                    checked: false
                                    onCheckedChanged: root.wireframe = checked

                                    ToolTip {
                                        anchors.fill: parent
                                        description: "Draw in wireframe mode"
                                    }
                                }
                            }
                        }

                        GroupBox {
                            id: cameraPanel
                            implicitWidth: parent.width

                            title: "Camera"

                            Column {
                                anchors.fill: parent
                                spacing: 0

                                GroupBox {
                                    implicitWidth: parent.width
                                    title: "Mode"
                                    flat: true

                                    RowLayout {
                                        anchors.fill: parent
                                        spacing: 0

                                        Button {
                                            id: orthoButton
                                            Layout.fillWidth: true
                                            Layout.preferredWidth: parent.width

                                            text: "Orthographic"
                                            checkable: true
                                            checked: false
                                            onCheckedChanged: root.camera.orthographic = checked
                                            onClicked: {
                                                checked = true;
                                                perspectiveButton.checked = false;
                                            }

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Orthographic Mode"
                                            }
                                        }

                                        Button {
                                            id: perspectiveButton
                                            Layout.fillWidth: true
                                            Layout.preferredWidth: parent.width

                                            text: "Perspective"
                                            checkable: true
                                            checked: true
                                            onCheckedChanged: root.camera.orthographic = !checked
                                            onClicked: {
                                                checked = true;
                                                orthoButton.checked = false;
                                            }

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Perspective Mode"
                                            }
                                        }
                                    }
                                }

                                GroupBox {
                                    implicitWidth: parent.width
                                    title: "View"
                                    flat: true

                                    GridLayout {
                                        anchors.fill: parent
                                        columns: 2
                                        rowSpacing: 0
                                        columnSpacing: 0

                                        Button {
                                            Layout.fillWidth: true
                                            Layout.preferredWidth: parent.width
                                            text: "Front"

                                            onClicked: if(camera) camera.viewFromFront()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Front View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            Layout.preferredWidth: parent.width
                                            text: "Back"

                                            onClicked: if(camera) camera.viewFromBack()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Back View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            text: "Left"

                                            onClicked: if(camera) camera.viewFromLeft()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Left View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            text: "Right"

                                            onClicked: if(camera) camera.viewFromRight()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Right View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            text: "Top"

                                            onClicked: if(camera) camera.viewFromTop()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Top View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            text: "Bottom"

                                            onClicked: if(camera) camera.viewFromBottom()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Bottom View"
                                            }
                                        }

                                        Button {
                                            Layout.fillWidth: true
                                            Layout.columnSpan: 2
                                            text: "Isometric"

                                            onClicked: if(camera) camera.viewIsometric()

                                            ToolTip {
                                                anchors.fill: parent
                                                description: "Isometric View"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 1
                    color: "darkgrey"
                }
            }
        }
    }

    Image {
        id: toolPanelSwitch
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.topMargin: 26
        anchors.rightMargin: 3
        source: toolPanel.visible ? "qrc:/icon/minus.png" : "qrc:/icon/plus.png"
        width: 12
        height: width

        MouseArea {
            anchors.fill: parent
            propagateComposedEvents: true
            onClicked: toolPanel.visible = !toolPanel.visible
        }
    }
}
