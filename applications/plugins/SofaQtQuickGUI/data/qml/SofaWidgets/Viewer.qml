import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Controls.Styles 1.3
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import QtGraphicalEffects 1.0
import SofaBasics 1.0
import SofaInteractors 1.0
import Viewer 1.0
import Scene 1.0
import "qrc:/SofaCommon/SofaApplicationScript.js" as SofaApplicationScript

Viewer {
    id: root
    clip: true
    backgroundColor: "#FF404040"
    backgroundImageSource: "qrc:/icon/sofaLogoAlpha.png"
    wireframe: false
    culling: true
    antialiasing: false

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

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        acceptedButtons: Qt.AllButtons

        property alias interactor: interactorLoader.item
        Loader {
            id: interactorLoader
            sourceComponent: SofaApplicationScript.Application.interactorComponent
            onLoaded: {
                var interactor = item;
                interactor.scene   = root.scene;
                interactor.viewer  = root;
                interactor.init();
            }
        }

        onClicked: {
            if(!activeFocus)
                focus = true;

            if(interactor)
                interactor.mouseClicked(mouse);
        }

        onDoubleClicked: {
            if(!activeFocus)
                focus = true;

            if(interactor)
                interactor.mouseDoubleClicked(mouse);
        }

        onPressed: {
            if(!activeFocus)
                focus = true;

            if(interactor)
                interactor.mousePressed(mouse);
        }

        onReleased: {
            if(interactor)
                interactor.mouseReleased(mouse);
        }

        onWheel: {
            if(interactor)
                interactor.mouseWheel(wheel);
        }

        onPositionChanged: {
            if(interactor)
                interactor.mouseMove(mouse);
        }

        Keys.onPressed: {
            if(event.isAutoRepeat) {
                event.accepted = true;
                return;
            }

            if(scene)
                scene.keyPressed(event);

            if(interactor)
                interactor.keyPressed(event);

            event.accepted = true;
        }

        Keys.onReleased: {
            if(event.isAutoRepeat) {
                event.accepted = true;
                return;
            }

            if(scene)
                scene.keyReleased(event);

            if(interactor)
                interactor.keyReleased(event);

            event.accepted = true;
        }
    }

    readonly property alias crosshairGizmo: crosshairGizmo
    Item {
        id: crosshairGizmo
        anchors.centerIn: parent
        visible: false

        function show() {
            popAnimation.complete();
            visible = true;
        }

        function hide() {
            popAnimation.complete();
            visible = false;
        }

        function pop() {
            popAnimation.restart();
        }

        SequentialAnimation {
            id: popAnimation

            ScriptAction    {script: {crosshairGizmo.visible = true;}}
            NumberAnimation {target:  crosshairGizmo; properties: "opacity"; from: 1.0; to: 0.0; duration: 2000;}
            ScriptAction    {script: {crosshairGizmo.visible = false; crosshairGizmo.opacity = crosshairGizmo.defaultOpacity;}}
        }

        readonly property real defaultOpacity: 0.75
        opacity: defaultOpacity
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
            onWheel: {
                //flickable.;
                wheel.accepted = true
            }

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

                            GridLayout {
                                anchors.fill: parent
                                columnSpacing: 0
                                rowSpacing: 2
                                columns: 2

                                Label {
                                    Layout.fillWidth: true
                                    text: "Wireframe"
                                }

                                Switch {
                                    id: wireframeSwitch
                                    Layout.alignment: Qt.AlignCenter
                                    Component.onCompleted: checked = root.wireframe
                                    onCheckedChanged: root.wireframe = checked

                                    ToolTip {
                                        anchors.fill: parent
                                        description: "Draw in wireframe mode"
                                    }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: "Culling"
                                }

                                Switch {
                                    id: cullingSwitch
                                    Layout.alignment: Qt.AlignCenter
                                    Component.onCompleted: checked = root.culling
                                    onCheckedChanged: root.culling = checked

                                    ToolTip {
                                        anchors.fill: parent
                                        description: "Enable culling"
                                    }
                                }
/*
                                // TODO: antialiasing not implemented yet
                                Label {
                                    Layout.fillWidth: true
                                    text: "Antialiasing"
                                }

                                Switch {
                                    id: antialiasingSwitch
                                    Layout.alignment: Qt.AlignCenter
                                    Component.onCompleted: checked = root.antialiasing
                                    onCheckedChanged: root.antialiasing = checked

                                    ToolTip {
                                        anchors.fill: parent
                                        description: "Enable Antialiasing"
                                    }
                                }
*/
                                Label {
                                    Layout.fillWidth: true
                                    text: "Background"
                                }

                                Rectangle {
                                    Layout.preferredWidth: wireframeSwitch.implicitWidth
                                    Layout.preferredHeight: wireframeSwitch.implicitHeight
                                    Layout.alignment: Qt.AlignCenter
                                    color: "darkgrey"
                                    radius: 2

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: backgroundColorPicker.open()
                                    }

                                    ColorDialog {
                                        id: backgroundColorPicker
                                        title: "Please choose a background color"
                                        showAlphaChannel: true

                                        property color previousColor
                                        Component.onCompleted: {
                                            previousColor = root.backgroundColor;
                                            color = previousColor;
                                            currentColor = color;
                                        }

                                        onCurrentColorChanged: root.backgroundColor = currentColor

                                        onAccepted: previousColor = currentColor
                                        onRejected: currentColor = previousColor
                                    }

                                    Rectangle {
                                        anchors.fill: parent
                                        anchors.margins: 2
                                        color: Qt.rgba(root.backgroundColor.r, root.backgroundColor.g, root.backgroundColor.b, 1.0)

                                        ToolTip {
                                            anchors.fill: parent
                                            description: "Background color"
                                        }
                                    }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: "Logo"
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 0

                                    TextField {
                                        id: logoTextField
                                        Layout.fillWidth: true
                                        Component.onCompleted: text = root.backgroundImageSource
                                        onAccepted: root.backgroundImageSource = text
                                    }

                                    Button {
                                        Layout.preferredWidth: 22
                                        Layout.preferredHeight: Layout.preferredWidth
                                        iconSource: "qrc:/icon/open.png"

                                        onClicked: openLogoDialog.open()

                                        FileDialog {
                                            id: openLogoDialog
                                            title: "Please choose a logo"
                                            selectFolder: true
                                            selectMultiple: false
                                            selectExisting: true
                                            property var resultTextField
                                            onAccepted: {
                                                logoTextField.text = Qt.resolvedUrl(fileUrl)
                                                logoTextField.accepted();
                                            }
                                        }
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
