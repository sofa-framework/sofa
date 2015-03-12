import QtQuick 2.0
import QtQuick.Controls 1.3
import QtQuick.Layouts 1.0
import SceneData 1.0

GridLayout {
    id: root

    columns: 4

    columnSpacing: 1
    rowSpacing: 1

    property Scene scene
    property var sceneData
    onSceneDataChanged: updateObject();

    readonly property alias name:       dataObject.name
    readonly property alias type:       dataObject.type
    readonly property alias properties: dataObject.properties
    readonly property alias value:      dataObject.value
    readonly property alias modified:   dataObject.modified

    property bool readOnly: false

    QtObject {
        id: dataObject

        property bool initing: true
        property string name
        property string type
        property var properties
        property var value
        property bool modified: false

        property bool readOnly: initing || root.readOnly || properties.readOnly || track.checked || link.checked

        onValueChanged: modified = true
    }

    property int nameLabelWidth: -1
    readonly property int nameLabelImplicitWidth: nameLabel.implicitWidth

    function updateObject() {
        if(!sceneData)
            return;

        var object              = sceneData.object();

        dataObject.initing      = true;

        dataObject.name         = object.name;
        dataObject.type         = object.type;
        dataObject.properties   = object.properties;
        dataObject.value        = object.value;

        dataObject.initing      = false;

        dataObject.modified     = false;
    }

    function updateData() {
        if(!sceneData)
            return;

        sceneData.setValue(dataObject.value);
        updateObject();

        dataObject.modified = false;
    }

    Text {
        id: nameLabel
        text: name + " "
        Layout.preferredWidth: -1 === nameLabelWidth ? implicitWidth : nameLabelWidth
    }

    Column {
        Layout.fillWidth: true

        RowLayout {
            anchors.left: parent.left
            anchors.right: parent.right

            TextField {
                id: linkTextField
                Layout.fillWidth: true
                visible: link.checked
                placeholderText: "Link: @./path/component.data"
            }
/*
            Image {
                source: "qrc:/icon/ok.png"
            }
*/
        }

        Loader {
            id: loader
            anchors.left: parent.left
            anchors.right: parent.right
            asynchronous: false

            Component.onCompleted: createItem();
            Connections {
                target: root
                onSceneDataChanged: loader.createItem();
            }

            function createItem() {
                var type = root.type;
                var properties = root.properties;

                if(0 === type.length) {
                    type = typeof(root.value);

                    if("object" === type)
                        if(Array.isArray(value))
                            type = "array";
                }

                if("undefined" === type) {
                    loader.source = "";
                    console.warn("Type unknown for data: " + name);
                } else {
                    //console.log(type, name);
                    loader.setSource("qrc:/SofaDataTypes/DataType_" + type + ".qml", {"dataObject": dataObject});
                    if(Loader.Ready !== loader.status)
                        loader.sourceComponent = dataTypeNotSupportedComponent;
                }

                dataObject.modified = false;
            }
        }
    }

    Button {
        id: link
        Layout.preferredWidth: 14
        Layout.preferredHeight: Layout.preferredWidth
        checkable: true

        Image {
            anchors.fill: parent
            source: "qrc:/icon/link.png"
        }
    }

    CheckBox {
        id: track
        Layout.preferredWidth: 20
        Layout.preferredHeight: Layout.preferredWidth
        checked: false

        onClicked: root.updateObject();

        // update every 50ms during simulation
        Timer {
            interval: 50
            repeat: true
            running: scene.play && track.checked
            onTriggered: root.updateObject();
        }

        // update at each step during step-by-step simulation
        Connections {
            target: !scene.play && track.checked ? scene : null
            onStepEnd: root.updateObject();
        }
    }

    Button {
        visible: dataObject.modified
        text: "Undo"
        onClicked: root.updateObject();
    }

    Button {
        Layout.columnSpan: 3
        Layout.fillWidth: true
        visible: dataObject.modified
        text: "Update"
        onClicked: root.updateData();
    }

    /*Image {
        id: requestUpdate
        Layout.preferredWidth: 12
        Layout.preferredHeight: Layout.preferredWidth + 1
        visible: dataObject.modified
        verticalAlignment: Image.AlignBottom
        fillMode: Image.PreserveAspectFit

        source: "qrc:/icon/rightArrow.png"

        MouseArea {
            anchors.fill: parent
            onClicked: root.updateData();
        }
    }*/

    Component {
        id: dataTypeNotSupportedComponent
        TextField {
            readOnly: true
            enabled: !readOnly
            text: "Data type not supported: " + (0 != root.type.length ? root.type : "Unknown")
        }
    }
}
