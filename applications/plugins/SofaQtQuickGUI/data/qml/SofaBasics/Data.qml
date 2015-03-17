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
    readonly property alias description:dataObject.description
    readonly property alias type:       dataObject.type
    readonly property alias group:      dataObject.group
    readonly property alias properties: dataObject.properties
    readonly property alias value:      dataObject.value
    readonly property alias modified:   dataObject.modified

    property bool readOnly: false
    property bool showName: true
    property bool showLinkButton: true
    property bool showTrackButton: true
    property alias track: trackButton.checked

    QtObject {
        id: dataObject

        property bool initing: true
        property string name
        property string description
        property string type
        property string group
        property var properties
        property string link
        property var value
        property bool modified: false

        property bool readOnly: initing || root.readOnly || properties.readOnly || trackButton.checked || linkButton.checked

        onValueChanged: modified = true
        onModifiedChanged: if(modified && properties.autoUpdate) root.updateData();
    }

    property int nameLabelWidth: -1
    readonly property int nameLabelImplicitWidth: nameLabel.implicitWidth

    function updateObject() {
        if(!sceneData)
            return;

        var object              = sceneData.object();

        dataObject.initing      = true;

        dataObject.name         = object.name;
        dataObject.description  = object.description;
        dataObject.type         = object.type;
        dataObject.group        = object.group;
        dataObject.properties   = object.properties;
        dataObject.link         = object.link;
        dataObject.value        = object.value;

        dataObject.initing      = false;

        dataObject.modified     = false;
    }

    function updateData() {
        if(!sceneData)
            return;

        sceneData.setValue(dataObject.value);
        updateObject();
    }

    function updateLink() {
        if(!sceneData)
            return;

        sceneData.setLink(linkTextField.visible ? linkTextField.text : "");
        updateObject();
    }

    Text {
        id: nameLabel
        Layout.preferredWidth: -1 === nameLabelWidth ? implicitWidth : nameLabelWidth
        Layout.alignment: Qt.AlignTop
        visible: root.showName
        text: dataObject.name + " "
        font.italic: true

        ToolTip {
            anchors.fill: parent
            description: dataObject.description
        }
    }

    ColumnLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.alignment: Qt.AlignTop

        RowLayout {
            Layout.fillWidth: true
            visible: 0 !== dataObject.name.length && (linkButton.checked || (0 !== dataObject.link.length && !root.showLinkButton))
            spacing: 0

            TextField {
                id: linkTextField
                Layout.fillWidth: true
                placeholderText: "Link: @./path/component." + dataObject.name
                textColor: 0 === dataObject.link.length ? "black" : "green"

                onTextChanged: updateLink();
            }

            Image {
                Layout.preferredWidth: 16
                Layout.preferredHeight: Layout.preferredWidth
                source: 0 === dataObject.link.length ? "qrc:/icon/invalid.png" : "qrc:/icon/correct.png"
            }
        }

        Loader {
            id: loader
            Layout.fillWidth: true
            Layout.fillHeight: true
            asynchronous: false

            Component.onCompleted: createItem();
            Connections {
                target: root
                onSceneDataChanged: {
                    if(root.sceneData)
                        loader.createItem();
                    else
                        loader.destroyItem();
                }
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

                //console.log(type, name);

                if("undefined" === type) {
                    loader.source = "";
                    console.warn("Type unknown for data: " + name);
                } else {
                    loader.setSource("qrc:/SofaDataTypes/DataType_" + type + ".qml", {"dataObject": dataObject});
                    if(Loader.Ready !== loader.status)
                        loader.sourceComponent = dataTypeNotSupportedComponent;
                }

                dataObject.modified = false;
            }

            function destroyItem() {
                loader.setSource("");

                dataObject.modified = false;
            }
        }
    }

    Item {
        Layout.preferredWidth: 20
        Layout.preferredHeight: Layout.preferredWidth
        Layout.alignment: Qt.AlignTop
        visible: root.showLinkButton

        Button {
            id: linkButton
            anchors.fill: parent
            anchors.margins: 3
            checkable: true

            ToolTip {
                anchors.fill: parent
                description: "Link the data with another"
            }

            onClicked: updateLink()

            Image {
                anchors.fill: parent
                source: "qrc:/icon/link.png"
            }
        }
    }

    CheckBox {
        id: trackButton
        Layout.preferredWidth: 20
        Layout.preferredHeight: Layout.preferredWidth
        Layout.alignment: Qt.AlignTop
        visible: root.showTrackButton && 0 !== dataObject.name.length
        checked: false

        onClicked: root.updateObject();

        ToolTip {
            anchors.fill: parent
            description: "Track the data value during simulation"
        }

        // update every 50ms during simulation
        Timer {
            interval: 50
            repeat: true
            running: scene.play && trackButton.checked
            onTriggered: root.updateObject();
        }

        // update at each step during step-by-step simulation
        Connections {
            target: !scene.play && trackButton.checked ? scene : null
            onStepEnd: root.updateObject();
        }
    }

    Button {
        visible: dataObject.modified
        text: "Undo"
        onClicked: root.updateObject();

        ToolTip {
            anchors.fill: parent
            description: "Undo changes in the data value"
        }
    }

    Button {
        Layout.columnSpan: 3
        Layout.fillWidth: true
        visible: dataObject.modified
        text: "Update"
        onClicked: root.updateData();

        ToolTip {
            anchors.fill: parent
            description: "Update the data value"
        }
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
