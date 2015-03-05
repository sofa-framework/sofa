import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import SceneData 1.0

GridLayout {
    id: root

    columns: 4

    columnSpacing: 0
    rowSpacing: 0

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

        property string name
        property string type
        property var properties
        property var value
        property bool modified: false

        property bool readOnly: root.readOnly || track.checked

        property var editedValue
        onEditedValueChanged: modified = true;
    }

    property int nameLabelWidth: -1
    readonly property int nameLabelImplicitWidth: nameLabel.implicitWidth

    function updateObject() {
        if(!sceneData)
            return;

        var object = sceneData.object();

        dataObject.name = object.name;
        dataObject.type = object.type;
        dataObject.properties = object.properties;
        dataObject.value = object.value;
        dataObject.editedValue = object.value;

        dataObject.modified = false;
    }

    function updateData() {
        if(!sceneData)
            return;

        sceneData.setValue(dataObject.editedValue);
        updateObject();

        dataObject.modified = false;
    }

    Text {
        id: nameLabel
        text: name + " "
        Layout.preferredWidth: -1 === nameLabelWidth ? implicitWidth : nameLabelWidth
    }

    Loader {
        id: loader
        Layout.fillWidth: true
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

            loader.setSource("qrc:/SofaDataTypes/DataType_" + type + ".qml", {"dataObject": dataObject});
            if(Loader.Ready !== loader.status)
                loader.sourceComponent = dataTypeNotSupportedComponent;

            dataObject.modified = false;
        }
    }

    Button {
        Layout.preferredWidth: 14
        Layout.preferredHeight: Layout.preferredWidth
        checkable: true
        //iconSource: "qrc:/icon/link.png"

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
        Layout.columnSpan: 4
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
