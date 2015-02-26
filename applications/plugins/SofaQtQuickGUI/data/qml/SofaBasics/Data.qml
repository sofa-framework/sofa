import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.0
import SceneData 1.0

GridLayout {
    id: root

    columns: 4

    property Scene scene
    property var sceneData

    readonly property alias name: dataObject.name
    readonly property alias type: dataObject.type
    readonly property alias properties: dataObject.properties
    readonly property alias value: dataObject.value

    QtObject {
        id: dataObject

        property string name
        property string type
        property var properties
        property var value
    }

    property bool readOnly: false

    property int nameLabelWidth: -1
    readonly property int nameLabelImplicitWidth: nameLabel.implicitWidth

    Component.onCompleted: updateItem();
    onSceneDataChanged: updateItem();

    function updateItem() {
        if(!sceneData)
            return;

        var object = sceneData.object();

        dataObject.name = object.name;
        dataObject.type = object.type;
        dataObject.properties = object.properties;
        dataObject.value = object.value;
    }

    function updateData() {
        if(!sceneData)
            return;

        sceneData.setValue(value);
        updateItem();
    }

    Text {
        id: nameLabel
        text: name + " "
        Layout.preferredWidth: -1 === nameLabelWidth ? implicitWidth : nameLabelWidth
    }

    Loader {
        Layout.fillWidth: true
        sourceComponent: evaluateComponent()
    }

    CheckBox {
        id: track
        checked: false

        Connections {
            target: track.checked ? scene : null
            onStepEnd: root.updateItem();
        }
    }

    Button {
        id: requestUpdate
        text: "U"
        Layout.preferredWidth: 16
        Layout.preferredHeight: Layout.preferredWidth
        onClicked: {
            root.updateData();
        }
    }

    function evaluateComponent() {
        var type = root.type;
        if(0 === type.length) {
            type = typeof(root.value);

            if("object" === type)
                if(Array.isArray(value))
                    type = "dynamicArray";
        }

        if("string" === type)
            return stringView;
        else if("boolean" === type)
            return booleanView;
        else if("number" === type)
        {
            if(!readOnly && undefined !== properties.min && undefined !== properties.max)
                return rangedNumberView;
            else
                return numberView;
        }
        else if("staticInDynamicArray" === type)
            return tableView;
        else if("staticArray" === type) {
            if(value.length <= 7)
                return staticArrayView; // TODO: tableView;
            else
                return staticArrayView;
        }
        else if("dynamicArray" === type) {
            if(value.length <= 7)
                return dynamicArrayView; // TODO: tableView;
            else
                return dynamicArrayView;
        }

        return notsupported;
    }

    Component {
        id: notsupported
        TextField {
            readOnly: true
            text: "Data type not supported"
        }
    }

    Component {
        id: stringView
        TextField {
            readOnly: root.readOnly || track.checked
            text: root.value.toString()
        }
    }

    Component {
        id: booleanView
        Switch {
            enabled: !(root.readOnly || track.checked)
            checked: root.value
        }
    }

    Component {
        id: rangedNumberView
        SpinBox {
            enabled: !(root.readOnly || track.checked)
            value: root.value
            minimumValue: root.properties.min
            maximumValue: root.properties.max
            stepSize: root.properties.step ? root.properties.step : 1
        }
    }

    Component {
        id: numberView
        TextField {
            readOnly: root.readOnly || track.checked
            text: root.value
        }
    }

    Component {
        id: tableView
        ListView {
            enabled: !(root.readOnly || track.checked)
            model: ListModel {
                Component.onCompleted: populate();

                function populate() {
                    clear();

                    for(var i = 0; i < root.value.length; ++i)
                        append({"values": root.value[i]});
                }
            }

            delegate: Text {
                text: root.value.toString()
            }

            Connections {
                target: root
                onValueChanged: model.populate();
            }

            //"Static in Dynamic Array: " + root.value
        }
    }

    Component {
        id: staticArrayView
        TextField {
            readOnly: root.readOnly || track.checked
            text: root.value.toString()
        }
    }

    Component {
        id: dynamicArrayView
        TextField {
            readOnly: root.readOnly || track.checked
            text: root.value.toString()
        }
    }
}
