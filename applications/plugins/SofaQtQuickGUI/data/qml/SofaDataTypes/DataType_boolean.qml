import QtQuick 2.0
import QtQuick.Controls 1.3

Item {
    id: root
    implicitWidth: control.implicitWidth
    implicitHeight: control.implicitHeight

    property var dataObject

    Switch {
        id: control
        anchors.centerIn: parent
        enabled: !dataObject.readOnly
        onCheckedChanged: {
            if(checked !== dataObject.value)
                dataObject.value = checked;
        }

        Binding {
            target: control
            property: "checked"
            value: dataObject.value
            when: !dataObject.readOnly
        }
    }
}
