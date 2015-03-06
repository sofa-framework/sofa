import QtQuick 2.0
import QtQuick.Controls 1.2

Item {
    id: root

    property var dataObject

    Switch {
        anchors.centerIn: parent
        enabled: !dataObject.readOnly
        checked: dataObject.value
        onCheckedChanged: {
            if(checked !== dataObject.editedValue)
                dataObject.editedValue = checked;
        }
    }
}
