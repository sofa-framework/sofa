import QtQuick 2.0
import QtQuick.Controls 1.3

TextField {
    id: root

    property var dataObject

    readOnly: dataObject.readOnly
    enabled: !dataObject.readOnly
    text: undefined !== dataObject.value ? dataObject.value.toString() : ""

    Binding {
        target: dataObject
        property: "value"
        value: root.text
        when: !dataObject.readOnly
    }
}
