import QtQuick 2.0
import QtQuick.Controls 1.3

SpinBox {
    id: root

    property var dataObject

    enabled: !dataObject.readOnly
    value: dataObject.value
    minimumValue: undefined !== dataObject.properties.min ? dataObject.properties.min : -Number.MAX_VALUE
    maximumValue: undefined !== dataObject.properties.max ? dataObject.properties.max :  Number.MAX_VALUE
    stepSize: undefined !== dataObject.properties.step ? dataObject.properties.step : 1
    decimals: undefined !== dataObject.properties.decimals ? dataObject.properties.decimals : 0

    Binding {
        target: dataObject
        property: "value"
        value: root.value
        when: !dataObject.readOnly
    }
}

