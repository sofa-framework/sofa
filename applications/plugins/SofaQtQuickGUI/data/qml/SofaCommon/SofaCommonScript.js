.pragma library

function InstanciateComponent(url) {
    var component = Qt.createComponent(url);
    var incubator = component.incubateObject();
    incubator.forceCompletion();

    return incubator.object;
}
