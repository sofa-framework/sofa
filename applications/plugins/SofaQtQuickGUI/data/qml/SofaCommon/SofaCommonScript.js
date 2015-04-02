.pragma library

function InstanciateComponent(component, parent, properties, verbose) {
    if(!parent)
        parent = null;

    if(!properties)
        properties = null;

    if(undefined === verbose)
        verbose = true;

    if(3 === component.status) {
        if(verbose)
            console.warn("ERROR creating component:", component.errorString());

        return null;
    }

    var incubator = component.incubateObject(parent, properties);
    incubator.forceCompletion();

    return incubator.object;
}

function InstanciateURLComponent(url, parent, properties, verbose) {
    if(!parent)
        parent = null;

    if(!properties)
        properties = null;

    if(undefined === verbose)
        verbose = true;

    var component = Qt.createComponent(url);
    if(3 === component.status) {
        if(verbose)
            console.warn("ERROR creating component:", component.errorString());

        return null;
    }

    var incubator = component.incubateObject(parent, properties);
    incubator.forceCompletion();

    return incubator.object;
}
