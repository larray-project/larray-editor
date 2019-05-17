from larray_editor.editor import setitem_pattern, setattr_pattern


def test_setitem():
    # new array
    input = 'data = ndtest(10)'
    m = setitem_pattern.match(input)
    assert m is None

    # update array
    input = 'data[:] = 0'
    varname, selection = setitem_pattern.match(input).groups()
    assert varname == 'data'
    assert selection == ':'

    # testing array
    input = 'data[2010:2012] == data2[2010:2012]'
    m = setitem_pattern.match(input)
    assert m is None

    # session - new array
    input = 'ses["data"] = ndtest(10)'
    varname, selection = setitem_pattern.match(input).groups()
    assert varname == 'ses'
    assert selection == '"data"'

    # session - update array
    input = 'ses["data"][:] = 0'
    varname, selection = setitem_pattern.match(input).groups()
    assert varname == 'ses'
    assert selection == '"data"'

    # session - testing array
    input = 'ses["data"] == ses2["data"]'
    m = setitem_pattern.match(input)
    assert m is None


def test_setattr():
    # new array
    input = 'data = ndtest(10)'
    m = setattr_pattern.match(input)
    assert m is None

    # update array metadata
    input = 'data.meta.title = "my array"'
    m = setattr_pattern.match(input)
    assert m is None

    # session - new array
    input = 'ses.data = ndtest(10)'
    varname, attrname = setattr_pattern.match(input).groups()
    assert varname == 'ses'
    assert attrname == 'data'

    # session - update array
    input = 'ses.data[:] = 0'
    varname, attrname = setattr_pattern.match(input).groups()
    assert varname == 'ses'
    assert attrname == 'data'

    # session - update array metadata
    input = 'ses.data.meta.title = "my array"'
    m = setattr_pattern.match(input)
    assert m is None
