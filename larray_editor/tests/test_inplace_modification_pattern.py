from larray_editor.editor import UPDATE_VARIABLE_PATTERN


def test_pattern():
    matching_patterns = [
        'arr1[1] = 2',
        'arr1[1]= 2',
        'arr1[1]=2',
        "arr1['a'] = arr2",
        "arr1[func(mapping['a'])] = arr2",
        "arr1.i[0, 0] = arr2",
        "arr1.iflat[0, 0] = arr2",
        "arr1.points[0, 0] = arr2",
        "arr1.ipoints[0, 0] = arr2",
        "arr1[0] += arr2",
        "arr1[0] -= arr2",
        "arr1[0] *= arr2",
        "arr1[0] /= arr2",
        "arr1[0] %= arr2",
        "arr1[0] //= arr2",
        "arr1[0] **= arr2",
        "arr1[0] &= arr2",
        "arr1[0] |= arr2",
        "arr1[0] ^= arr2",
        "arr1[0] >>= arr2",
        "arr1[0] <<= arr2",
        "arr1.data[1] = 2",
        "arr1.data = np.array([1, 2, 3])"
    ]
    for pattern in matching_patterns:
        match = UPDATE_VARIABLE_PATTERN.match(pattern)
        assert match is not None and match.group('variable') == 'arr1'

    for pattern in [
        "df.loc[1] = 2",
        "df.iloc[1] = 2"
    ]:
        match = UPDATE_VARIABLE_PATTERN.match(pattern)
        assert match is not None and match.group('variable') == 'df'

    # no match
    for pattern in [
        "arr1[0]",
        "arr1.method()",
        "arr1[0].method()",
        "arr1[0].method(arg=thing)",
        "arr1[0].method(arg==thing)",
        # this test fails but I don't think it is possible to fix it with regex
        # "arr1[func('[]=0')].method()"
    ]:
        assert UPDATE_VARIABLE_PATTERN.match(pattern) is None
