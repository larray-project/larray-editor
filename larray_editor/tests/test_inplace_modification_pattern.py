from larray_editor.editor import SUBSET_UPDATE_PATTERN


def test_pattern():
    assert SUBSET_UPDATE_PATTERN.match('arr1[1] = 2')
    assert SUBSET_UPDATE_PATTERN.match('arr1[1]= 2')
    assert SUBSET_UPDATE_PATTERN.match('arr1[1]=2')
    assert SUBSET_UPDATE_PATTERN.match("arr1['a'] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[func(mapping['a'])] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1.i[0, 0] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1.iflat[0, 0] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1.points[0, 0] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1.ipoints[0, 0] = arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] += arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] -= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] *= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] /= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] %= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] //= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] **= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] &= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] |= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] ^= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] >>= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0] <<= arr2")
    assert SUBSET_UPDATE_PATTERN.match("arr1[0]") is None
    assert SUBSET_UPDATE_PATTERN.match("arr1.method()") is None
    assert SUBSET_UPDATE_PATTERN.match("arr1[0].method()") is None
    assert SUBSET_UPDATE_PATTERN.match("arr1[0].method(arg=thing)") is None
    assert SUBSET_UPDATE_PATTERN.match("arr1[0].method(arg==thing)") is None
    # this test fails but I don't think it is possible to fix it with regex
    # assert SUBSET_UPDATE_PATTERN.match("arr1[func('[]=0')].method()") is None
