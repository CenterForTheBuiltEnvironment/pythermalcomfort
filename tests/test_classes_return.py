from dataclasses import dataclass

import pytest

# Import the AutoStrMixin class
from pythermalcomfort.classes_return import AutoStrMixin


@dataclass(repr=False)
class TestDataClass(AutoStrMixin):
    field1: int
    field2: str
    field3: list


def test_autostr_with_dataclass():
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    expected_output = (
        "-------- TestDataClass --------\n"
        "field1 : 42\n"
        "field2 : test\n"
        "field3 : [1, 2, 3]"
    )
    assert str(obj) == expected_output


def test_autostr_with_multiline_field():
    obj = TestDataClass(field1=42, field2="test", field3=["line1", "line2"])
    expected_output = (
        "-------- TestDataClass --------\n"
        "field1 : 42\n"
        "field2 : test\n"
        "field3 : ['line1', 'line2']"
    )
    assert str(obj) == expected_output


def test_autostr_empty_dataclass():
    @dataclass
    class EmptyDataClass(AutoStrMixin):
        pass

    obj = EmptyDataClass()
    expected_output = "-------- EmptyDataClass --------"
    assert str(obj) == expected_output


def test_autostr_repr_method():
    """Test that the __repr__ method returns the same as __str__"""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    expected_output = (
        "-------- TestDataClass --------\n"
        "field1 : 42\n"
        "field2 : test\n"
        "field3 : [1, 2, 3]"
    )
    assert repr(obj) == expected_output
    # Verify that __repr__ returns the same as __str__
    assert repr(obj) == str(obj)


def test_autostr_getitem_method():
    """Test that the __getitem__ method works correctly"""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    assert obj["field1"] == 42
    assert obj["field2"] == "test"
    assert obj["field3"] == [1, 2, 3]


def test_autostr_getitem_method_key_error():
    """Test that __getitem__ raises AttributeError for non-existent keys"""
    obj = TestDataClass(field1=42, field2="test", field3=[1, 2, 3])
    with pytest.raises(KeyError):
        _ = obj["non_existent"]
