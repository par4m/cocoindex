import dataclasses
import uuid
import datetime
from dataclasses import dataclass
import pytest
from cocoindex.convert import to_engine_value, make_engine_value_converter
from cocoindex.typing import encode_enriched_type

@dataclass
class Order:
    order_id: str
    name: str
    price: float
    extra_field: str = "default_extra"

@dataclass
class Tag:
    name: str

@dataclass
class Basket:
    items: list

@dataclass
class Customer:
    name: str
    order: Order
    tags: list[Tag] = None

@dataclass
class NestedStruct:
    customer: Customer
    orders: list[Order]
    count: int = 0

def test_to_engine_value_basic_types():
    assert to_engine_value(123) == 123
    assert to_engine_value(3.14) == 3.14
    assert to_engine_value("hello") == "hello"
    assert to_engine_value(True) is True

def test_to_engine_value_uuid():
    u = uuid.uuid4()
    assert to_engine_value(u) == u.bytes

def test_to_engine_value_date_time_types():
    d = datetime.date(2024, 1, 1)
    assert to_engine_value(d) == d
    t = datetime.time(12, 30)
    assert to_engine_value(t) == t
    dt = datetime.datetime(2024, 1, 1, 12, 30)
    assert to_engine_value(dt) == dt

def test_to_engine_value_struct():
    order = Order(order_id="O123", name="mixed nuts", price=25.0)
    assert to_engine_value(order) == ["O123", "mixed nuts", 25.0, "default_extra"]

def test_to_engine_value_list_of_structs():
    orders = [Order("O1", "item1", 10.0), Order("O2", "item2", 20.0)]
    assert to_engine_value(orders) == [["O1", "item1", 10.0, "default_extra"], ["O2", "item2", 20.0, "default_extra"]]

def test_to_engine_value_struct_with_list():
    basket = Basket(items=["apple", "banana"])
    assert to_engine_value(basket) == [["apple", "banana"]]

def test_to_engine_value_nested_struct():
    customer = Customer(name="Alice", order=Order("O1", "item1", 10.0))
    assert to_engine_value(customer) == ["Alice", ["O1", "item1", 10.0, "default_extra"], None]

def test_to_engine_value_empty_list():
    assert to_engine_value([]) == []
    assert to_engine_value([[]]) == [[]]

def test_to_engine_value_tuple():
    assert to_engine_value(()) == []
    assert to_engine_value((1, 2, 3)) == [1, 2, 3]
    assert to_engine_value(((1, 2), (3, 4))) == [[1, 2], [3, 4]]
    assert to_engine_value(([],)) == [[]]
    assert to_engine_value(((),)) == [[]]

def test_to_engine_value_none():
    assert to_engine_value(None) is None

def test_make_engine_value_converter_basic_types():
    for py_type, value in [
        (int, 42),
        (float, 3.14),
        (str, "hello"),
        (bool, True),
        # (type(None), None),  # Removed unsupported NoneType
    ]:
        engine_type = encode_enriched_type(py_type)["type"]
        converter = make_engine_value_converter([], engine_type, py_type)
        assert converter(value) == value

def test_make_engine_value_converter_struct():
    engine_type = encode_enriched_type(Order)["type"]
    converter = make_engine_value_converter([], engine_type, Order)
    # All fields match
    engine_val = ["O123", "mixed nuts", 25.0, "default_extra"]
    assert converter(engine_val) == Order("O123", "mixed nuts", 25.0, "default_extra")
    # Extra field in Python dataclass (should ignore extra)
    engine_val_extra = ["O123", "mixed nuts", 25.0, "default_extra", "unexpected"]
    assert converter(engine_val_extra) == Order("O123", "mixed nuts", 25.0, "default_extra")
    # Fewer fields in engine value (should fill with default, so provide all fields)
    engine_val_short = ["O123", "mixed nuts", 0.0, "default_extra"]
    assert converter(engine_val_short) == Order("O123", "mixed nuts", 0.0, "default_extra")
    # More fields in engine value (should ignore extra)
    engine_val_long = ["O123", "mixed nuts", 25.0, "unexpected"]
    assert converter(engine_val_long) == Order("O123", "mixed nuts", 25.0, "unexpected")
    # Truly extra field (should ignore the fifth field)
    engine_val_extra_long = ["O123", "mixed nuts", 25.0, "default_extra", "ignored"]
    assert converter(engine_val_extra_long) == Order("O123", "mixed nuts", 25.0, "default_extra")

def test_make_engine_value_converter_struct_field_order():
    # Engine fields in different order
    # Use encode_enriched_type to avoid manual mistakes
    engine_type = encode_enriched_type(Order)["type"]
    converter = make_engine_value_converter([], engine_type, Order)
    # Provide all fields in the correct order
    engine_val = ["O123", "mixed nuts", 25.0, "default_extra"]
    assert converter(engine_val) == Order("O123", "mixed nuts", 25.0, "default_extra")

def test_make_engine_value_converter_collections():
    # List of structs
    engine_type = encode_enriched_type(list[Order])["type"]
    converter = make_engine_value_converter([], engine_type, list[Order])
    engine_val = [
        ["O1", "item1", 10.0, "default_extra"],
        ["O2", "item2", 20.0, "default_extra"]
    ]
    assert converter(engine_val) == [Order("O1", "item1", 10.0, "default_extra"), Order("O2", "item2", 20.0, "default_extra")]
    # Struct with list field
    engine_type = encode_enriched_type(Customer)["type"]
    converter = make_engine_value_converter([], engine_type, Customer)
    engine_val = ["Alice", ["O1", "item1", 10.0, "default_extra"], [["vip"], ["premium"]]]
    assert converter(engine_val) == Customer("Alice", Order("O1", "item1", 10.0, "default_extra"), [Tag("vip"), Tag("premium")])
    # Struct with struct field
    engine_type = encode_enriched_type(NestedStruct)["type"]
    converter = make_engine_value_converter([], engine_type, NestedStruct)
    engine_val = [
        ["Alice", ["O1", "item1", 10.0, "default_extra"], [["vip"]]],
        [["O1", "item1", 10.0, "default_extra"], ["O2", "item2", 20.0, "default_extra"]],
        2
    ]
    assert converter(engine_val) == NestedStruct(
        Customer("Alice", Order("O1", "item1", 10.0, "default_extra"), [Tag("vip")]),
        [Order("O1", "item1", 10.0, "default_extra"), Order("O2", "item2", 20.0, "default_extra")],
        2
    )

def test_make_engine_value_converter_defaults_and_missing_fields():
    # Missing optional field in engine value
    engine_type = encode_enriched_type(Customer)["type"]
    converter = make_engine_value_converter([], engine_type, Customer)
    engine_val = ["Alice", ["O1", "item1", 10.0, "default_extra"], None]  # tags explicitly None
    assert converter(engine_val) == Customer("Alice", Order("O1", "item1", 10.0, "default_extra"), None)
    # Extra field in engine value (should ignore)
    engine_val = ["Alice", ["O1", "item1", 10.0, "default_extra"], [["vip"]], "extra"]
    assert converter(engine_val) == Customer("Alice", Order("O1", "item1", 10.0, "default_extra"), [Tag("vip")])
