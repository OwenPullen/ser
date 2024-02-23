import json
from ser.params import Params, save_params, load_params
from tempfile import TemporaryDirectory
from os import remove
import pytest
from dataclasses import dataclass, asdict
from pathlib import Path


PARAMS_FILE = Path("params.json")

@pytest.fixture
def data():
    p = Params("test", 1, 2, 3, "ABC123")
    temp_dir = TemporaryDirectory()
    params_path = temp_dir.name / PARAMS_FILE
    with open(params_path, "w") as f:
        json.dump(asdict(p), f, indent=2)
    yield p, temp_dir.name
    temp_dir.cleanup()


def test_params_types(data):
    path = data[1]
    print(path)
    df = load_params(path)
    assert df == data[0]

def test_save_params(data):
    p = data[0]
    path = data[1]
    save_params(path, p)
    with open(path / PARAMS_FILE, "r") as f:
        df = json.load(f)
    assert df == asdict(p)



# def save_params_values():
#     p = Params("test", 1, 2, 3, "ABC123")
#     save_params("{temp_dir}" + "test", p)
#     with open("temp_dir" +  "test/params.json", "r") as f:
#         df = json.load(f)

#     assert data["name"] == "test"
#     assert data["epochs"] == 1
#     assert data["batch_size"] == 2
#     assert data["learning_rate"] == 3
#     assert data["commit"] == "ABC123"

# def load_params_values():