from odin.model import Odin_model


def test_create_model():
    model = Odin_model()
    assert(model is not None)


def test_load_model():
    model = Odin_model()
    model.load("set")
