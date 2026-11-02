from whiteboard_extraction.web.app import create_app


def test_index_loads():
    client = create_app().test_client()
    assert client.get("/").status_code == 200


def test_healthz():
    client = create_app().test_client()
    assert client.get("/healthz").status_code == 200