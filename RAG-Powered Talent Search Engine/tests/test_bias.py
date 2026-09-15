from app.services.bias import demographic_bias_report


def test_bias_report_warns_when_representation_ratio_is_low():
    candidates = [
        {"gender": "Female"},
        {"gender": "Female"},
        {"gender": "Female"},
        {"gender": "Male"},
    ]

    report = demographic_bias_report(candidates)

    assert report["representation_ratio"] < 0.8
    assert report["warning"] is not None


def test_bias_report_handles_missing_field():
    report = demographic_bias_report([{"name": "A"}])
    assert report["representation_ratio"] is None
    assert report["groups"] == []
