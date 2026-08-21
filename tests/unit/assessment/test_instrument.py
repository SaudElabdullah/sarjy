from sarjy.contexts.assessment.domain.instrument import Instrument

DEF = {
    "id": "ocean_mini_ipip",
    "version": 1,
    "scale": {
        "min": 1,
        "max": 5,
        "labels": [
            "Very inaccurate",
            "Moderately inaccurate",
            "Neither",
            "Moderately accurate",
            "Very accurate",
        ],
    },
    "traits": {
        "O": "Openness",
        "C": "Conscientiousness",
        "E": "Extraversion",
        "A": "Agreeableness",
        "N": "Neuroticism",
    },
    "bands": {"low": [1, 2.4], "moderate": [2.5, 3.5], "high": [3.6, 5]},
    "items": [
        {"no": 1, "trait": "E", "reverse": False, "text": "I am the life of the party."},
        {"no": 2, "trait": "A", "reverse": False, "text": "I sympathize with others' feelings."},
        {"no": 6, "trait": "E", "reverse": True, "text": "I don't talk a lot."},
    ],
    "scoring": "mean",
}


def test_from_definition_parses_items_and_bands() -> None:
    ins = Instrument.from_definition(DEF)
    assert ins.id == "ocean_mini_ipip" and ins.total_items == 3
    assert ins.item(6).reverse is True and ins.item(1).trait == "E"
    assert ins.bands["moderate"] == (2.5, 3.5)
    assert [i.no for i in ins.items_for_trait("E")] == [1, 6]
    assert ins.scale_labels[4] == "Very accurate"
