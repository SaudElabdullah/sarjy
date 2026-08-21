from sarjy.contexts.guardrails.infrastructure.gemini_classifier import (
    ClassifierOut,
    GeminiClassifier,
)


class FakeLLM:
    async def generate_json(self, req, schema):  # type: ignore[no-untyped-def]
        assert schema is ClassifierOut and req.temperature == 0
        return ClassifierOut(
            category="medical_legal_financial", is_injection=False, severity=1, confidence=0.8
        )

    def stream(self, req):  # type: ignore[no-untyped-def]
        raise NotImplementedError


async def test_classifier_maps_output() -> None:
    c = await GeminiClassifier(FakeLLM()).classify(["how many pills"])  # type: ignore[arg-type]
    assert c.category == "medical_legal_financial" and c.confidence == 0.8
