import json

import httpx
import pytest
from openai import AsyncOpenAI

from meetple_ai.model import ModelOutputError, OpenAISearchModel


def response_payload(content, status="completed"):
    return {
        "id": "resp_test",
        "created_at": 1,
        "model": "test-model",
        "object": "response",
        "status": status,
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }
        ],
    }


async def test_real_sdk_uses_strict_schema_and_disables_storage(request_data, intent):
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_payload(
                [{"type": "output_text", "text": intent.model_dump_json(), "annotations": []}]
            ),
        )

    async with AsyncOpenAI(
        api_key="test-only",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        actual = await OpenAISearchModel(client, "test-model").interpret(request_data, ["운동"])
    assert actual == intent
    assert sent[0]["store"] is False
    assert sent[0]["text"]["format"]["strict"] is True
    assert sent[0]["text"]["format"]["type"] == "json_schema"
    model_input = json.loads(sent[0]["input"])
    assert model_input["hasLocation"] is True
    assert "latitude" not in model_input
    assert "longitude" not in model_input


@pytest.mark.parametrize(
    "status,content",
    [
        ("completed", [{"type": "refusal", "refusal": "test refusal"}]),
        ("incomplete", []),
    ],
)
async def test_refusal_and_incomplete_output_fail_closed(request_data, status, content):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=response_payload(content, status)))
    async with AsyncOpenAI(
        api_key="test-only", max_retries=0, http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        with pytest.raises(ModelOutputError):
            await OpenAISearchModel(client, "test-model").interpret(request_data, ["운동"])
