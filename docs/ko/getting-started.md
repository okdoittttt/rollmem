# 시작하기

## 설치

```bash
pip install rollmem
```

Python 3.9 이상이 필요합니다. 런타임 의존성은 없습니다.

## 첫 메모리 만들기

`RollingMemory`를 생성하고, 턴을 추가하고, 컨텍스트를 읽어 봅니다:

```python
from rollmem import RollingMemory

mem = RollingMemory(max_tokens=2000)

mem.add_user_message("Hi, I'm planning a trip to Korea.")
mem.add_assistant_message("Great! When are you going?")

print(mem.get_context())
# user: Hi, I'm planning a trip to Korea.
# assistant: Great! When are you going?
```

`add_user_message`, `add_assistant_message`, `add_system_message`,
`add_tool_message`는 익스포트된 `USER`, `ASSISTANT`, `SYSTEM`, `TOOL` role
상수를 사용하는 `add_message`의 편의 래퍼입니다. 그 외의 role 문자열도
`add_message`로 추가할 수 있습니다.

에이전트형 턴은 tool call과 그 결과를 담을 수 있습니다:

```python
from rollmem import ASSISTANT, ToolCall

mem.add_message(
    ASSISTANT,
    "",
    tool_calls=[ToolCall(id="c1", name="get_weather", arguments='{"city": "Seoul"}')],
)
mem.add_tool_message("sunny, 23C", tool_call_id="c1")
```

tool call과 그에 연결된 결과들은 하나의 원자적 유닛으로 정리(pruning)되므로,
버퍼가 고아 tool 결과로 시작하는 일이 없습니다
([동작 원리](how-it-works.md#tool-call-units) 참고).

`summarize_fn`이 없으면 rollmem은 **버퍼 전용**으로 동작합니다: 버퍼가
`max_tokens`를 초과하면 가장 오래된 턴이 그냥 버려집니다. 요지를 보존하려면
요약 함수를 주입하세요.

## 요약 함수 연결하기

`summarize_fn`은 현재 요약과 버퍼에서 밀려나는 메시지들을 받아 새 요약을
반환합니다. 여기가 LLM을 호출하는 지점입니다:

```python
from anthropic import Anthropic
from rollmem import RollingMemory

client = Anthropic()

def summarize(existing_summary, messages):
    folded = "\n".join(str(m) for m in messages)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Update this running conversation summary with the new turns.\n"
                "Keep it under 200 words; compress, do not just append.\n\n"
                f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                f"New turns:\n{folded}"
            ),
        }],
    )
    return response.content[0].text

mem = RollingMemory(max_tokens=2000, summarize_fn=summarize)
```

!!! tip "요약 함수가 압축하도록 프롬프트하세요"
    rollmem은 `summarize_fn`이 반환하는 값을 제한 없이 그대로 저장합니다.
    단순히 덧붙이기만 하는 요약 함수는 요약을 무한정 키웁니다 — 압축하도록
    지시하거나, 콜백 안에서 길이를 제한하세요.
    [동작 원리](how-it-works.md#limitations)를 참고하세요.

## 비동기 앱에서 사용하기

asyncio 코드에서는 `AsyncRollingMemory`를 사용하세요. 동작은 동일하지만 —
같은 퇴출 유닛, 같은 직렬화 포맷 — `add_*` 메서드가 코루틴이고
`summarize_fn`으로 코루틴 함수를 받을 수 있습니다:

```python
from anthropic import AsyncAnthropic
from rollmem import AsyncRollingMemory

client = AsyncAnthropic()

async def summarize(existing_summary, messages):
    folded = "\n".join(str(m) for m in messages)
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Update this running conversation summary with the new turns.\n"
                "Keep it under 200 words; compress, do not just append.\n\n"
                f"Current summary:\n{existing_summary or '(empty)'}\n\n"
                f"New turns:\n{folded}"
            ),
        }],
    )
    return response.content[0].text

mem = AsyncRollingMemory(max_tokens=2000, summarize_fn=summarize)

await mem.add_user_message("Hi, I'm planning a trip to Korea.")
```

일반(동기) `summarize_fn`도 받으므로, 콜백을 다시 쓰지 않고도 클래스를 바꿀
수 있습니다. 반대 방향은 막혀 있습니다: 동기 `RollingMemory`에 코루틴 함수를
넘기면 코루틴이 요약으로 조용히 저장되는 대신 `TypeError`가 발생합니다.
동시성 보장은 [동작 원리](how-it-works.md#sync-and-async)를 참고하세요.

## 토큰을 정확하게 세기 {#counting-tokens-accurately}

기본 토큰 카운터는 대략적인 단어 수 계산입니다 — 데모에는 충분하지만, 실제
토큰 예산에는 모델에 맞는 정확한 카운터를 주입하세요:

```python
import tiktoken
from rollmem import RollingMemory

enc = tiktoken.get_encoding("cl100k_base")

mem = RollingMemory(
    max_tokens=2000,
    token_counter=lambda text: len(enc.encode(text)),
)
```

`token_counter`는 메시지 한 건의 텍스트(`str`)를 받아 `int`를 반환합니다.
전달되는 텍스트는 `Message.token_text()`로, tool call의 이름과 인자를
포함하므로 tool 페이로드도 예산에 반영됩니다.

## 채팅 루프에서 사용하기

`get_messages()`는 누적 요약을 `system` 턴으로 앞에 붙인 `list[Message]`를
반환합니다 — 프로바이더의 메시지 형식으로 바로 변환할 수 있습니다:

```python
history = [
    {"role": m.role, "content": m.content}
    for m in mem.get_messages()
]
```

`get_context()`는 같은 내용의 문자열 형태이므로 두 결과는 절대 어긋나지
않습니다.

에이전트형 대화라면 tool 필드(`m.tool_calls`, `m.tool_call_id`)도 프로바이더
스키마에 맞게 매핑하세요 — 위 스니펫은 일반 텍스트 턴만 다룹니다.
