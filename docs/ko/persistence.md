# 영속화

`RollingMemory`는 메모리에 존재하지만, 상태를 평범한 `dict`로 직렬화해 두었다가
나중에 복원할 수 있습니다 — 프로세스, 세션, 머신을 넘나들 수 있습니다. 저장
형식과 위치(JSON 파일, 데이터베이스 컬럼, 캐시 등)는 사용자가 선택합니다.

## 저장

`to_dict()`는 대화 **상태**, 즉 누적 요약과 원문 버퍼를 담습니다.

```python
import json

raw = json.dumps(mem.to_dict())
# `raw`를 어디든 저장하세요: 파일, DB 컬럼, Redis, ...
```

결과 매핑은 세 개의 키를 가집니다:

```json
{
  "version": 2,
  "summary": "The user is planning a trip to Korea...",
  "buffer": [
    {"role": "user", "content": "What about the weather in Busan?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "c1", "name": "get_weather", "arguments": "{\"city\": \"Busan\"}"}
      ]
    },
    {"role": "tool", "content": "sunny, 23C", "tool_call_id": "c1"}
  ]
}
```

메시지의 선택적 필드(`id`, `tool_calls`, `tool_call_id`, `metadata`)는 비어
있으면 생략됩니다. 따라서 role/content만 있는 일반 메시지는 이 필드들이 생기기
전과 똑같이 간결한 형태로 직렬화됩니다.

`version`은 향후 호환성을 위한 스키마 버전입니다 — `from_dict`는 지원하지
않는 버전의 데이터를 잘못 해석하는 대신 거부합니다. 쓰기는 항상 현재
버전(2)으로 이루어지며, rollmem 0.0.x가 저장한 버전 1 데이터도 그대로 읽을 수
있습니다.

## 복원

```python
mem = RollingMemory.from_dict(
    json.loads(raw),
    max_tokens=2000,
    summarize_fn=summarize,   # 콜백은 직렬화되지 않습니다 — 다시 주입하세요
    # token_counter=...
)
```

`AsyncRollingMemory`도 같은 `to_dict()` / `from_dict()`와 같은 포맷을
가지므로, 한 클래스가 저장한 상태를 다른 클래스에서 불러올 수 있습니다 —
예를 들어 동기로 기록한 대화를 asyncio 앱에서 이어갈 수 있습니다.

## 저장되는 것과 저장되지 않는 것

| | `to_dict()`로 저장 | `from_dict()`에 다시 전달 |
|---|---|---|
| 누적 요약 | ✅ | |
| 메시지 버퍼 | ✅ | |
| `max_tokens` | | ✅ |
| `summarize_fn` | | ✅ |
| `token_counter` | | ✅ |

토큰 예산과 콜백은 **상태가 아니라 런타임 설정**입니다. 호출 가능 객체는
안전하게 직렬화할 수 없고, 예산은 실행마다 달라질 수 있습니다 — 그래서 둘 다
복원 시점에 다시 전달합니다.

## 복원 시맨틱

`from_dict`는 버퍼를 **원문 그대로** 복원하며 정리(pruning) 로직을 실행하지
않습니다. 이는 다음을 의미합니다:

- 로드 과정에서 예기치 않은 `summarize_fn` 호출이 일어나거나 턴이 버려지는
  일이 절대 없습니다.
- 저장할 때보다 작은 `max_tokens`로 복원하면 버퍼가 일시적으로 예산을 초과할
  수 있습니다 — 예산은 다음 `add_message` 시점에 다시 적용됩니다.
- 로드 시점에 버퍼를 재정렬하지 않습니다: assistant 턴이나 고아 tool 결과로
  시작하는 저장본은 그대로 복원되며, 다음 퇴출이 일어날 때까지 그 상태로
  유지됩니다.

## 예외

`data`의 직렬화 버전이 지원되지 않으면 `from_dict`는 `ValueError`를
발생시킵니다.
