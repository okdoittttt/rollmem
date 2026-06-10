# API 레퍼런스

rollmem의 전체 공개 API입니다. 여기에 문서화된 모든 것은 최상위 `rollmem`
패키지에서 임포트할 수 있으며 `__all__`에 등록되어 있습니다.

!!! note
    아래 클래스 레퍼런스는 소스 코드의 docstring에서 자동 생성되므로 영문으로
    표시됩니다.

## RollingMemory

::: rollmem.RollingMemory
    options:
      inherited_members: true

## AsyncRollingMemory

::: rollmem.AsyncRollingMemory
    options:
      inherited_members: true

## Message

::: rollmem.Message

## ToolCall

::: rollmem.ToolCall

## 콜백 프로토콜

rollmem은 LLM 프로바이더에 묶일 수 있는 두 가지 동작을 호출자가 주입하게
함으로써 프로바이더 중립을 유지합니다:

```python
SummarizeFn = Callable[[str, Sequence[Message]], str]
AsyncSummarizeFn = Callable[[str, Sequence[Message]], Awaitable[str]]
TokenCounter = Callable[[str], int]
```

- **`SummarizeFn`** — `summarize_fn(existing_summary, messages_to_fold)`이 새
  요약을 반환합니다. 정리(pruning) 한 번당, 퇴출되는 모든 메시지를 모아
  정확히 한 번 호출됩니다 — 경계 정렬은 퇴출 묶음을 키울 수는 있어도 호출
  횟수를 늘리지는 않습니다.
- **`AsyncSummarizeFn`** — 같은 콜백의 코루틴 형태로, `AsyncRollingMemory`가
  받습니다(일반 `SummarizeFn`도 받습니다). 동기 `RollingMemory`에 코루틴
  함수를 넘기면 `TypeError`가 발생합니다.
- **`TokenCounter`** — `token_counter(text)`가 메시지 한 건의 텍스트에 대한
  토큰 수를 반환합니다. 기본값은 단어 수 기반 추정입니다. 전달되는 텍스트는
  `Message.token_text()` — content에 tool call의 이름·인자·연결 정보를 더한
  것 — 이므로 tool 페이로드도 예산에 반영됩니다.

## Role 상수

`add_*_message` 헬퍼가 사용하는 관례적인 role 문자열입니다. rollmem이 이를
강제하지는 않습니다 — `add_message`는 어떤 role 문자열이든 허용합니다.

| 상수 | 값 |
|---|---|
| `USER` | `"user"` |
| `ASSISTANT` | `"assistant"` |
| `SYSTEM` | `"system"` |
| `TOOL` | `"tool"` |
