# rollmem

LLM 애플리케이션을 위한 독립형, **의존성 없는** 롤링 대화 메모리 —
누적 요약(running summary)과 최근 메시지 버퍼의 조합입니다.

LangChain의 `ConversationSummaryBufferMemory`에서 영감을 받았지만, LangChain을
비롯한 어떤 외부 의존성도 없습니다. 요약-버퍼 패턴만 필요할 때 **대화 메모리**,
**컨텍스트 압축**, 긴 대화 처리에 유용합니다.

## 왜 rollmem인가?

`ConversationSummaryBufferMemory`는 훌륭한 패턴입니다: 최근 턴은 원문 그대로
유지하고, 오래된 턴은 누적 요약으로 접어 넣어 컨텍스트 크기를 제한합니다.
하지만 그것 하나를 위해 LangChain 전체를 설치하는 것은 부담스럽습니다.

rollmem은 이 아이디어를 작고 프로바이더 중립적인 패키지로 추출했습니다:

- **런타임 의존성 제로.** Python 표준 라이브러리만 사용합니다.
- **프로바이더 중립.** 요약 방법(`summarize_fn`)과 토큰 계산 방법
  (`token_counter`)을 호출자가 주입합니다 — rollmem 자체는 중립을 유지하므로
  어떤 LLM 프로바이더와도, 심지어 LLM 없이도 동작합니다.
- **완전한 타입 지원.** `py.typed`를 포함하므로 타입 체커가 모든 어노테이션을
  인식합니다.
- **최소한의 표면적.** 메모리 클래스 하나, 메시지 타입 하나, 콜백 프로토콜
  둘. 설정 항목이 난립하지 않습니다.

## 한눈에 보기

```python
from rollmem import RollingMemory

def summarize(existing_summary, messages):
    # 여기에 어떤 LLM이든 연결하세요; 새 요약 문자열을 반환하면 됩니다
    folded = " ".join(m.content for m in messages)
    return (existing_summary + " " + folded).strip()

mem = RollingMemory(max_tokens=2000, summarize_fn=summarize)

mem.add_user_message("Hi, I'm planning a trip to Korea.")
mem.add_assistant_message("Great! When are you going?")

print(mem.get_context())    # 요약(system 턴) + 최근 버퍼
```

새 턴은 원문 버퍼에 쌓입니다. 버퍼가 `max_tokens`를 초과하면 가장 오래된
턴들이 누적 요약으로 접혀 들어갑니다 — 전체 대화의 요지는 보존하면서 프롬프트
컨텍스트 크기는 제한됩니다.

## 다음 단계

- [시작하기](getting-started.md) — 설치하고 첫 메모리를 만들어 봅니다.
- [동작 원리](how-it-works.md) — 버퍼, 요약, 정리(pruning) 알고리즘과 그
  한계.
- [영속화](persistence.md) — 메모리 상태 저장과 복원.
- [API 레퍼런스](api.md) — 소스에서 생성된 전체 공개 API.

## 라이선스

MIT
