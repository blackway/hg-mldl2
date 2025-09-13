# 대규모 언어 모델로 텍스트 생성하기: 발표 자료

## 1. 주제 소개
- 본 발표는 대규모 언어 모델(LLM, Large Language Model)을 활용한 텍스트 생성 실습을 다룹니다.
- EXAONE-3.5와 GPT-4o, Gemini API 등 다양한 최신 언어 모델을 활용하여 실제 상품 Q&A 챗봇을 구현하는 방법을 소개합니다.

---

## 2. 실습 환경 및 준비
- Python, Jupyter Notebook 환경에서 실습 진행
- 주요 라이브러리: `transformers`, `openai`, `requests`, `numpy`, `scipy`
- 모델 및 API:
  - EXAONE-3.5 (HuggingFace Transformers)
  - GPT-4o (OpenAI API)
  - Gemini (Google Generative Language API)

---

## 3. EXAONE-3.5로 상품 질문 답변 생성
### 3.1 모델 및 토크나이저 준비
```python
from transformers import AutoTokenizer, pipeline
exaone_tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
pipe = pipeline(task="text-generation", model="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", tokenizer=exaone_tokenizer, device=0, trust_remote_code=True)
```

### 3.2 프롬프트 예시
- 시스템 역할: 쇼핑몰 Q&A 챗봇
- 답변 예시: 확정적 답변 대신 담당자가 확인 후 안내한다는 친절한 답변 생성
```python
messages = [
    {"role": "system", "content": "너는 쇼핑몰 홈페이지에 올라온 질문에 대답하는 Q&A 챗봇이야. 확정적인 답변을 하지 말고 제품 담당자가 정확한 답변을 하기 위해 시간이 필요하다는 간단하고 친절한 답변을 생성해줘."},
    {"role": "user", "content": "이 다이어리에 내년도 공휴일이 표시되어 있나요?"}
]
output = pipe(messages, max_new_tokens=200)
```

---

## 4. 토큰 디코딩 전략
### 4.1 기본 샘플링
- 소프트맥스(softmax)로 확률 분포 계산 후 샘플링
```python
import numpy as np
from scipy.special import softmax
logits = np.array([1, 2, 3, 4, 100])
probas = softmax(logits)
np.random.multinomial(100, probas)
```
- temperature 조절로 확률 분포의 날카로움 조정

### 4.2 top-k, top-p 샘플링
- top-k: 확률이 높은 k개 토큰만 샘플링
- top-p: 누적 확률이 p 이하인 토큰만 샘플링
```python
output = pipe(messages, max_new_tokens=200, do_sample=True, top_k=10)
output = pipe(messages, max_new_tokens=200, do_sample=True, top_p=0.9)
```

---

## 5. GPT-4o, Gemini API로 텍스트 생성
### 5.1 GPT-4o (OpenAI)
- OpenAI API를 활용한 챗봇 응답 생성
- 주요 파라미터: model, messages, temperature, top_p 등
```python
from openai import OpenAI
client = OpenAI(api_key="OpenAI 키를 입력하세요")
completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(completion.choices[0].message.content)
```

### 5.2 Gemini API (Google)
- Gemini API를 활용한 챗봇 응답 생성
- REST API 방식, API 키 필요
```python
import requests
GEMINI_API_KEY = "여기에 Gemini API 키를 입력하세요"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
def gemini_chat(messages):
    prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, json=data)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"
# gemini_response = gemini_chat(messages)
# print(gemini_response)
```

---

## 6. 실습 시 유의사항
- 각 API의 키와 사용량 제한, 요금 정책을 반드시 확인
- 모델별로 입력 포맷, 파라미터, 출력 형식이 다르므로 공식 문서 참고
- GPU 환경에서 실행 권장 (특히 EXAONE-3.5)

---

## 7. 결론 및 Q&A
- 다양한 LLM을 활용한 텍스트 생성 실습을 통해 실제 서비스에 적용 가능한 챗봇, Q&A 시스템 구현 방법을 익힘
- 추가 질문 및 토론 환영
