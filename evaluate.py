"""RAG 검색과 답변 품질을 간단히 검증하는 실행용 스크립트입니다."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app import DATA_DIR, build_answer_chain, build_retriever, format_answer, make_context


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_CASES_FILE = PROJECT_ROOT / "eval_questions.json"
REPORT_FILE = PROJECT_ROOT / "eval_report.json"


def normalize(text: str) -> str:
    """띄어쓰기 차이 때문에 키워드 검사가 실패하지 않도록 정규화합니다."""
    return re.sub(r"\s+", "", text).lower()


def evaluate_case(case: dict[str, Any], retriever: Any, answer_chain: Any) -> dict[str, Any]:
    """한 질문의 검색 출처, 답변 키워드, 거절 여부를 검사합니다."""
    question = case["question"]
    documents = retriever.invoke(question)
    sources = sorted({str(document.metadata.get("source", "")) for document in documents})
    context = make_context(documents)
    answer = format_answer(answer_chain.invoke({"context": context, "question": question}))

    expected_sources = set(case.get("expected_sources", []))
    source_pass = expected_sources.issubset(set(sources))
    normalized_answer = normalize(answer)
    answer_keywords = [str(keyword) for keyword in case.get("answer_keywords", [])]
    keyword_pass = all(normalize(keyword) in normalized_answer for keyword in answer_keywords)

    if case.get("must_refuse"):
        refusal_pass = normalize("문서에서 확인할 수 없습니다") in normalized_answer
    else:
        refusal_pass = True

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "retrieved_sources": sources,
        "source_pass": source_pass,
        "keyword_pass": keyword_pass,
        "refusal_pass": refusal_pass,
        "passed": source_pass and keyword_pass and refusal_pass,
    }


def main() -> None:
    """테스트셋을 실행하고 사람이 확인할 수 있는 JSON 보고서를 저장합니다."""
    load_dotenv(PROJECT_ROOT / ".env")
    cases = json.loads(TEST_CASES_FILE.read_text(encoding="utf-8"))

    print(f"문서 폴더: {DATA_DIR}")
    print(f"테스트 케이스: {len(cases)}개")
    print("검색 인덱스와 답변 모델을 준비하는 중...")
    retriever = build_retriever()
    answer_chain = build_answer_chain()

    results = []
    for case in cases:
        print(f"검증 중: {case['id']}")
        result = evaluate_case(case, retriever, answer_chain)
        results.append(result)
        print("  PASS" if result["passed"] else "  FAIL")

    passed = sum(result["passed"] for result in results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0,
        "results": results,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과: {passed}/{len(results)} 통과")
    print(f"상세 보고서: {REPORT_FILE}")

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
