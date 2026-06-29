import os
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class MeasurementData(BaseModel):
    key: str
    name_ja: str
    value: Optional[float] = None
    unit: str
    confidence: str


class SessionData(BaseModel):
    created_at: str
    height_cm: float
    weight_kg: Optional[float] = None
    gender: str
    has_side_view: bool
    measurements: List[MeasurementData]


class AnalysisRequest(BaseModel):
    sessions: List[SessionData]


@router.post("/insights")
async def get_insights(request: AnalysisRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY が設定されていません", "analysis": None}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        sessions_text = ""
        for i, s in enumerate(request.sessions, 1):
            sessions_text += f"\n【採寸記録 {i}】({s.created_at[:10]})\n"
            sessions_text += f"身長: {s.height_cm}cm"
            if s.weight_kg:
                sessions_text += f"  体重: {s.weight_kg}kg"
            gender_ja = "女性" if s.gender == "female" else "男性" if s.gender == "male" else "未回答"
            sessions_text += f"  性別: {gender_ja}\n"
            for m in s.measurements:
                if m.value is not None:
                    sessions_text += f"  {m.name_ja}: {m.value}{m.unit} ({m.confidence})\n"

        prompt = f"""あなたは体型・ファッション・健康の専門家アシスタントです。
以下の採寸データを基に、日本語で詳しいアドバイスをしてください。

{sessions_text}

以下の観点から分析してください：
1. **体型の特徴**: 現在の体型の特徴や比率について
2. **サイズ・ファッションアドバイス**: 服のサイズ選びや似合うシルエットについて
3. **健康・フィットネスのポイント**: BMIや体型バランスから見えること（複数回データがある場合は変化の傾向も）
4. **まとめ**: 総合的なコメント

親しみやすく、具体的に答えてください。"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        return {"analysis": message.content[0].text}

    except Exception as e:
        return {"error": str(e), "analysis": None}
