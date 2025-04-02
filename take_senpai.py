import streamlit as st
from openai import OpenAI
from datetime import datetime
import pandas as pd
import gspread
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from google.oauth2.service_account import Credentials

# ========== 設定 ==========
SHEET_NAME = "takesenpai_logs"
WORKSHEET_NAME = "logs"
JSON_KEY = "takesenpai-logger-7835567fe6ce.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

client = OpenAI(api_key="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")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ========== Google Sheets認証 ==========
creds = Credentials.from_service_account_file(JSON_KEY, scopes=SCOPES)
gs_client = gspread.authorize(creds)
sheet = gs_client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)

# ========== Streamlit UI ==========
st.set_page_config(page_title="たけ先輩 - 記憶&プロファイル版", page_icon="🧠")
st.title("たけ先輩 🧑‍🏫（記憶を持ち、覚えるAI）")
st.markdown("社員番号を入力して話しかけてください。たけ先輩は会話からあなたの情報を覚えていきます。")

user_id = st.text_input("👤 社員番号を入力：")
user_input = st.text_area("📋 今日の予定・悩み・学習内容など：", height=150)
submit = st.button("たけ先輩に送る")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if submit and user_input.strip() != "" and user_id.strip() != "":
    with st.spinner("たけ先輩が記憶とプロフィールを確認中..."):

        # ========== 会話ログ取得 ==========
        logs_df = pd.DataFrame(sheet.get_all_records())
        user_logs = logs_df[logs_df["社員番号"].astype(str) == user_id]

        # ========== プロファイル取得 ==========
        try:
            profile_sheet = gs_client.open(SHEET_NAME).worksheet("Numbers")
            profile_df = pd.DataFrame(profile_sheet.get_all_records())
            user_profile = profile_df[profile_df["社員番号"].astype(str) == user_id]
            profile_text = ""
            if not user_profile.empty:
                row = user_profile.iloc[0]
                profile_text = f"""
                ユーザーの基本情報：
                ・名前：{row.get('名前', '不明')}
                ・所属：{row.get('所属', '不明')}
                ・年齢：{row.get('年齢', '不明')}
                ・住まい：{row.get('住まい', '不明')}
                """
        except:
            profile_text = ""

        # ========== セッション履歴 ==========
        history_text = "\n".join([
            f"あなた：{u}\nたけ先輩：{r}"
            for u, r in st.session_state.chat_history[-5:]
        ])
        # ========== RAG検索（直近5件も含める） ==========
        if not user_logs.empty:
            texts = [
                f"ユーザー：{row['ユーザー入力']} たけ先輩：{row['たけ先輩の返答']}"
                for _, row in user_logs.iterrows()
            ]
            embeddings = model.encode(texts)
            dim = embeddings.shape[1]
            local_index = faiss.IndexFlatL2(dim)
            local_index.add(np.array(embeddings))
            query_vec = model.encode([user_input])
            D, I = local_index.search(np.array(query_vec), 3)

            semantic_recall = "\n".join([texts[i] for i in I[0]])
            recent_recall = "\n".join([
                f"ユーザー：{row['ユーザー入力']} たけ先輩：{row['たけ先輩の返答']}"
                for _, row in user_logs.tail(5).iterrows()
            ])
            recalled = recent_recall + "\n" + semantic_recall
        else:
            recalled = ""

        # ========== GPT返答プロンプト ==========
        prompt = f"""
あんたはIT訓練担当のタケちゃんマンや。  
・喋り方は猛虎弁、タメ口、兄貴っぽいノリでな。
・技術的な質問（例：〇〇の使い方、学習方法、設計）には、しっかり詳しく長めに説明してください。
・雑談・気持ち・モチベーション系の話題には、軽いツッコミや冗談で返してください。
・語尾はフレンドリーに、距離感は近く、気楽に話すようにしてください。
・ただし、毎回「頼ってくれ」や「一緒に頑張ろう」などの定型文で終わる必要はありません。技術の説明やツッコミで自然に終わるようにしてください。感動系の締めはたまにでいいです。

【このユーザーの基本情報】
{profile_text}

【最近の会話履歴】
{history_text}

【意味的に近い過去の会話】
{recalled}

【今回の発言】
{user_input}

この流れに合う返答を短めで頼むわ。頼りにしとるで。
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content.strip()

        # ========== Sheetsに保存 ==========
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, user_id, user_input, reply])
        st.session_state.chat_history.append((user_input, reply))
        st.success("🧠 たけ先輩の返答が届きました！（記憶にも保存済み）")

        # ========== プロファイル抽出用プロンプト ==========
        profile_prompt = f"""
以下のユーザー発言から、覚えておくべきプロフィール情報を抽出してください。

発言：{user_input}

抽出例：
好きな技術: React
苦手な分野: CSS
話し方タイプ: 雑談タイプ
"""
        profile_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": profile_prompt}]
        )
        profile_text_raw = profile_response.choices[0].message.content.strip()

        # ========== プロファイル更新処理 ==========
        lines = profile_text_raw.splitlines()
        profile_updates = {}
        for line in lines:
            match = re.match(r"(.+?)\s*[:：]\s*(.+)", line)
            if match:
                key = match.group(1).strip()
                val = match.group(2).strip()
                if key in ["好きな言語"]:
                    key = "好きな技術"
                profile_updates[key] = val

        try:
            if not user_profile.empty:
                row_idx = user_profile.index[0] + 2
                for key, val in profile_updates.items():
                    if key not in profile_df.columns:
                        profile_sheet.add_cols(1)
                        profile_sheet.update_cell(1, len(profile_df.columns) + 1, key)
                        col_idx = len(profile_df.columns) + 1
                    else:
                        col_idx = profile_df.columns.get_loc(key) + 1
                    profile_sheet.update_cell(row_idx, col_idx, val)
        except Exception as e:
            st.warning(f"プロフィール更新中にエラー：{e}")

# ========== 会話履歴表示（チャットUI・自然順） ==========
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 たけ先輩との会話")
    for u, r in st.session_state.chat_history:
        st.markdown(f'''
<!-- あなた -->
<div style="display: flex; justify-content: flex-end; margin-bottom: 5px;">
  <div style="background-color: #DCF8C6; padding: 10px 12px; border-radius: 15px; max-width: 70%;">
    {u}
  </div>
</div>

<!-- たけ先輩 -->
<div style="display: flex; justify-content: flex-start; margin-bottom: 30px;">
  <div style="background-color: #F1F0F0; padding: 10px 12px; border-radius: 15px; max-width: 70%;">
    たけ先輩：{r}
  </div>
</div>
''', unsafe_allow_html=True)
