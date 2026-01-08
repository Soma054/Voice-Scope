import streamlit as st
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO
import plotly.graph_objects as go

# ページ設定
st.set_page_config(
    page_title="Voice-Scope",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# セッション状態の初期化
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'calibration_done' not in st.session_state:
    st.session_state.calibration_done = False
if 'selected_insight' not in st.session_state:
    st.session_state.selected_insight = 0

# ニューモーフィズムCSS
st.markdown("""
<style>
    :root {
        --bg: #F4F3FF;
        --surface: #F8F7FF;
        --surface2: #F1EFFF;
        --text: #2B2A33;
        --muted: #6B6A77;
        --primary: #7C3AED;
        --primary-light: #A78BFA;
        --accent: #EC4899;
        --shadow-light: #ffffff;
        --shadow-dark: rgba(32,24,72,.12);
    }
    
    /* 背景 */
    .stApp {
        background: radial-gradient(1200px 800px at 20% 10%, #fff 0%, var(--bg) 55%, #EEF0FF 100%);
    }
    
    /* ニューモーフィズムカード */
    .neu-card {
        background: var(--surface);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 10px 10px 24px var(--shadow-dark), -10px -10px 24px var(--shadow-light);
        border: 1px solid rgba(124, 58, 237, 0.06);
        margin: 20px 0;
    }
    
    /* タイトルスタイル */
    .main-title {
        font-size: 3em;
        font-weight: bold;
        color: var(--primary);
        text-align: center;
        margin: 40px 0 20px 0;
        text-shadow: 2px 2px 4px rgba(124, 58, 237, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: var(--muted);
        font-size: 1.1em;
        margin-bottom: 40px;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        background: var(--surface);
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        color: var(--primary);
        font-weight: bold;
        box-shadow: 5px 5px 10px var(--shadow-dark), -5px -5px 10px var(--shadow-light);
        transition: all 0.2s ease;
        width: 100%;
        font-size: 1.1em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 7px 7px 14px var(--shadow-dark), -7px -7px 14px var(--shadow-light);
    }
    
    .stButton > button:active {
        box-shadow: inset 5px 5px 10px var(--shadow-dark), inset -5px -5px 10px var(--shadow-light);
        transform: translateY(1px);
    }
    
    /* メトリクスカード */
    .metric-card {
        background: var(--surface2);
        border-radius: 14px;
        padding: 15px;
        border: 1px solid rgba(124, 58, 237, 0.1);
        box-shadow: inset 6px 6px 12px rgba(32,24,72,.08), inset -6px -6px 12px rgba(255,255,255,.85);
        margin: 10px 0;
    }
    
    /* バッジ */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 12px;
        font-size: 0.9em;
        font-weight: bold;
        background: #DCFCE7;
        color: #166534;
    }
    
    .badge-warning {
        background: #FEF3C7;
        color: #92400E;
    }
    
    /* インサイトカード */
    .insight-card {
        background: var(--surface2);
        border-radius: 14px;
        padding: 16px;
        margin: 10px 0;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: inset 6px 6px 12px rgba(32,24,72,.08), inset -6px -6px 12px rgba(255,255,255,.85);
    }
    
    .insight-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
    }
    
    .insight-card.active {
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
    }
    
    /* ステップインジケーター */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 30px 0;
        font-size: 1.1em;
    }
    
    .step {
        color: var(--muted);
    }
    
    .step.active {
        color: var(--primary);
        font-weight: bold;
    }
    
    .step.done {
        color: #10b981;
    }
    
    /* 警告ボックス */
    .warning-box {
        background: #FDF4FF;
        border: 1px solid #FBCFE8;
        color: #831843;
        padding: 15px;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    /* グリッドレイアウト */
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    
    /* ヘッダー */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* カスタムメトリクス */
    .custom-metric {
        text-align: center;
        padding: 20px;
        background: var(--surface);
        border-radius: 15px;
        box-shadow: 5px 5px 10px var(--shadow-dark), -5px -5px 10px var(--shadow-light);
    }
    
    .custom-metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: var(--primary);
        font-family: monospace;
    }
    
    .custom-metric-label {
        font-size: 0.9em;
        color: var(--muted);
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ユーティリティ関数
def change_page(page):
    st.session_state.page = page
    st.rerun()

def calculate_metrics(y, sr):
    """音声からメトリクスを計算（ダミーデータ）"""
    # 実際はここでlibrosaを使って計算
    metrics = {
        'stability': np.random.randint(60, 90),
        'clarity': np.random.randint(50, 80),
        'resonance': np.random.randint(45, 75),
        'brightness': np.random.randint(40, 80),
        'power': np.random.randint(40, 85)
    }
    
    ideal = {
        'stability': 75,
        'clarity': 72,
        'resonance': 68,
        'brightness': 66,
        'power': 52
    }
    
    return metrics, ideal

def create_radar_chart(metrics, ideal, highlight_idx=None):
    """レーダーチャートを作成"""
    labels = ['安定性', 'クリアさ', '響き・抜け', '明るさ', '音圧']
    
    me_values = [metrics['stability'], metrics['clarity'], metrics['resonance'], 
                 metrics['brightness'], metrics['power']]
    ideal_values = [ideal['stability'], ideal['clarity'], ideal['resonance'], 
                    ideal['brightness'], ideal['power']]
    
    fig = go.Figure()
    
    # 自分のデータ
    fig.add_trace(go.Scatterpolar(
        r=me_values,
        theta=labels,
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.2)',
        line=dict(color='#7C3AED', width=3),
        name='自分 (Source)',
        marker=dict(size=8, color='#7C3AED')
    ))
    
    # 理想のデータ
    fig.add_trace(go.Scatterpolar(
        r=ideal_values,
        theta=labels,
        fill='toself',
        fillcolor='rgba(236, 72, 153, 0.1)',
        line=dict(color='#EC4899', width=3, dash='dash'),
        name='理想 (Target)',
        marker=dict(size=8, color='#EC4899')
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(248, 247, 255, 0.5)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linecolor='rgba(32, 24, 72, 0.1)',
                gridcolor='rgba(32, 24, 72, 0.1)'
            ),
            angularaxis=dict(
                linecolor='rgba(32, 24, 72, 0.1)',
                gridcolor='rgba(32, 24, 72, 0.1)'
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.1,
            xanchor='center',
            orientation='h'
        ),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14, color='#2B2A33')
    )
    
    return fig

# ==================== ページ定義 ====================

def page_home():
    """ホーム画面"""
    st.markdown('<div class="main-title">🎤 Voice-Scope</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">科学的なデータで、理想の声とのギャップを可視化。</div>', unsafe_allow_html=True)
    
    # 中央寄せのカード
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        
        st.markdown("### Voice Lab")
        st.markdown("あなたの声を科学的に分析し、理想の声とのギャップを可視化します。")
        
        st.markdown("---")
        
        # モード選択
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🎤 比較分析\n\n*Calibration First*", key="calib_mode"):
                change_page('calibration')
        
        with col_b:
            if st.button("⚡️ Quick診断\n\n*診断のみ*", key="quick_mode"):
                change_page('input')
        
        st.markdown("---")
        
        if st.button("📈 成長記録 (Import)", key="import"):
            st.info("この機能は今後実装予定です")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 注意書き
        st.markdown("""
        <div style='text-align:center; margin-top:30px; color:#999; font-size:0.9em;'>
        ⚠️ データはサーバーに保存されません<br>
        医療機器ではありません
        </div>
        """, unsafe_allow_html=True)

def page_calibration():
    """校正テスト画面"""
    # ステップインジケーター
    st.markdown("""
    <div class="step-indicator">
        <span class="step active">1. 校正</span>
        <span>›</span>
        <span class="step">2. 本番</span>
        <span>›</span>
        <span class="step">3. 解析</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        
        st.markdown("### 🔭 校正テスト (推奨)")
        st.markdown("""
        あなたの「本来の声質」を測定し、比較精度を高めます。
        
        **「あー」と3秒間発声してください。**
        """)
        
        # マイクビジュアライザー（プレースホルダー）
        st.markdown("""
        <div style='text-align:center; padding:30px; background:rgba(124,58,237,0.05); border-radius:15px; margin:20px 0;'>
        <div style='font-size:3em;'>🎤</div>
        <div style='color:#7C3AED; margin-top:10px;'>録音準備完了</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ファイルアップロード
        uploaded_calib = st.file_uploader(
            "または校正用音声ファイルをアップロード",
            type=["wav", "mp3"],
            key="calib_upload"
        )
        
        if uploaded_calib:
            st.success("✅ 校正音声を読み込みました")
            st.session_state.calibration_done = True
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔴 REC (Calibration)", key="rec_calib", type="primary"):
                if uploaded_calib:
                    st.session_state.calibration_done = True
                    change_page('input')
                else:
                    st.warning("音声ファイルをアップロードしてください")
        
        with col_b:
            if st.button("スキップする", key="skip_calib"):
                change_page('input')
        
        st.markdown('</div>', unsafe_allow_html=True)

def page_input():
    """録音・アップロード画面"""
    # ステップインジケーター
    if st.session_state.calibration_done:
        st.markdown("""
        <div class="step-indicator">
            <span class="step done">✔ 校正</span>
            <span>›</span>
            <span class="step active">2. 本番</span>
            <span>›</span>
            <span class="step">3. 解析</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="step-indicator">
            <span class="step active">2. 本番</span>
            <span>›</span>
            <span class="step">3. 解析</span>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 💿 理想の声 (Target)")
        
        target_file = st.file_uploader(
            "ファイルを選択",
            type=["mp3", "wav"],
            key="target_upload",
            help="伴奏付きOK"
        )
        
        if target_file:
            st.success(f"✅ {target_file.name}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 🎤 自分の声 (Source)")
        st.markdown("*※マイク距離15cm推奨 / 環境音OFF*")
        
        source_file = st.file_uploader(
            "ファイルを選択",
            type=["mp3", "wav"],
            key="source_upload"
        )
        
        if source_file:
            st.success(f"✅ {source_file.name}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 解析開始ボタン
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔬 解析開始", key="start_analysis", type="primary"):
            if source_file:
                # 音声データを読み込み
                audio_bytes = BytesIO(source_file.read())
                y, sr = librosa.load(audio_bytes, sr=None)
                st.session_state.audio_data = y
                st.session_state.sr = sr
                change_page('loading')
            else:
                st.warning("自分の声をアップロードしてください")

def page_loading():
    """解析中画面"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="neu-card" style="text-align:center; padding:60px;">', unsafe_allow_html=True)
        
        with st.spinner(''):
            st.markdown("### 🔬 解析中...")
            st.markdown("Demucs AI分離 / Mid-Side処理実行中")
            
            # プログレスバー
            import time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = [
                "ボーカル抽出中...",
                "Mid-Side処理でハモリ除去中...",
                "5つの指標を計算中...",
                "レポート生成中..."
            ]
            
            for i, step in enumerate(steps):
                status_text.markdown(f"**{step}**")
                time.sleep(1)
                progress_bar.progress((i + 1) * 25)
            
            time.sleep(0.5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        change_page('result1')

def page_result1():
    """解析結果画面1: レーダーチャートと要点"""
    st.markdown('<div class="main-title">📊 解析レポート</div>', unsafe_allow_html=True)
    
    # メトリクス計算
    if st.session_state.audio_data is not None:
        metrics, ideal = calculate_metrics(st.session_state.audio_data, st.session_state.sr)
    else:
        # ダミーデータ
        metrics = {'stability': 62, 'clarity': 55, 'resonance': 48, 'brightness': 70, 'power': 40}
        ideal = {'stability': 75, 'clarity': 72, 'resonance': 68, 'brightness': 66, 'power': 52}
    
    # 信頼度バッジ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<span class="badge">信頼度: 高 (A)</span>', unsafe_allow_html=True)
    
    # 2カラムレイアウト
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 比較レーダー + ハイライト")
        st.markdown("右の「要点」をクリックすると、該当軸を強調します")
        
        # レーダーチャート
        fig = create_radar_chart(metrics, ideal, st.session_state.selected_insight)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("自分=<span class='badge' style='background:rgba(124,58,237,0.2);color:#4c1d95'>紫 実線</span> / 理想=<span class='badge' style='background:rgba(236,72,153,0.15);color:#9d174d'>ピンク 破線</span>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("### 要点（クリック）")
        st.markdown("5指標から見える強み/弱み/バランス")
        
        # インサイト生成
        me_values = [metrics['stability'], metrics['clarity'], metrics['resonance'], 
                     metrics['brightness'], metrics['power']]
        labels = ['安定性', 'クリアさ', '響き・抜け', '明るさ', '音圧']
        
        max_idx = np.argmax(me_values)
        min_idx = np.argmin(me_values)
        balance = 100 - np.std(me_values) * 2
        
        insights = [
            {"title": f"強み: {labels[max_idx]}", "body": "自分スコアが最も高い軸。まずはここを維持しつつ他を底上げ。", "idx": max_idx},
            {"title": f"改善候補: {labels[min_idx]}", "body": "相対的に低い軸。理想との差が大きいなら優先度高。", "idx": min_idx},
            {"title": "差が大きい項目", "body": "理想との差分が最大の軸を優先表示。", "idx": 2},
            {"title": f"バランス: {balance:.0f}/100", "body": "全体の凸凹の少なさ。凸凹が大きいほど一部だけ突出。", "idx": None}
        ]
        
        for i, insight in enumerate(insights):
            active = "active" if i == st.session_state.selected_insight else ""
            if st.button(
                f"**{insight['title']}**\n\n{insight['body']}", 
                key=f"insight_{i}",
                use_container_width=True
            ):
                st.session_state.selected_insight = i
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI コメント
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("""
    **🤖 AI Lab Comment:**
    
    安定性はプロレベルですが、**「明るさ (Brightness)」**にギャップがあります。
    
    🔍 **検索ヒント**: 「軟口蓋 上げ方」「鼻腔共鳴 トレーニング」
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ナビゲーションボタン
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🏠 トップへ戻る", key="home1"):
            st.session_state.audio_data = None
            st.session_state.calibration_done = False
            change_page('home')
    
    with col3:
        if st.button("詳細メトリクス →", key="next_page", type="primary"):
            change_page('result2')

def page_result2():
    """解析結果画面2: 詳細メトリクスと推奨"""
    st.markdown('<div class="main-title">📈 詳細メトリクス</div>', unsafe_allow_html=True)
    
    # メトリクス計算
    if st.session_state.audio_data is not None:
        metrics, ideal = calculate_metrics(st.session_state.audio_data, st.session_state.sr)
    else:
        metrics = {'stability': 62, 'clarity': 55, 'resonance': 48, 'brightness': 70, 'power': 40}
        ideal = {'stability': 75, 'clarity': 72, 'resonance': 68, 'brightness': 66, 'power': 52}
    
    # 専門/簡略モード切り替え
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        mode = st.toggle("Advanced View (専門モード)", key="advanced_mode")
    
    # メトリクス詳細
    st.markdown('<div class="neu-card">', unsafe_allow_html=True)
    
    metric_details = [
        {
            "simple": "安定性", 
            "pro": "Jitter (Stability)", 
            "value": metrics['stability'], 
            "ideal": ideal['stability'],
            "unit": "%",
            "desc": "ピッチの揺れの少なさ。高いほど安定した発声。",
            "range": "60-80: 一般的 / 80-90: 良好 / 90+: プロレベル"
        },
        {
            "simple": "クリアさ", 
            "pro": "HNR (Clarity)", 
            "value": metrics['clarity'], 
            "ideal": ideal['clarity'],
            "unit": "dB",
            "desc": "ノイズに対する声の明瞭さ。息漏れが少ないほど高い。",
            "range": "50-60: 改善余地 / 60-75: 良好 / 75+: 非常にクリア"
        },
        {
            "simple": "響き・抜け", 
            "pro": "Formant Ratio", 
            "value": metrics['resonance'], 
            "ideal": ideal['resonance'],
            "unit": "Idx",
            "desc": "声の共鳴の豊かさ。フォルマント周波数のバランス。",
            "range": "40-60: 通常 / 60-75: 豊か / 75+: 非常に豊か"
        },
        {
            "simple": "明るさ", 
            "pro": "Spectral Centroid", 
            "value": metrics['brightness'], 
            "ideal": ideal['brightness'],
            "unit": "Hz",
            "desc": "声のトーンの明るさ。高周波成分の量。",
            "range": "30-50: 暗い / 50-70: 通常 / 70+: 明るい"
        },
        {
            "simple": "音圧", 
            "pro": "RMS (Power)", 
            "value": metrics['power'], 
            "ideal": ideal['power'],
            "unit": "dB",
            "desc": "声の力強さ。音量ではなく、エネルギーの安定性。",
            "range": "30-45: 弱い / 45-60: 適切 / 60+: 強い"
        }
    ]
    
    for metric in metric_details:
        diff = metric['value'] - metric['ideal']
        diff_color = "#10b981" if diff >= 0 else "#ef4444"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="flex:1;">
                    <div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">
                        {metric['pro'] if mode else metric['simple']}
                    </div>
                    <div style="font-size:0.85em; color:var(--muted);">
                        Target: {metric['ideal']} {metric['unit'] if mode else ''}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:monospace; font-size:1.8em; font-weight:bold; color:var(--primary);">
                        {metric['value']}
                        <span style="font-size:0.5em; color:var(--muted);">{metric['unit'] if mode else ''}</span>
                    </div>
                    <div style="font-size:0.9em; color:{diff_color}; font-weight:bold;">
                        ({diff:+d})
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if mode:
            with st.expander(f"📖 {metric['pro']} の詳細"):
                st.markdown(f"**説明**: {metric['desc']}")
                st.markdown(f"**参考範囲**: {metric['range']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 推奨トレーニング
    st.markdown('<div class="neu-card" style="margin-top:20px;">', unsafe_allow_html=True)
    st.markdown("### 🎯 推奨トレーニング")
    
    # 最も改善が必要な項目
    diffs = {k: metrics[k] - ideal[k] for k in metrics.keys()}
    worst_metric = min(diffs, key=diffs.get)
    
    recommendations = {
        'stability': {
            'title': 'ピッチ安定性トレーニング',
            'methods': ['ロングトーン練習（15秒以上）', 'チューナーを使った音程確認', 'ビブラートコントロール練習'],
            'equipment': ['クリップ式チューナー', 'メトロノーム', 'ピッチ矯正アプリ']
        },
        'clarity': {
            'title': '発声クリアネス向上',
            'methods': ['リップロール', '声帯閉鎖訓練', '息の支え（丹田呼吸）'],
            'equipment': ['ボイストレーニングチューブ', 'ストロー', '加湿器']
        },
        'resonance': {
            'title': '共鳴トレーニング',
            'methods': ['鼻腔共鳴練習（ハミング）', '軟口蓋を上げる練習', '胸声と頭声のミックス'],
            'equipment': ['共鳴確認用録音機材', 'ボーカルマイク', '防音マット']
        },
        'brightness': {
            'title': 'トーンの明るさ改善',
            'methods': ['軟口蓋上げ練習', '高音域トレーニング', '笑顔での発声練習'],
            'equipment': ['ハイトーン用練習曲', 'EQツール', 'スペクトラムアナライザー']
        },
        'power': {
            'title': '音圧・パワーアップ',
            'methods': ['腹式呼吸マスター', 'ダイアフラムサポート', '発声のアタック強化'],
            'equipment': ['コンプレッサー（録音時）', 'ポップガード', '呼吸トレーニンググッズ']
        }
    }
    
    rec = recommendations[worst_metric]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📚 {rec['title']}")
        st.markdown("**推奨メソッド:**")
        for method in rec['methods']:
            st.markdown(f"- {method}")
    
    with col2:
        st.markdown("#### 🛠 おすすめ機材")
        for eq in rec['equipment']:
            st.markdown(f"- {eq}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ナビゲーション
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← 要点分析に戻る", key="back"):
            change_page('result1')
    
    with col3:
        if st.button("🏠 トップへ戻る", key="home2"):
            st.session_state.audio_data = None
            st.session_state.calibration_done = False
            change_page('home')
    
    # データエクスポート
    st.markdown('<div class="neu-card" style="margin-top:20px;">', unsafe_allow_html=True)
    st.markdown("### 💾 データ管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 レポートをPDFで保存", key="export_pdf"):
            st.info("PDF出力機能は今後実装予定です")
    
    with col2:
        if st.button("📈 履歴に追加（ローカル保存）", key="save_history"):
            st.info("履歴機能は今後実装予定です")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== メインアプリケーション ====================

def main():
    # ページルーティング
    page = st.session_state.page
    
    if page == 'home':
        page_home()
    elif page == 'calibration':
        page_calibration()
    elif page == 'input':
        page_input()
    elif page == 'loading':
        page_loading()
    elif page == 'result1':
        page_result1()
    elif page == 'result2':
        page_result2()

if __name__ == "__main__":
    main()
