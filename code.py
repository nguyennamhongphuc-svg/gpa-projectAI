# =========================================================
# AI GPA PREDICTOR - STREAMLIT APP
# Author: ChatGPT
# Mô tả:
# Ứng dụng dự đoán GPA sinh viên bằng Machine Learning
# =========================================================

# =========================================================
# IMPORT THƯ VIỆN
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =========================================================
# CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="AI GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CSS GIAO DIỆN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #071421;
    color: white;
}

h1, h2, h3 {
    color: #FFD700;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F33, #102A43);
}

.stMetric {
    background-color: #102A43;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #00D4FF;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TIÊU ĐỀ
# =========================================================

st.title("🎓 AI GPA Predictor Dashboard")
st.markdown("""
### Dự đoán GPA sinh viên bằng Machine Learning
""")

# =========================================================
# ĐỌC DỮ LIỆU CSV
# =========================================================

try:
    df = pd.read_csv("Dataset_GPA_Thuc_Te_250_Responses.csv")

except FileNotFoundError:
    st.error("❌ Không tìm thấy file Dataset_GPA_Thuc_Te_250_Responses.csv")
    st.stop()

# =========================================================
# XỬ LÝ DỮ LIỆU
# =========================================================

# Copy dataset
data = df.copy()

# Encode dữ liệu phân loại
data["Làm_Thêm"] = data["Làm_Thêm"].map({
    "Có": 1,
    "Không": 0
})

data["Tham_Gia_CLB"] = data["Tham_Gia_CLB"].map({
    "Có": 1,
    "Không": 0
})

data["Hình_Thức_Học"] = data["Hình_Thức_Học"].map({
    "Học nhóm": 1,
    "Tự học": 0
})

# =========================================================
# FEATURES & TARGET
# =========================================================

X = data.drop([
    "GPA",
    "Mã_Sinh_Viên"
], axis=1)

y = data["GPA"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# HUẤN LUYỆN MODEL RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# DỰ ĐOÁN TEST
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# ĐỘ CHÍNH XÁC
# =========================================================

r2 = r2_score(y_test, y_pred)

# =========================================================
# SIDEBAR - INPUT
# =========================================================

st.sidebar.title("⚙️ Nhập thông tin sinh viên")

study_hours = st.sidebar.slider(
    "📚 Số giờ học mỗi tuần",
    0,
    50,
    20
)

subjects = st.sidebar.slider(
    "📖 Số môn đang học",
    1,
    10,
    5
)

part_time = st.sidebar.selectbox(
    "💼 Có làm thêm không?",
    ["Có", "Không"]
)

sleep_time = st.sidebar.slider(
    "😴 Thời gian ngủ mỗi đêm",
    4,
    12,
    7
)

club = st.sidebar.selectbox(
    "🎯 Có tham gia CLB?",
    ["Có", "Không"]
)

attendance = st.sidebar.slider(
    "🏫 Tỉ lệ đi học (%)",
    0,
    100,
    80
)

study_method = st.sidebar.selectbox(
    "🧠 Hình thức học",
    ["Tự học", "Học nhóm"]
)

social_media = st.sidebar.slider(
    "📱 Thời gian dùng MXH",
    0,
    10,
    3
)

predict_button = st.sidebar.button("🚀 Dự đoán GPA")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Dự đoán kết quả",
    "📊 Phân tích dữ liệu (EDA)",
    "🕸️ So sánh"
])

# =========================================================
# TAB 1 - DỰ ĐOÁN
# =========================================================

with tab1:

    st.subheader("🎯 Kết quả dự đoán GPA")

    if predict_button:

        # Encode input
        part_time_encoded = 1 if part_time == "Có" else 0

        club_encoded = 1 if club == "Có" else 0

        study_method_encoded = (
            1 if study_method == "Học nhóm"
            else 0
        )

        # DataFrame input
        input_data = pd.DataFrame({
            "Số_Giờ_Học_Tuần": [study_hours],
            "Số_Môn_Đang_Học": [subjects],
            "Làm_Thêm": [part_time_encoded],
            "Thời_Gian_Ngủ": [sleep_time],
            "Tham_Gia_CLB": [club_encoded],
            "Điểm_Danh_%": [attendance],
            "Hình_Thức_Học": [study_method_encoded],
            "Thời_Gian_Mạng_Xã_Hội": [social_media]
        })

        # Predict
        predicted_gpa = model.predict(input_data)[0]

        predicted_gpa = round(predicted_gpa, 2)

        # =================================================
        # HIỂN THỊ GPA
        # =================================================

        st.metric(
            label="📌 GPA Dự Đoán",
            value=f"{predicted_gpa}/4.0"
        )

        # =================================================
        # TƯ VẤN
        # =================================================

        st.subheader("💡 Tư vấn học tập")

        if predicted_gpa < 2.5:

            st.error("""
⚠️ GPA của bạn đang ở mức thấp.

Bạn nên:
- Tăng thời gian học
- Giảm mạng xã hội
- Đi học đầy đủ hơn
- Ngủ đủ giấc
            """)

        elif predicted_gpa < 3.2:

            st.warning("""
📚 GPA ở mức khá.

Bạn có thể cải thiện bằng cách:
- Tăng giờ tự học
- Giảm xao nhãng
- Tập trung hơn vào việc học
            """)

        else:

            st.success("""
🏆 GPA rất tốt!

Bạn đang có thói quen học tập hiệu quả.
Hãy tiếp tục duy trì phong độ này!
            """)

            # Hiệu ứng bóng bay
            st.balloons()

        # =================================================
        # THÔNG TIN MODEL
        # =================================================

        st.info(f"""
🤖 Thuật toán sử dụng: Random Forest Regressor

🎯 Độ chính xác R²: {round(r2, 3)}
        """)

    else:
        st.warning("⬅️ Hãy nhập thông tin ở Sidebar để bắt đầu.")

# =========================================================
# TAB 2 - EDA
# =========================================================

with tab2:

    st.subheader("📊 Phân tích dữ liệu")

    # =====================================================
    # SCATTER PLOT
    # =====================================================

    st.markdown("### 📚 Tương quan giữa giờ học và GPA")

    scatter_fig = px.scatter(
        df,
        x="Số_Giờ_Học_Tuần",
        y="GPA",
        color="GPA",
        template="plotly_dark"
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.markdown("### 🧠 Mức độ ảnh hưởng của các yếu tố")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark",
        color="Importance"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

# =========================================================
# TAB 3 - RADAR CHART
# =========================================================

with tab3:

    st.subheader("🕸️ So sánh với sinh viên trung bình")

    # =====================================================
    # GIÁ TRỊ USER
    # =====================================================

    user_values = [
        study_hours,
        attendance,
        sleep_time * 10,
        social_media * 10
    ]

    # =====================================================
    # GIÁ TRỊ TRUNG BÌNH
    # =====================================================

    avg_values = [
        df["Số_Giờ_Học_Tuần"].mean(),
        df["Điểm_Danh_%"].mean(),
        df["Thời_Gian_Ngủ"].mean() * 10,
        df["Thời_Gian_Mạng_Xã_Hội"].mean() * 10
    ]

    categories = [
        "📚 Giờ học",
        "🏫 Attendance",
        "😴 Ngủ",
        "📱 MXH"
    ]

    # =====================================================
    # RADAR CHART
    # =====================================================

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Bạn'
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=categories,
        fill='toself',
        name='Trung bình'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=True,
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info("""
📌 Radar Chart giúp bạn so sánh thói quen học tập hiện tại
với mức trung bình của 250 sinh viên trong dataset.
    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
© 2026 AI GPA Predictor Dashboard
| Machine Learning + Streamlit + Plotly
""")
