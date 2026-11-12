# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os

from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Career Mock Interview",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "test_submitted" not in st.session_state:
    st.session_state.test_submitted = False

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

# =====================================================
# RECORDINGS FOLDER
# =====================================================

os.makedirs(
    "recordings",
    exist_ok=True
)

# =====================================================
# DATASET PATH
# =====================================================

DATA_PATH = r"C:\Users\shenbagam\videos\history\master_dataset_cleaned.csv"

# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig"
    )

df = load_dataset()

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def preprocess(text):

    text = str(text).lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    return text


# =====================================================
# KEYWORD SCORING
# =====================================================

def keyword_score(
        user_answer,
        expected_keywords
):

    answer = str(
        user_answer
    ).lower()

    keywords = [
        x.strip().lower()
        for x in str(
            expected_keywords
        ).split(",")
        if x.strip()
    ]

    if len(keywords) == 0:
        return 0

    matched = sum(
        1
        for k in keywords
        if k in answer
    )

    return matched / len(keywords)

# =====================================================
# BANNER
# =====================================================

try:

    banner = Image.open(
        "banner_image.png"
    )

    st.image(
        banner,
        use_container_width=True
    )

except:
    pass

# =====================================================
# TITLE
# =====================================================

st.title(
    "🎯 AI Mock Interview Coach"
)

# =====================================================
# LIVE CAMERA + MICROPHONE
# =====================================================

st.subheader(
    "🎥 Live Camera & Microphone"
)

st.info(
    "Allow camera and microphone permissions."
)

try:

    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode
    )

    webrtc_ctx = webrtc_streamer(
        key="interview_stream",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": True
        },
        async_processing=True
    )

    if webrtc_ctx.state.playing:

        st.success(
            "Camera & Microphone Active"
        )

    else:

        st.warning(
            "Click START and allow permissions."
        )

except ImportError:

    st.error(
        "Install using:\n\npip install streamlit-webrtc av"
    )

except Exception as e:

    st.error(
        f"Error: {e}"
    )


# =====================================================
# SCREEN RECORDING
# =====================================================

st.subheader(
    "🖥 Screen Recording"
)

screen_recorder = """
<div>

<button id="start">
Start Screen Recording
</button>

<button id="stop">
Stop Recording
</button>

<p id="status"></p>

<script>

let mediaRecorder;
let recordedChunks = [];

document.getElementById('start')
.onclick = async ()=>{

    const stream =
    await navigator.mediaDevices.getDisplayMedia({
        video:true,
        audio:true
    });

    mediaRecorder =
    new MediaRecorder(stream);

    recordedChunks = [];

    mediaRecorder.ondataavailable =
    (event)=>{

        if(event.data.size > 0){
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.start();

    document.getElementById(
    'status'
    ).innerText='Recording...';
};

document.getElementById('stop')
.onclick = ()=>{

    mediaRecorder.stop();

    mediaRecorder.onstop = ()=>{

        const blob =
        new Blob(recordedChunks,{
        type:'video/webm'
        });

        const url =
        URL.createObjectURL(blob);

        const a =
        document.createElement('a');

        a.href=url;

        a.download=
        'interview_recording.webm';

        a.click();

        document.getElementById(
        'status'
        ).innerText =
        'Downloaded';
    };
};

</script>

</div>
"""

components.html(
    screen_recorder,
    height=200
)


# =====================================================
# CAREER SELECTION
# =====================================================

career_types = sorted(
    df["career_type"]
    .dropna()
    .unique()
)

selected_career = st.selectbox(
    "Select Career Path",
    career_types
)

career_df = df[
    df["career_type"]
    ==
    selected_career
]

# =====================================================
# SIDEBAR SETTINGS
# =====================================================

st.sidebar.header(
    "Interview Settings"
)

difficulty_options = ["All"]

if "difficulty" in career_df.columns:

    difficulty_options.extend(
        sorted(
            career_df[
                "difficulty"
            ]
            .dropna()
            .unique()
        )
    )

selected_difficulty = st.sidebar.selectbox(
    "Difficulty",
    difficulty_options
)

subject_options = ["All"]

subject_options.extend(
    sorted(
        career_df[
            "subject"
        ]
        .dropna()
        .unique()
    )
)

selected_subject = st.sidebar.selectbox(
    "Subject",
    subject_options
)

num_questions = st.sidebar.slider(
    "Number of Questions",
    5,
    20,
    10
)


# =====================================================
# FILTERS
# =====================================================

filtered_df = career_df.copy()

if selected_subject != "All":

    filtered_df = filtered_df[
        filtered_df["subject"]
        ==
        selected_subject
    ]

if (
    selected_difficulty != "All"
    and
    "difficulty" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["difficulty"]
        ==
        selected_difficulty
    ]

# =====================================================
# START INTERVIEW
# =====================================================

if st.button(
    "🚀 Start Interview"
):

    st.session_state.interview_started = True

    st.session_state.test_submitted = False

    st.session_state.questions = (
        filtered_df.sample(
            n=min(
                num_questions,
                len(filtered_df)
            ),
            random_state=42
        )
        .reset_index(
            drop=True
        )
    )


# =====================================================
# DISPLAY QUESTIONS
# =====================================================

if st.session_state.interview_started:

    questions_df = (
        st.session_state.questions
    )

    st.header(
        "🎯 Interview Questions"
    )

    answers = {}

    for idx, row in questions_df.iterrows():

        st.markdown("---")

        st.write(
            f"### Question {idx+1}"
        )

        st.write(
            row["question"]
        )

        st.caption(
            f"Subject: {row['subject']}"
        )

        answers[idx] = st.text_area(

            "Your Answer",

            key=f"answer_{idx}",

            height=120
        )

    # =================================================
    # PROGRESS BAR
    # =================================================

    answered_count = sum(

        1

        for ans in answers.values()

        if ans.strip()
    )

    total_questions = len(
        questions_df
    )

    progress = (
        answered_count
        /
        total_questions
    )

    st.progress(
        progress
    )

    st.write(

        f"Answered "

        f"{answered_count}"

        f" of "

        f"{total_questions}"
    )

    # =================================================
    # SHORT ANSWER WARNINGS
    # =================================================

    for idx in range(total_questions):

        ans = answers[idx]

        if (
            len(ans.split()) < 5
            and
            ans.strip()
        ):

            st.warning(
                f"Question {idx+1}: "
                "Answer looks too short."
            )

    # =================================================
    # EVALUATE ANSWERS
    # =================================================

    evaluate_btn = st.button(
        "✅ Evaluate Answers"
    )

    if evaluate_btn:

        st.session_state.test_submitted = True

        st.session_state.answers = answers

        st.session_state.questions_df = questions_df
# =====================================================
# EVALUATION ENGINE
# =====================================================

if st.session_state.test_submitted:

    questions_df = (
        st.session_state.questions_df
    )

    answers = (
        st.session_state.answers
    )

    st.markdown("---")

    st.header(
        "📊 Interview Evaluation Report"
    )

    scores = []

    feedbacks = []

    strengths = []

    weaknesses = []

    subject_scores = {}

    # ===============================================
    # QUESTION LEVEL EVALUATION
    # ===============================================

    for idx, row in questions_df.iterrows():

        expected_answer = str(
            row["answers"]
        )

        expected_keywords = str(
            row["expected_keywords"]
        )

        subject = str(
            row["subject"]
        )

        user_answer = str(
            answers[idx]
        )

        if user_answer.strip() == "":

            final_score = 0

            feedback = (
                "❌ No answer provided."
            )

        else:

            expected = preprocess(
                expected_answer
            )

            actual = preprocess(
                user_answer
            )

            try:

                tfidf_matrix = (

                    TfidfVectorizer()

                    .fit_transform(

                        [
                            expected,
                            actual
                        ]
                    )
                )

                similarity = (

                    cosine_similarity(

                        tfidf_matrix[0:1],

                        tfidf_matrix[1:2]

                    )[0][0]
                )

            except:

                similarity = 0

            # =======================================
            # KEYWORD SCORE
            # =======================================

            k_score = keyword_score(

                user_answer,

                expected_keywords
            )

            # =======================================
            # FINAL HYBRID SCORE
            # =======================================

            final_score = (

                similarity * 0.7

                +

                k_score * 0.3
            )

            final_score = round(
                final_score * 10,
                2
            )

            if final_score >= 8:

                feedback = (
                    "✅ Excellent Answer"
                )

            elif final_score >= 6:

                feedback = (
                    "👍 Good Answer. Add more depth."
                )

            elif final_score >= 4:

                feedback = (
                    "⚠ Average Answer. Improve explanations."
                )

            else:

                feedback = (
                    "❌ Needs Improvement."
                )

        scores.append(
            final_score
        )

        feedbacks.append(
            feedback
        )

        if subject not in subject_scores:

            subject_scores[subject] = []

        subject_scores[subject].append(
            final_score
        )

    
    # ===============================================
    # OVERALL SCORE
    # ===============================================

    overall_score = round(
        np.mean(scores),
        2
    )

    readiness = round(

        (overall_score / 10) * 100,

        2
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Overall Score",
            f"{overall_score}/10"
        )

    with col2:

        st.metric(
            "Interview Readiness",
            f"{readiness}%"
        )


    # ===============================================
    # SUBJECT WISE ANALYSIS
    # ===============================================

    st.subheader(
        "📚 Subject Wise Performance"
    )

    subject_names = []

    subject_averages = []

    for subject, values in subject_scores.items():

        avg = round(

            np.mean(values),

            2
        )

        subject_names.append(
            subject
        )

        subject_averages.append(
            avg
        )

        st.write(
            f"✅ {subject}: {avg}/10"
        )

        if avg >= 8:

            strengths.append(
                subject
            )

        elif avg < 5:

            weaknesses.append(
                subject
            )

    # ===============================================
    # STRENGTHS
    # ===============================================

    st.subheader(
        "🏆 Strength Areas"
    )

    if strengths:

        for item in sorted(
            list(
                set(
                    strengths
                )
            )
        ):

            st.success(item)

    else:

        st.info(
            "No strong areas identified yet."
        )

    # ===============================================
    # WEAKNESSES
    # ===============================================

    st.subheader(
        "📌 Need Improvement"
    )

    if weaknesses:

        for item in sorted(
            list(
                set(
                    weaknesses
                )
            )
        ):

            st.error(item)

    else:

        st.success(
            "No weak areas identified."
        )

    # ===============================================
    # BAR CHART
    # ===============================================

    st.subheader(
        "📊 Question Wise Scores"
    )

    fig_bar, ax_bar = plt.subplots(
        figsize=(10, 4)
    )

    ax_bar.bar(

        range(
            1,
            len(scores)+1
        ),

        scores,

        color="skyblue"
    )

    ax_bar.set_title(
        "Question Wise Performance"
    )

    ax_bar.set_xlabel(
        "Question Number"
    )

    ax_bar.set_ylabel(
        "Score"
    )

    ax_bar.set_ylim(
        0,
        10
    )

    st.pyplot(
        fig_bar
    )

    # ===============================================
    # SUBJECT PERFORMANCE CHART
    # ===============================================

    st.subheader(
        "📈 Subject Performance"
    )

    fig_subject, ax_subject = plt.subplots(
        figsize=(8, 4)
    )

    ax_subject.bar(

        subject_names,

        subject_averages,

        color="green"
    )

    ax_subject.set_ylim(
        0,
        10
    )

    ax_subject.set_ylabel(
        "Average Score"
    )

    ax_subject.tick_params(
        axis="x",
        rotation=45
    )

    st.pyplot(
        fig_subject
    )
    # ===============================================
    # PERFORMANCE DISTRIBUTION PIE CHART
    # ===============================================

    st.subheader(
        "🥧 Performance Distribution"
    )

    excellent_count = sum(
        score >= 8
        for score in scores
    )

    good_count = sum(
        5 <= score < 8
        for score in scores
    )

    improvement_count = sum(
        score < 5
        for score in scores
    )

    fig_pie, ax_pie = plt.subplots()

    ax_pie.pie(

        [
            excellent_count,
            good_count,
            improvement_count
        ],

        labels=[
            "Excellent",
            "Good",
            "Needs Improvement"
        ],

        autopct="%1.1f%%",

        colors=[
            "green",
            "orange",
            "red"
        ],

        startangle=120
    )

    st.pyplot(
        fig_pie
    )

    

    