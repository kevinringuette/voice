# Voice Grader (Streamlit) — GitHub-ready
# See README.md for setup/deploy instructions.
import os, io, json, time, wave, queue, threading
from typing import Dict, List, Optional, Any
import numpy as np, pandas as pd, requests, streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.set_page_config(page_title="Voice Grader (Demo)", layout="wide")

OPENAI_MODEL_ASSESS = os.environ.get("OPENAI_ASSESS_MODEL", "gpt-4o-mini")
OPENAI_MODEL_STT     = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
N8N_ROSTER_URL       = os.environ.get("N8N_ROSTER_URL", "")
N8N_RUBRIC_URL       = os.environ.get("N8N_RUBRIC_URL", "")
N8N_SAVE_GRADE_URL   = os.environ.get("N8N_SAVE_GRADE_URL", "")
RTC_CONFIGURATION    = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

SAMPLE_ROSTER = ["Ava Li", "Jacob Coleman", "Mateo Ruiz", "Noah Kim"]
SAMPLE_RUBRIC = {"assignment": "Unit 4 Problem Set","categories":[
    {"name":"Correctness","max":10,"desc":"Final answers correct"},
    {"name":"Method","max":10,"desc":"Proper method/justification"},
    {"name":"Clarity","max":5,"desc":"Readable work / notation"},
    {"name":"Completeness","max":5,"desc":"All parts attempted"}]}

def _ensure_gradebook():
    ss = st.session_state
    for s in ss.roster:
        if s not in ss.gradebook:
            ss.gradebook[s] = {"scores": {c["name"]:0 for c in ss.rubric["categories"]}, "comments": ""}
        if s not in ss.locks:
            ss.locks[s] = set()

def _init_state():
    ss = st.session_state
    ss.setdefault("client", None)
    ss.setdefault("audio_queue", queue.Queue())
    ss.setdefault("transcript_live", "")
    ss.setdefault("current_student", None)
    ss.setdefault("roster", SAMPLE_ROSTER.copy())
    ss.setdefault("rubric", json.loads(json.dumps(SAMPLE_RUBRIC)))
    ss.setdefault("gradebook", {})
    ss.setdefault("locks", {})
    ss.setdefault("stop_event", threading.Event())
    ss.setdefault("worker_thread", None)
    ss.setdefault("vad_threshold", 0.01)
    ss.setdefault("silence_sec", 0.7)
    ss.setdefault("chunk_max_sec", 6.0)
    _ensure_gradebook()
_init_state()

def wav_bytes_from_float32_mono(samples: np.ndarray, sample_rate: int) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate); wf.writeframes(int16.tobytes())
    return buf.getvalue()

def rubric_max_for(cat_name: str) -> float:
    for c in st.session_state.rubric["categories"]:
        if c["name"] == cat_name: return float(c["max"])
    return 0.0

def total_for_student(student: str) -> float:
    row = st.session_state.gradebook.get(student, {}); scores = row.get("scores", {})
    return float(sum(scores.get(c["name"], 0.0) for c in st.session_state.rubric["categories"]))

def gradebook_to_dataframe() -> pd.DataFrame:
    cats = [c["name"] for c in st.session_state.rubric["categories"]]; rows = []
    for s in st.session_state.roster:
        data = st.session_state.gradebook[s]; row = {"Student": s}
        for c in cats: row[c] = data["scores"].get(c, 0)
        row["Total"] = total_for_student(s); row["Comments"] = data.get("comments",""); rows.append(row)
    return pd.DataFrame(rows)

def get_openai_client() -> Optional[OpenAI]:
    key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY","")
    if not key: return None
    try: return OpenAI(api_key=key)
    except Exception: return None

def transcribe_chunk_openai(wav_bytes: bytes) -> Optional[str]:
    client = st.session_state.client
    if client is None: return None
    try:
        resp = client.audio.transcriptions.create(
            model=OPENAI_MODEL_STT, file=("chunk.wav", io.BytesIO(wav_bytes), "audio/wav"), response_format="json")
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "to_dict"):
            d = resp.to_dict(); text = d.get("text")
        return (text or "").strip() or None
    except Exception as e:
        st.toast(f"STT error: {e}", icon="⚠️"); return None

def assess_chunk_with_llm(chunk_text: str) -> Optional[Dict[str, Any]]:
    client = st.session_state.client
    if client is None: return None
    roster = st.session_state.roster; rubric = st.session_state.rubric; current = st.session_state.current_student
    system = {"role":"system","content":(
        "You are a precise grading controller. Given a partial transcript chunk from a math teacher speaking aloud while grading, "
        "return STRICT JSON describing which student is being graded, which rubric fields to update, and any comments to append. "
        "Use EXACT rubric category names. If the teacher changes their mind ('actually full credit'), override prior values by sending absolute scores. "
        "Only set 'student' when the chunk clearly identifies a new student by name."
    )}
    user = {"role":"user","content": json.dumps({
        "roster": roster, "rubric": rubric, "current_student": current, "chunk": chunk_text,
        "schema": {
            "student": "null or string (name from roster) if a NEW student is announced; otherwise null",
            "updates": [{"category":"string","score":"number or null","score_delta":"number or null","comment":"optional string"}],
            "comment_append":"optional string", "finalize":"boolean"
        }
    }, ensure_ascii=False)}
    try:
        resp = client.chat.completions.create(model=OPENAI_MODEL_ASSESS, response_format={"type":"json_object"},
                                              messages=[system, user], temperature=0.0)
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        st.toast(f"LLM error: {e}", icon="⚠️"); return None

def apply_llm_actions(actions: Dict[str, Any]):
    if not actions: return
    student_field = actions.get("student")
    if student_field:
        name_l = student_field.strip().lower(); match = None
        for s in st.session_state.roster:
            sl = s.lower()
            if sl == name_l or name_l in sl or sl in name_l: match = s; break
        if match: st.session_state.current_student = match
    current = st.session_state.current_student
    if not current: return
    for upd in actions.get("updates", []) or []:
        cat = upd.get("category"); if not cat: continue
        if cat in st.session_state.locks[current]: continue
        max_pts = rubric_max_for(cat)
        if upd.get("score") is not None:
            v = max(0.0, min(float(upd["score"]), max_pts)); st.session_state.gradebook[current]["scores"][cat] = v
        elif upd.get("score_delta") is not None:
            old = float(st.session_state.gradebook[current]["scores"].get(cat, 0.0))
            v = max(0.0, min(old + float(upd["score_delta"]), max_pts)); st.session_state.gradebook[current]["scores"][cat] = v
        cmt = upd.get("comment")
        if cmt:
            prev = st.session_state.gradebook[current].get("comments","")
            st.session_state.gradebook[current]["comments"] = (prev + " " + cmt).strip()
    c_append = actions.get("comment_append")
    if c_append:
        prev = st.session_state.gradebook[current].get("comments","")
        st.session_state.gradebook[current]["comments"] = (prev + " " + c_append).strip()
    if actions.get("finalize"): st.session_state.current_student = None

def start_worker_if_needed():
    ss = st.session_state
    if ss.worker_thread and ss.worker_thread.is_alive(): return
    ss.stop_event.clear(); ss.worker_thread = threading.Thread(target=_audio_consumer_loop, daemon=True); ss.worker_thread.start()

def stop_worker():
    ss = st.session_state; ss.stop_event.set()
    try:
        while not ss.audio_queue.empty(): ss.audio_queue.get_nowait()
    except Exception: pass

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    try:
        pcm = frame.to_ndarray(); mono = pcm.mean(axis=0) if pcm.ndim==2 else pcm
        if mono.dtype == np.int16: mono = mono.astype(np.float32)/32768.0
        elif mono.dtype == np.int32: mono = mono.astype(np.float32)/2147483648.0
        else: mono = mono.astype(np.float32)
        st.session_state.audio_queue.put((mono, int(frame.sample_rate))); return frame
    except Exception: return frame

def _audio_consumer_loop():
    ss = st.session_state; seg = []; seg_sr = None; last_voice_t = time.time()
    def flush_segment():
        nonlocal seg, seg_sr
        if not seg or seg_sr is None: seg=[]; seg_sr=None; return
        samples = np.concatenate(seg, axis=0) if len(seg)>1 else seg[0]
        wav_b = wav_bytes_from_float32_mono(samples, seg_sr); seg=[]
        text = transcribe_chunk_openai(wav_b)
        if text:
            ss.transcript_live += ((" " if ss.transcript_live else "") + text)
            actions = assess_chunk_with_llm(text); apply_llm_actions(actions)
    while not ss.stop_event.is_set():
        try: mono, sr = ss.audio_queue.get(timeout=0.1)
        except queue.Empty:
            now = time.time()
            if seg and (now - last_voice_t) > ss.silence_sec: flush_segment()
            continue
        energy = float(np.sqrt(np.mean(np.square(mono)) + 1e-12)); voiced = energy > ss.vad_threshold
        if not seg: seg_sr = sr
        seg.append(mono); now = time.time()
        seg_len_sec = sum(len(x) for x in seg) / float(seg_sr or 16000)
        if voiced: last_voice_t = now
        if (now - last_voice_t) > ss.silence_sec or seg_len_sec >= ss.chunk_max_sec:
            flush_segment(); last_voice_t = now

st.title("🎙️ Voice Grader — hands‑free rubric filling (demo)")
with st.sidebar:
    st.subheader("Setup")
    k = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY",""))
    if k and k != os.environ.get("OPENAI_API_KEY",""): os.environ["OPENAI_API_KEY"] = k
    if st.session_state.client is None: st.session_state.client = get_openai_client()
    st.caption("STT model: " + OPENAI_MODEL_STT); st.caption("Analysis model: " + OPENAI_MODEL_ASSESS)
    with st.expander("Data sources (optional: n8n)"):
        roster_url = st.text_input("n8n roster URL", value=N8N_ROSTER_URL)
        rubric_url = st.text_input("n8n rubric URL", value=N8N_RUBRIC_URL)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load roster"):
                try:
                    r = requests.get(roster_url, timeout=8); r.raise_for_status()
                    data = r.json(); roster = data.get("students") or data.get("roster") or []
                    if roster:
                        st.session_state.roster = [s for s in roster if isinstance(s, str)]
                        _ensure_gradebook(); st.success(f"Loaded {len(roster)} students")
                    else: st.warning("No students found; keeping sample.")
                except Exception as e: st.warning(f"Roster load failed: {e}")
        with c2:
            if st.button("Load rubric"):
                try:
                    r = requests.get(rubric_url, timeout=8); r.raise_for_status()
                    data = r.json(); cats = data.get("categories") or []
                    if cats:
                        st.session_state.rubric = {"assignment": data.get("assignment","Assignment"),
                          "categories":[{"name":c["name"],"max":float(c.get("max",10)),"desc":c.get("desc","")} for c in cats if "name" in c]}
                        _ensure_gradebook(); st.success("Rubric loaded")
                    else: st.warning("No categories found; keeping sample.")
                except Exception as e: st.warning(f"Rubric load failed: {e}")
    st.divider(); st.subheader("Controls")
    st.slider("Silence to cut a chunk (sec)", 0.3, 2.0, st.session_state.silence_sec, 0.1, key="silence_sec")
    st.slider("VAD energy threshold", 0.001, 0.05, st.session_state.vad_threshold, 0.001, key="vad_threshold")
    st.slider("Max chunk length (sec)", 4.0, 15.0, st.session_state.chunk_max_sec, 0.5, key="chunk_max_sec")

cA, cB = st.columns([1,1])
with cA:
    st.markdown("#### Roster"); st.write(pd.DataFrame({"Student": st.session_state.roster}))
with cB:
    st.markdown(f"#### Rubric — *{st.session_state.rubric['assignment']}*")
    st.table(pd.DataFrame(st.session_state.rubric["categories"]))

st.markdown("---")
mc, tc = st.columns([1,2])
with mc:
    st.markdown("#### Microphone"); info = st.empty()
    webrtc_streamer(key="voice", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback, async_processing=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start processing", type="primary"):
            if st.session_state.client is None: st.error("Set OPENAI_API_KEY first.")
            else: start_worker_if_needed(); info.info("Listening… Say a student's name, then speak rubric feedback.")
    with c2:
        if st.button("⏹ Stop"):
            stop_worker(); info.warning("Stopped.")
    st.markdown("**Current student:**"); st.success(st.session_state.current_student or "— waiting for a name —")
with tc:
    st.markdown("#### Live transcript"); st.caption("Running transcript used for updates.")
    st.text_area("Transcript", value=st.session_state.transcript_live, height=180, key="transcript_view")

st.markdown("---")
left, right = st.columns([1.2, 1.8])
with left:
    st.markdown("### Current student • teacher can edit (overrides AI)")
    student = st.session_state.current_student or (st.session_state.roster[0] if st.session_state.roster else None)
    if student:
        cats = [c["name"] for c in st.session_state.rubric["categories"]]
        df = pd.DataFrame([{"Student": student, **{c: st.session_state.gradebook[student]["scores"].get(c, 0) for c in cats},
                            "Total": total_for_student(student),
                            "Comments": st.session_state.gradebook[student]["comments"]}])
        edited = st.data_editor(df, column_config={**{c: st.column_config.NumberColumn(c, min_value=0.0, max_value=rubric_max_for(c), step=1.0) for c in cats},
                           "Comments": st.column_config.TextColumn("Comments")}, key="editor_current", use_container_width=True, num_rows="fixed", hide_index=True)
        row0 = edited.iloc[0].to_dict()
        for c in cats:
            old = st.session_state.gradebook[student]["scores"].get(c, 0)
            if float(row0[c]) != float(old):
                st.session_state.gradebook[student]["scores"][c] = float(row0[c]); st.session_state.locks[student].add(c)
        if (row0["Comments"] or "") != (st.session_state.gradebook[student]["comments"] or ""):
            st.session_state.gradebook[student]["comments"] = row0["Comments"]
    else: st.info("Say the student's full name to begin (e.g., ‘Now grading Jacob Coleman…’).")
with right:
    st.markdown("### Gradebook (live)"); gb = gradebook_to_dataframe()
    st.dataframe(gb, use_container_width=True, hide_index=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("💾 Export CSV"):
            p = io.BytesIO(); gb.to_csv(p, index=False); st.download_button("Download grades.csv", p.getvalue(), file_name="grades.csv", mime="text/csv")
    with b2:
        if st.button("🧹 Reset transcript"): st.session_state.transcript_live = ""; st.rerun()
    with b3:
        if st.button("📤 Send to n8n webhook"):
            url = os.environ.get("N8N_SAVE_GRADE_URL", N8N_SAVE_GRADE_URL)
            if not url: st.error("Set N8N_SAVE_GRADE_URL env var.")
            else:
                payload = {"assignment": st.session_state.rubric.get("assignment","Assignment"),
                           "grades":[{"student": s, "scores": st.session_state.gradebook[s]["scores"],
                                      "total": total_for_student(s), "comments": st.session_state.gradebook[s]["comments"]}
                                     for s in st.session_state.roster]}
                try: r = requests.post(url, json=payload, timeout=10); r.raise_for_status(); st.success(f"Posted {len(gb)} rows to n8n → Airtable")
                except Exception as e: st.error(f"Save failed: {e}")
st.markdown("---"); st.caption("Mic via WebRTC • STT via low-latency model • JSON decisions • Teacher edits lock fields.")
