import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analyzer_engine import AudioAnalyzer
from src.stem_separator import StemSeparator
from src.transition_engine import TransitionEngine
from src.mix_renderer import MixRenderer
from src.youtube_downloader import download_playlist_batch
from src.utils import ensure_dirs, logger, get_file_hash
import json
import zipfile

# Initialize
ensure_dirs()

# Set ffmpeg path for pydub
try:
    import imageio_ffmpeg
    import shutil
    _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_src)
    _ffmpeg_exe = os.path.join(_ffmpeg_dir, 'ffmpeg.exe') if os.name == 'nt' else os.path.join(_ffmpeg_dir, 'ffmpeg')
    
    if not os.path.exists(_ffmpeg_exe):
        shutil.copy2(_ffmpeg_src, _ffmpeg_exe)
        
    os.environ['PATH'] = _ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_exe
except ImportError:
    import shutil
    ffmpeg_system = shutil.which("ffmpeg")
    if ffmpeg_system:
        from pydub import AudioSegment
        AudioSegment.converter = ffmpeg_system

analyzer = AudioAnalyzer()
separator = StemSeparator()
transition_engine = TransitionEngine()
renderer = MixRenderer()

# Load Preference Weights
def load_preference_weights():
    """Load learned weights from test_app.py training"""
    weights_file = "preference_weights.json"
    try:
        if os.path.exists(weights_file):
            import json
            with open(weights_file, 'r') as f:
                data = json.load(f)
                type_weights = data.get('types', {})
                bar_weights_raw = data.get('bars', {})
                
                # Convert string keys to integers for bar weights
                bar_weights = {}
                for k, v in bar_weights_raw.items():
                    try:
                        bar_weights[int(k)] = v
                    except (ValueError, TypeError):
                        bar_weights[k] = v
                
                return type_weights, bar_weights
    except Exception as e:
        st.warning(f"Could not load weights: {e}")
    
    # Default weights if file doesn't exist
    return {
        'crossfade': 0.5,
        'bass_swap': 1.6,
        'cut': 1.2,
        'filter_fade': 1.0,
        'mashup': 1.0
    }, {
        4: 1.2,
        8: 1.5
    }

st.set_page_config(layout="wide", page_title="Semi-Auto DJ Bot")

st.title("🎧 Semi-Auto Hip-Hop DJ Bot")
st.markdown("자동 분석과 세밀한 커스터마이징을 지원하는 반자동 믹스 모드입니다. **타임라인** 기반으로 곡의 길이와 트랜지션을 조절하세요.")

# Session State
if 'playlist' not in st.session_state:
    st.session_state['playlist'] = [] # List of analyzed track dicts
if 'transitions' not in st.session_state:
    st.session_state['transitions'] = [] # List of chosen specs
if 'candidates' not in st.session_state:
    st.session_state['candidates'] = [] # List of List of candidates per transition
if 'type_weights' not in st.session_state:
    type_w, bar_w = load_preference_weights()
    st.session_state['type_weights'] = type_w
    st.session_state['bar_weights'] = bar_w
if 'final_mix_result' not in st.session_state:
    st.session_state['final_mix_result'] = None

# --- Sorting & Auto-Planning Algorithms (Imported from Auto context) ---
_CAMELOT = {
    ('C', 'Major'): (8, 'B'), ('G', 'Major'): (9, 'B'), ('D', 'Major'): (10, 'B'),
    ('A', 'Major'): (11, 'B'), ('E', 'Major'): (12, 'B'), ('B', 'Major'): (1, 'B'),
    ('F#', 'Major'): (2, 'B'), ('C#', 'Major'): (3, 'B'), ('G#', 'Major'): (4, 'B'),
    ('D#', 'Major'): (5, 'B'), ('A#', 'Major'): (6, 'B'), ('F', 'Major'): (7, 'B'),
    ('A', 'Minor'): (8, 'A'), ('E', 'Minor'): (9, 'A'), ('B', 'Minor'): (10, 'A'),
    ('F#', 'Minor'): (11, 'A'), ('C#', 'Minor'): (12, 'A'), ('G#', 'Minor'): (1, 'A'),
    ('D#', 'Minor'): (2, 'A'), ('A#', 'Minor'): (3, 'A'), ('F', 'Minor'): (4, 'A'),
    ('C', 'Minor'): (5, 'A'), ('G', 'Minor'): (6, 'A'), ('D', 'Minor'): (7, 'A'),
}

def _to_camelot(key_str):
    try:
        parts = key_str.strip().split(' ')
        return _CAMELOT.get((parts[0], parts[1] if len(parts) > 1 else 'Major'), (1, 'B'))
    except: return (1, 'B')

def get_key_distance(key1, key2):
    n1, l1 = _to_camelot(key1)
    n2, l2 = _to_camelot(key2)
    num_dist = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return num_dist if l1 == l2 else (0 if num_dist == 0 else num_dist + 1)

def smart_sort_playlist(playlist):
    if len(playlist) <= 2: return playlist
    start = min(playlist, key=lambda t: float(t['bpm']))
    sorted_list = [start]
    remaining = [t for t in playlist if t is not start]
    while remaining:
        current = sorted_list[-1]
        cur_bpm, cur_key = float(current['bpm']), current.get('key', 'C Major')
        best_score, best_track = -999, None
        for track in remaining:
            t_bpm, t_key = float(track['bpm']), track.get('key', 'C Major')
            key_dist = get_key_distance(cur_key, t_key)
            key_score = 100 if key_dist == 0 else 80 if key_dist == 1 else 40 if key_dist == 2 else max(0, 20 - key_dist * 8)
            bpm_diff = abs(t_bpm - cur_bpm)
            bpm_score = 50 if bpm_diff < 3 else 35 if bpm_diff < 8 else 15 if bpm_diff < 15 else 0 if bpm_diff < 25 else -30
            total = key_score + bpm_score
            if total > best_score:
                best_score = total; best_track = track
        sorted_list.append(best_track)
        remaining.remove(best_track)
    return sorted_list

def dedup_tracks(tracks):
    seen = set()
    result = []
    for t in tracks:
        name = os.path.splitext(os.path.basename(t.get('filepath', t.get('filename', ''))))[0].lower().strip()
        if name and name not in seen:
            seen.add(name); result.append(t)
    return result

# Helper to load tracks from files
def load_tracks_from_files(uploaded_files):
    tracks = []
    seen_hashes = set()
    temp_dir = Path("cache/uploads")
    temp_dir.mkdir(exist_ok=True, parents=True)
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"📊 Analyzing {i+1}/{total_files}: {uploaded_file.name}")
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            file_hash = get_file_hash(str(file_path))
            if file_hash in seen_hashes: continue
            seen_hashes.add(file_hash)

            stems = {}
            analysis = analyzer.analyze_track(str(file_path), stems)
            analysis['filename'] = uploaded_file.name
            analysis['filepath'] = str(file_path)
            analysis['stems'] = stems
            tracks.append(analysis)
        except Exception as e:
            st.error(f"Error analyzing {uploaded_file.name}: {e}")
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✅ Analysis Complete! ({len(tracks)} tracks)")
    return tracks

# Helper to load tracks from YouTube
def load_tracks_from_youtube(yt_url):
    tracks = []
    with st.status("📥 YouTube에서 다운로드 및 분석 중...", expanded=True) as dl_status:
        status_text = st.empty()
        def update_status(text): dl_status.write(text)
        downloaded = download_playlist_batch(yt_url, progress_callback=update_status)
        if not downloaded:
            st.error("❌ 다운로드된 곡이 없습니다. URL을 확인해주세요.")
            return []
            
        seen_titles = set()
        unique = []
        for d in downloaded:
            key = d['title'].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(d)
                
        total = len(unique)
        for i, d in enumerate(unique):
            dl_status.write(f"📊 분석 중 {i+1}/{total}: {d['title']}")
            try:
                analysis = analyzer.analyze_track(d['filepath'], {})
                analysis['filename'] = d['title']
                analysis['filepath'] = d['filepath']
                analysis['stems'] = {}
                tracks.append(analysis)
            except Exception as e:
                st.warning(f"⚠️ {d['title']} 분석 실패: {e}")
                
        tracks = dedup_tracks(tracks)
        dl_status.update(label=f"✅ {len(tracks)}곡 완료!", state="complete")
        return tracks

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. 입력 방식 선택")
    
    input_mode = st.radio("입력 방식", ["🔗 YouTube 재생목록 URL", "📁 파일 업로드"], label_visibility="collapsed")
    
    if input_mode == "🔗 YouTube 재생목록 URL":
        yt_url = st.text_input("YouTube URL 입력", placeholder="https://youtube.com/playlist?list=...")
        if yt_url and st.button("🚀 다운로드 및 로드", type="primary", use_container_width=True):
            loaded = load_tracks_from_youtube(yt_url)
            if len(loaded) >= 2:
                st.session_state['playlist'] = loaded
                st.session_state['candidates'] = []
                st.session_state['final_mix_result'] = None
                st.rerun()
            else:
                st.error("❌ 너무 적은 곡이 다운로드 되었습니다.")
    else:
        uploaded_files = st.file_uploader("MP3/WAV 파일 선택", accept_multiple_files=True, type=['mp3', 'wav'])
        if uploaded_files and st.button("📊 업로드 및 분석", type="primary", use_container_width=True):
            loaded = load_tracks_from_files(uploaded_files)
            if len(loaded) >= 2:
                st.session_state['playlist'] = loaded
                st.session_state['candidates'] = []
                st.session_state['final_mix_result'] = None
                st.rerun()
    
    st.divider()
    
    # Weight Management
    with st.expander("⚙️ 트랜지션 선호도 설정", expanded=False):
        for t_type in ['crossfade', 'bass_swap', 'cut', 'filter_fade', 'mashup']:
            st.session_state['type_weights'][t_type] = st.slider(
                t_type, 0.1, 10.0, float(st.session_state['type_weights'].get(t_type, 1.0)), 0.1, key=f"weight_{t_type}"
            )
        col1, col2 = st.columns(2)
        if col1.button("💾 저장"):
            data = {'types': st.session_state['type_weights'], 'bars': st.session_state['bar_weights']}
            with open("preference_weights.json", 'w') as f:
                json.dump(data, f, indent=2)
            st.success("저장 완료")

# --- MAIN AREA ---
if not st.session_state['playlist']:
    st.info("👈 좌측에서 YouTube 재생목록 URL을 입력하거나 파일을 업로드하여 믹싱을 시작하세요.")
    st.stop()

st.header(f"2. 반자동 믹스 구성 ({len(st.session_state['playlist'])} 곡)")

# AUTO-PLANNER BUTTON
if not st.session_state.get('candidates'):
    if st.button("⚡ Smart Auto-Plan (초기 자동 구성)", type="primary", use_container_width=True):
        with st.status("🧠 AI 곡 정렬 및 트랜지션 계획 중...", expanded=True) as status:
            playlist = st.session_state['playlist']
            
            # Smart Sort
            status.write("🎯 Step 1: Harmonic 기반 스마트 곡 정렬...")
            st.session_state['playlist'] = smart_sort_playlist(playlist)
            playlist = st.session_state['playlist']
            
            # Apply initial timings (~60s per track based on highlights)
            for t in playlist:
                dur = float(t['duration'])
                hl = t.get('highlights', [])
                if len(dur > 90 and hl) > 0:
                    center = (hl[0].get('start', dur*0.3) + hl[0].get('end', dur*0.6)) / 2
                else:
                    center = dur * 0.4
                t['manual_in'] = max(0, center - 35)
                t['manual_out'] = min(dur, center + 35)
                
            # Generate Transitions
            status.write("🎲 Step 2: 최적 트랜지션 탐색 중...")
            cw = {'types': st.session_state['type_weights'], 'bars': st.session_state['bar_weights']}
            best_cands = []
            cur_entry_times = {0: 0.0}
            for i in range(len(playlist)-1):
                t_a = playlist[i]; t_b = playlist[i+1]
                t_a['play_end'] = t_a['manual_out']
                t_b['play_start'] = t_b['manual_in']
                opts = transition_engine.generate_random_candidates(t_a, t_b, count=6, weights=cw)
                best = transition_engine.select_best_candidate(opts, weights=cw, min_exit_time=cur_entry_times.get(i, 0))
                
                # Re-sort opts so 'best' is first
                opts.remove(best)
                opts.insert(0, best)
                
                cur_entry_times[i+1] = best['b_in_time']
                best_cands.append(opts)
            
            status.write("🎨 Step 3: 트랜지션 오디오 미리보기 렌더링...")
            for i, opts in enumerate(best_cands):
                if not opts[0].get('preview_path'):
                    opts[0]['preview_path'] = renderer.render_preview(playlist[i]['filepath'], playlist[i+1]['filepath'], opts[0])
                st.session_state[f"trans_{i}"] = 0
            
            st.session_state['candidates'] = best_cands
            status.update(label="✅ 초기 구성 완료! 타임라인에서 세부 조절하세요.", state="complete")
        st.rerun()

# TIMELINE UI
if st.session_state.get('candidates'):
    st.subheader("타임라인 (Timeline)")
    st.caption("각 곡의 재생 구간을 조절하거나 사이사이에 들어가는 트랜지션을 미리듣고 변경할 수 있습니다.")
    
    playlist = st.session_state['playlist']
    final_specs = []
    
    # Callback to handle transition re-rendering when In/Out sliders change
    def on_timing_change(track_idx):
        idx = track_idx - 1
        cw = {'types': st.session_state['type_weights'], 'bars': st.session_state['bar_weights']}
        if idx >= 0:
            playlist[idx]['play_end'] = playlist[idx].get('manual_out', playlist[idx]['duration'])
            opts = transition_engine.generate_random_candidates(playlist[idx], playlist[idx+1], count=6, weights=cw)
            opts[0]['preview_path'] = renderer.render_preview(playlist[idx]['filepath'], playlist[idx+1]['filepath'], opts[0])
            st.session_state['candidates'][idx] = opts
            st.session_state[f"trans_{idx}"] = 0
            
        if track_idx < len(playlist) - 1:
            playlist[track_idx]['play_start'] = playlist[track_idx].get('manual_in', 0)
            opts = transition_engine.generate_random_candidates(playlist[track_idx], playlist[track_idx+1], count=6, weights=cw)
            opts[0]['preview_path'] = renderer.render_preview(playlist[track_idx]['filepath'], playlist[track_idx+1]['filepath'], opts[0])
            st.session_state['candidates'][track_idx] = opts
            st.session_state[f"trans_{track_idx}"] = 0

    for i, track in enumerate(playlist):
        # 1. TRACK UI
        vol_info = f"{track.get('loudness_db', -99):.1f}dB"
        dur = float(track['duration'])
        
        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"**🎵 {i+1}. {track['filename'][:60]}**")
            col2.markdown(f"`{float(track['bpm']):.0f} BPM` | `{track.get('key', '?')}`")
            
            with st.expander(f"⚙️ 곡 재생 구간 조절 (현재: ~{int(track.get('manual_out', dur) - track.get('manual_in', 0))}초)"):
                m_in = st.slider("시작 (초)", 0.0, dur, track.get('manual_in', 0.0), key=f"in_{i}")
                m_out = st.slider("종료 (초)", 0.0, dur, track.get('manual_out', dur), key=f"out_{i}")
                if m_in != track.get('manual_in', 0.0) or m_out != track.get('manual_out', dur):
                    track['manual_in'] = m_in
                    track['play_start'] = m_in
                    track['manual_out'] = m_out
                    track['play_end'] = m_out
                    if st.button("🔄 이 곡 주변 트랜지션 다시 계산", key=f"recalc_{i}"):
                        on_timing_change(i)
                        st.rerun()

        # 2. TRANSITION UI (Between tracks)
        if i < len(playlist) - 1:
            opts = st.session_state['candidates'][i]
            # Ensure valid selection index
            sel_key = f"trans_{i}"
            if sel_key not in st.session_state or st.session_state[sel_key] >= len(opts):
                st.session_state[sel_key] = 0
            cur_idx = st.session_state[sel_key]
            active_trans = opts[cur_idx]
            
            final_specs.append(active_trans)
            
            icon = {"crossfade": "🔀", "bass_swap": "🔊", "cut": "✂️", "filter_fade": "🌊", "mashup": "🎚️"}.get(active_trans['type'], "🎛️")
            st.markdown(f"<div style='text-align: center; color: gray; margin: 10px 0;'> ⬇️ 연결: {icon} {active_trans['name']} ({active_trans.get('duration',0):.1f}s) ⬇️ </div>", unsafe_allow_html=True)
            
            with st.expander(f"🔀 트랜지션 변경: {active_trans['name']}"):
                col_play, col_opts = st.columns([1, 2])
                with col_play:
                    if active_trans.get('preview_path') and os.path.exists(active_trans['preview_path']):
                        st.audio(active_trans['preview_path'])
                    else:
                        if st.button("▶️ 미리듣기 렌더링", key=f"prev_{i}_{cur_idx}", use_container_width=True):
                            with st.spinner("미리보기 생성 중..."):
                                active_trans['preview_path'] = renderer.render_preview(playlist[i]['filepath'], playlist[i+1]['filepath'], active_trans)
                            st.rerun()
                
                with col_opts:
                    new_idx = st.radio(
                        "스타일 변경",
                        options=list(range(len(opts))),
                        format_func=lambda x: f"{opts[x]['name']} ({opts[x].get('duration',0):.1f}s)",
                        key=f"trans_tmp_{i}",
                        index=cur_idx,
                        horizontal=True
                    )
                    if new_idx != cur_idx:
                        st.session_state[sel_key] = new_idx
                        st.rerun()

    st.divider()
    
    # 3. GENERATION
    est_dur = sum([t.get('manual_out', float(t['duration'])) - t.get('manual_in', 0) for t in playlist])
    est_dur -= sum([s.get('duration', 0) for s in final_specs])
    st.subheader(f"3. 최종 렌더링 (예상 길이: {int(est_dur//60)}분 {int(est_dur%60)}초)")
    
    if st.button("🎧 최종 DJ 믹스 생성 (MP3 + 타임스탬프)", type="primary", use_container_width=True):
        with st.spinner("🎛️ 렌더링 중... 수 분이 소요될 수 있습니다."):
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"semi_auto_mix_{timestamp}.mp3")
            
            try:
                result = renderer.render_final_mix(playlist, final_specs, out_path)
                mp3_gen = result[0] if isinstance(result, tuple) else result
                lrc_gen = result[1] if isinstance(result, tuple) else result.replace(".mp3", ".lrc")
                
                zip_path = mp3_gen.replace(".mp3", ".zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(mp3_gen, os.path.basename(mp3_gen))
                    if os.path.exists(lrc_gen): zipf.write(lrc_gen, os.path.basename(lrc_gen))
                
                st.session_state['final_mix_result'] = {'mp3': mp3_gen, 'lrc': lrc_gen, 'timestamp': timestamp, 'zip': zip_path}
                st.success("✅ 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"Render Error: {e}")
                
    if st.session_state['final_mix_result']:
        res = st.session_state['final_mix_result']
        st.success(f"🎉 믹스 준비 완료! ({res['timestamp']})")
        if os.path.exists(res.get('mp3', '')): st.audio(res['mp3'])
        if os.path.exists(res.get('zip', '')):
            with open(res['zip'], "rb") as f:
                st.download_button("📥 다운로드 (MP3 + 가사연동 파일)", f, file_name=os.path.basename(res['zip']), mime="application/zip", use_container_width=True)

