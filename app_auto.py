import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analyzer_engine import AudioAnalyzer
from src.transition_engine import TransitionEngine
from src.mix_renderer import MixRenderer
from src.youtube_downloader import download_playlist_batch, get_playlist_info
from src.utils import ensure_dirs, logger
import json
import zipfile

# Initialize
ensure_dirs()

# Set ffmpeg path for pydub and other tools
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
transition_engine = TransitionEngine()
renderer = MixRenderer()

# Load Preference Weights
def load_preference_weights():
    weights_file = "preference_weights.json"
    try:
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                data = json.load(f)
                type_weights = data.get('types', {})
                bar_weights_raw = data.get('bars', {})
                bar_weights = {}
                for k, v in bar_weights_raw.items():
                    try: bar_weights[int(k)] = v
                    except: bar_weights[k] = v
                return type_weights, bar_weights
    except: pass
    return {
        'crossfade': 0.5, 'bass_swap': 1.6, 'cut': 1.2,
        'filter_fade': 1.0, 'mashup': 1.0
    }, {4: 1.2, 8: 1.5}

def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except: return None

# --- Sorting Algorithms ---

# Camelot Wheel: maps (root_note, mode) → (number, letter)
# This enables proper harmonic compatibility checking
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
    """Convert key string like 'C# Minor' to Camelot (number, letter)."""
    try:
        parts = key_str.strip().split(' ')
        root = parts[0]
        mode = parts[1] if len(parts) > 1 else 'Major'
        return _CAMELOT.get((root, mode), (1, 'B'))
    except:
        return (1, 'B')

def get_key_distance(key1, key2):
    """Camelot wheel distance. 0 = perfect, 1 = compatible, 2+ = clash."""
    n1, l1 = _to_camelot(key1)
    n2, l2 = _to_camelot(key2)
    # Number distance on the wheel (1-12, circular)
    num_dist = min(abs(n1 - n2), 12 - abs(n1 - n2))
    if l1 == l2:
        return num_dist  # Same mode: 0=same key, 1=adjacent
    else:
        # Cross mode (A↔B): same number = relative major/minor (compatible)
        if num_dist == 0:
            return 0  # Relative major/minor = very compatible
        return num_dist + 1  # Cross-mode + number shift = less compatible

def get_avg_energy(track):
    """Get average energy from a track."""
    e = track.get('energy', 0.5)
    if isinstance(e, list):
        return sum(e) / len(e) if e else 0.5
    return float(e)

def dedup_tracks(tracks):
    """Remove duplicate tracks by filename. Keep first occurrence."""
    seen = set()
    result = []
    for t in tracks:
        # Use basename without extension as dedup key
        name = os.path.splitext(os.path.basename(t.get('filepath', t.get('filename', ''))))[0].lower().strip()
        if name and name not in seen:
            seen.add(name)
            result.append(t)
    removed = len(tracks) - len(result)
    if removed > 0:
        logger.info(f"Dedup: removed {removed} duplicate tracks")
    return result

def smart_sort_playlist(playlist):
    """
    Sort for optimal DJ flow using harmonic mixing:
    - Nearest-neighbor by Camelot key distance + BPM proximity
    - Key compatibility is king (Camelot wheel)
    - BPM jumps are minimized
    - Energy flow used as tiebreaker
    """
    if len(playlist) <= 2:
        return playlist
    
    for t in playlist:
        t['_avg_energy'] = get_avg_energy(t)
    
    # Start with the lowest BPM track (natural intro)
    start = min(playlist, key=lambda t: float(t['bpm']))
    
    sorted_list = [start]
    remaining = [t for t in playlist if t is not start]
    
    while remaining:
        current = sorted_list[-1]
        cur_bpm = float(current['bpm'])
        cur_key = current.get('key', 'C Major')
        cur_energy = current.get('_avg_energy', 0.5)
        
        best_score = -999
        best_track = None
        
        for track in remaining:
            t_bpm = float(track['bpm'])
            t_key = track.get('key', 'C Major')
            t_energy = track.get('_avg_energy', 0.5)
            
            # Key compatibility (Camelot) — dominant factor
            key_dist = get_key_distance(cur_key, t_key)
            if key_dist == 0:
                key_score = 100     # Same key / relative major-minor
            elif key_dist == 1:
                key_score = 80      # Adjacent on Camelot wheel
            elif key_dist == 2:
                key_score = 40      # 2 steps away
            else:
                key_score = max(0, 20 - key_dist * 8)  # Clash
            
            # BPM proximity — strong factor
            bpm_diff = abs(t_bpm - cur_bpm)
            if bpm_diff < 3:
                bpm_score = 50
            elif bpm_diff < 8:
                bpm_score = 35
            elif bpm_diff < 15:
                bpm_score = 15
            elif bpm_diff < 25:
                bpm_score = 0
            else:
                bpm_score = -30     # Big BPM jump penalty
            
            # Energy continuity — tiebreaker
            energy_diff = abs(t_energy - cur_energy)
            energy_score = max(0, 15 - energy_diff * 30)
            
            total = key_score + bpm_score + energy_score
            if total > best_score:
                best_score = total
                best_track = track
        
        sorted_list.append(best_track)
        remaining.remove(best_track)
    
    for t in sorted_list:
        t.pop('_avg_energy', None)
    
    return sorted_list

# --- Page Config ---
st.set_page_config(layout="wide", page_title="🎧 DJ Bot AutoMix")

st.title("🎧 DJ Bot — 완전 자동 믹스")
st.markdown("**YouTube 재생목록 URL 또는 파일 업로드 → 자동 분석 → 자동 믹싱**")

# Session State
for key, default in [
    ('playlist', []), ('transitions', []), ('candidates', []),
    ('final_mix_result', None), ('yt_tracks', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if 'type_weights' not in st.session_state:
    tw, bw = load_preference_weights()
    st.session_state['type_weights'] = tw
    st.session_state['bar_weights'] = bw

# --- Analyze tracks (works for both youtube and uploaded) ---
def analyze_tracks(filepaths_and_names):
    """Analyze a list of (filepath, display_name) tuples."""
    tracks = []
    total = len(filepaths_and_names)
    progress = st.progress(0)
    status = st.empty()
    t0 = time.time()
    
    for i, (fpath, name) in enumerate(filepaths_and_names):
        status.text(f"📊 분석 중 {i+1}/{total}: {name}")
        try:
            analysis = analyzer.analyze_track(fpath, {})
            analysis['filename'] = name
            analysis['filepath'] = fpath
            analysis['stems'] = {}
            tracks.append(analysis)
        except Exception as e:
            st.warning(f"⚠️ {name} 분석 실패: {e}")
        
        progress.progress((i + 1) / total)
    
    elapsed = time.time() - t0
    status.text(f"✅ 분석 완료! {len(tracks)}곡, {int(elapsed)}초")
    return tracks

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎵 입력 방식 선택")
    
    input_mode = st.radio("입력 방식", ["🔗 YouTube 재생목록 URL", "📁 파일 업로드"], label_visibility="collapsed")
    
    if input_mode == "🔗 YouTube 재생목록 URL":
        st.markdown("YouTube 또는 YouTube Music 재생목록 링크를 붙여넣으세요.")
        yt_url = st.text_input("재생목록 URL", placeholder="https://music.youtube.com/playlist?list=...")
        
        if yt_url and st.button("🚀 다운로드 & 자동 믹스", type="primary", use_container_width=True):
            with st.status("📥 YouTube에서 음악 다운로드 중...", expanded=True) as dl_status:
                status_text = st.empty()
                
                def update_status(text):
                    dl_status.write(text)
                
                # Download all tracks via Python API
                downloaded = download_playlist_batch(yt_url, progress_callback=update_status)
                
                if not downloaded:
                    st.error("❌ 다운로드된 곡이 없습니다. URL을 확인해주세요.")
                else:
                    dl_status.write(f"✅ {len(downloaded)}곡 다운로드 완료! 분석 시작...")
                    
                    # Dedup downloads (playlist may have repeats)
                    seen_titles = set()
                    unique = []
                    for d in downloaded:
                        key = d['title'].lower().strip()
                        if key not in seen_titles:
                            seen_titles.add(key)
                            unique.append(d)
                    if len(unique) < len(downloaded):
                        dl_status.write(f"🔄 중복 {len(downloaded)-len(unique)}곡 제거 → {len(unique)}곡")
                    
                    # Analyze
                    file_list = [(d['filepath'], d['title']) for d in unique]
                    tracks = analyze_tracks(file_list)
                    tracks = dedup_tracks(tracks)  # Extra dedup by filepath
                    
                    if len(tracks) >= 2:
                        st.session_state['playlist'] = tracks
                        st.session_state['candidates'] = []
                        st.session_state['final_mix_result'] = None
                        dl_status.update(label=f"✅ {len(tracks)}곡 준비 완료!", state="complete")
                        st.rerun()
                    else:
                        st.error("❌ 최소 2곡 이상 필요합니다.")
    
    else:
        uploaded_files = st.file_uploader("MP3/WAV 파일 선택", accept_multiple_files=True, type=['mp3', 'wav'])
        
        if uploaded_files and st.button("📊 분석 & 자동 믹스", type="primary", use_container_width=True):
            temp_dir = Path("cache/uploads")
            temp_dir.mkdir(exist_ok=True, parents=True)
            
            file_list = []
            for uf in uploaded_files:
                fpath = temp_dir / uf.name
                with open(fpath, "wb") as f:
                    f.write(uf.getbuffer())
                file_list.append((str(fpath), uf.name))
            
            tracks = analyze_tracks(file_list)
            if len(tracks) >= 2:
                st.session_state['playlist'] = tracks
                st.session_state['candidates'] = []
                st.session_state['final_mix_result'] = None
                st.rerun()
    
    st.divider()
    with st.expander("⚙️ Preference Weights"):
        st.json(st.session_state['type_weights'])

# --- MAIN AREA ---
playlist = st.session_state['playlist']

if not playlist:
    st.info("👈 좌측에서 YouTube 재생목록 URL을 입력하거나 파일을 업로드하세요.")
    st.stop()

# --- AUTO MIX PIPELINE ---
st.header(f"📋 {len(playlist)}곡 로드됨")

# Show track list
cols = st.columns([1, 5, 1, 1, 1])
cols[0].markdown("**#**"); cols[1].markdown("**제목**")
cols[2].markdown("**BPM**"); cols[3].markdown("**Key**"); cols[4].markdown("**길이**")

for i, t in enumerate(playlist):
    cols = st.columns([1, 5, 1, 1, 1])
    cols[0].write(f"{i+1}")
    cols[1].write(t['filename'][:60])
    cols[2].write(f"{float(t['bpm']):.0f}")
    cols[3].write(t.get('key', '?'))
    dur = float(t['duration'])
    cols[4].write(f"{int(dur//60)}:{int(dur%60):02d}")

st.divider()

# --- ONE-CLICK MIX ---
if not st.session_state['candidates']:
    if st.button("⚡ 원클릭 자동 믹스 생성", type="primary", use_container_width=True):
        with st.status("🧠 AI 자동 믹싱 진행 중...", expanded=True) as status:
            custom_weights = {
                'types': st.session_state['type_weights'],
                'bars': st.session_state['bar_weights']
            }
            
            # STEP 1: Smart Sort
            status.write("🎯 Step 1: Camelot Wheel 기반 하모닉 정렬...")
            sorted_playlist = smart_sort_playlist(playlist)
            st.session_state['playlist'] = sorted_playlist
            playlist = sorted_playlist
            
            # STEP 1.5: Set per-track play limits
            # For DJ mixes, each track should play ~60-90s (not full song)
            n_tracks = len(playlist)
            if n_tracks <= 5:
                max_play = 120  # More time per track for short playlists
            elif n_tracks <= 15:
                max_play = 80
            else:
                max_play = 60   # Tight for large playlists
            
            status.write(f"🎯 곡당 ~{max_play}초씩 플레이 (총 {n_tracks}곡)")
            
            # Assign play windows based on highlights
            for t in playlist:
                dur = float(t['duration'])
                highlights = t.get('highlights', [])
                
                if highlights and len(highlights) > 0:
                    # Use best highlight as center point
                    hl = highlights[0]
                    hl_start = hl.get('start', dur * 0.3)
                    hl_end = hl.get('end', min(hl_start + max_play, dur))
                    center = (hl_start + hl_end) / 2
                else:
                    # Default: use middle of the track
                    center = dur * 0.4
                
                # Set play window around center
                half = max_play / 2
                play_start = max(0, center - half)
                play_end = min(dur, center + half)
                
                # Ensure minimum play time
                if play_end - play_start < 30:
                    play_start = max(0, dur * 0.2)
                    play_end = min(dur, play_start + max_play)
                
                t['play_start'] = play_start
                t['play_end'] = play_end
            
            # STEP 2: Build simple crossfade transitions
            status.write("🎲 Step 2: 크로스페이드 계산...")
            total_pairs = len(playlist) - 1
            plan_selections = []
            
            for i in range(total_pairs):
                t_a = playlist[i]; t_b = playlist[i+1]
                
                # Crossfade = 8 bars at average BPM of the two tracks
                avg_bpm = (float(t_a['bpm']) + float(t_b['bpm'])) / 2
                bars = 8
                xfade_sec = bars * 4 * (60.0 / avg_bpm)  # 4 beats per bar
                xfade_sec = max(4, min(12, xfade_sec))    # Clamp 4-12 seconds
                
                trans = {
                    'type': 'crossfade',
                    'duration': round(xfade_sec, 1),
                    'a_out_time': t_a.get('play_end', float(t_a['duration'])),
                    'b_in_time': t_b.get('play_start', 0.0),
                    'preview_path': None,
                }
                plan_selections.append(trans)
                status.write(f"  ✓ {i+1}/{total_pairs}: {t_a['filename'][:30]} → {t_b['filename'][:30]} (xfade {xfade_sec:.1f}s)")
            
            # Calculate estimated total duration
            est_total = 0
            for i, t in enumerate(playlist):
                play_dur = t.get('play_end', float(t['duration'])) - t.get('play_start', 0)
                est_total += play_dur
            # Subtract crossfade overlaps
            for trans in plan_selections:
                est_total -= trans['duration']
            
            est_min = int(est_total // 60)
            est_sec = int(est_total % 60)
            status.write(f"⏱️ 예상 믹스 길이: **{est_min}분 {est_sec}초**")
            
            # STEP 3: Skip previews (fast), go straight to plan
            for i, best in enumerate(plan_selections):
                st.session_state['candidates'].append([best])
                st.session_state[f"trans_{i}"] = 0
            
            status.update(label=f"✅ 믹스 계획 완료! (예상 {est_min}분 {est_sec}초)", state="complete")
            st.rerun()

# --- SHOW PLAN & GENERATE ---
if st.session_state['candidates']:
    st.divider()
    st.header("🎵 믹스 계획")
    
    final_specs = [opts[0] for opts in st.session_state['candidates']]
    
    for i, spec in enumerate(final_specs):
        ta = playlist[i]; tb = playlist[i+1]
        icon = {"crossfade": "🔀", "bass_swap": "🔊", "cut": "✂️", "filter_fade": "🌊", "mashup": "🎚️"}.get(spec['type'], "🎵")
        
        col1, col2, col3 = st.columns([4, 2, 2])
        col1.markdown(f"{icon} **{ta['filename'][:35]}** → **{tb['filename'][:35]}**")
        col2.write(f"{spec['type']} ({spec.get('duration', 0):.0f}s)")
        
        # Play preview if available
        if spec.get('preview_path') and os.path.exists(spec['preview_path']):
            col3.audio(spec['preview_path'], format="audio/mp3")
    
    st.divider()
    
    col_gen, col_reset = st.columns([3, 1])
    
    if col_gen.button("🎧 최종 믹스 렌더링 (고품질 MP3)", key="gen_mix", type="primary", use_container_width=True):
        with st.spinner("🎛️ 렌더링 중... 곡 수에 따라 수 분이 소요됩니다."):
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"auto_mix_{timestamp}.mp3")
            
            try:
                result = renderer.render_final_mix(st.session_state['playlist'], final_specs, out_path)
                
                if isinstance(result, tuple):
                    mp3_gen, lrc_gen = result
                else:
                    mp3_gen = result
                    lrc_gen = result.replace(".mp3", ".lrc")
                
                zip_path = mp3_gen.replace(".mp3", ".zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(mp3_gen, os.path.basename(mp3_gen))
                    if os.path.exists(lrc_gen):
                        zipf.write(lrc_gen, os.path.basename(lrc_gen))
                
                st.session_state['final_mix_result'] = {
                    'mp3': mp3_gen, 'lrc': lrc_gen,
                    'timestamp': timestamp, 'zip': zip_path
                }
                st.success("✅ 믹스 렌더링 완료!")
                st.rerun()
            except Exception as e:
                st.error(f"렌더링 오류: {e}")
                logger.error(f"Render Error: {e}")
    
    if col_reset.button("🔄 다시 계획", use_container_width=True):
        st.session_state['candidates'] = []
        st.rerun()
    
    # Download UI
    if st.session_state['final_mix_result']:
        res = st.session_state['final_mix_result']
        st.divider()
        st.success(f"🎉 믹스 생성 완료! ({res['timestamp']})")
        
        if os.path.exists(res.get('mp3', '')):
            st.audio(res['mp3'], format="audio/mp3")
        
        if os.path.exists(res.get('zip', '')):
            with open(res['zip'], "rb") as f:
                st.download_button(
                    "📥 다운로드 (MP3 + LRC)", f,
                    file_name=os.path.basename(res['zip']),
                    mime="application/zip",
                    use_container_width=True
                )
