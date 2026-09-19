"""전면 API(Realtime) 봇 진입점. 로컬 경로는 app/main.py — 그쪽은 건드리지 않는다.

실행: python -m app.main_realtime            (젯슨: ./run.sh 가 pipeline 을 보고 고른다)
      python -m app.main_realtime --no-wake  (호출어 없이 바로 대화 — 노트북 확인용)
설계: docs/superpowers/specs/2026-09-16-realtime-pipeline-design.md
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import threading
import time
from pathlib import Path

from .config import settings

log = logging.getLogger("jaeha_bot.main_realtime")


def check_startup(models: dict, env, *, wake: bool = True) -> str:
    """조용히 안 깨어나는 봇을 만들지 않는다 — 문제면 기동에서 멈춘다."""
    key = (env.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY 가 없다 — .env 를 확인할 것")
    wcfg = models.get("wake", {}) or {}
    if wake and wcfg.get("enabled", True):
        mode = ((wcfg.get("onnx") or {}).get("verify") or {}).get("mode", "whisper")
        if wcfg.get("detector", "stt") != "onnx" or mode != "embed":
            raise RuntimeError(
                "전면 API 에는 whisper 가 없다 — wake.detector: onnx, wake.onnx.verify.mode: embed "
                f"여야 한다(지금 detector={wcfg.get('detector')}, mode={mode})")
    return key


def _add_file_log() -> None:
    """로컬 봇(app/main.py)처럼 logs/jaeha_YYYYMMDD.log 에도 남긴다.

    🔴 2026-09-19 전면 API 봇은 터미널에만 찍어서, '호출어만 말했는데 뒷말로 본다'를
       [호출] 뒷말 줄로 사후 확인할 수 없었다.
    """
    from datetime import datetime
    d = Path(__file__).resolve().parent.parent / "logs"
    d.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(d / f"jaeha_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(h)


def build_instructions(music_on: bool) -> str:
    system = settings.prompts["system"]
    if music_on:
        from .music import adjust_prompt
        system = adjust_prompt(system)
    return system


def _build_music(ycfg: dict):
    """app/main.py 의 _build_music 과 같은 내용. 그 파일을 import 하면 STT·TTS 가 따라온다."""
    try:
        from .audio_player import AudioPlayer, default_library
        from .config import BASE_DIR
        from .music import MusicController
        from .youtube import YouTubePlayer, YouTubeSearch

        library = default_library()
        search = youtube = None
        key = os.environ.get(ycfg.get("key_env", "YOUTUBE_DATA_KEY"), "")
        if key:
            player = YouTubePlayer(volume=int(ycfg.get("volume", 80)))
            lack = player.missing()
            if lack:
                log.warning("노래: 준비물이 없어 유튜브를 끈다 %s", lack)
            else:
                youtube = player
                search = YouTubeSearch(
                    key, cache_path=BASE_DIR / "logs" / "youtube_cache.json",
                    max_duration_s=int(ycfg.get("max_duration_s", 600)),
                    cache_days=float(ycfg.get("cache_days", 7)))
        else:
            log.warning("노래: 유튜브 키가 없어 로컬 음원만")
        log.info("노래 틀기 켬 — 로컬 %d곡%s", len(library.playable("song")),
                 " + 유튜브" if youtube else "")
        return MusicController(library=library, local=AudioPlayer(library), search=search,
                               youtube=youtube,
                               default_query=str(ycfg.get("default_query", "동요")))
    except Exception as e:
        log.warning("노래 틀기 구성 실패(없이 계속): %s: %s", type(e).__name__, str(e)[:120])
        return None


class _MicThread:
    """AudioSource.read()(블로킹)를 스레드에서 돌려 큐로 넘긴다."""

    def __init__(self, source) -> None:
        self.source = source
        self.q: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="rt-mic")
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except Exception as e:
                log.warning("마이크 읽기 실패: %s", e)
                time.sleep(0.1)
                continue
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)


async def _bridge(mic: _MicThread, aq: asyncio.Queue, rate: int) -> None:
    loop = asyncio.get_running_loop()
    while True:
        frame = await loop.run_in_executor(None, mic.get)
        if frame is not None:
            await aq.put((frame, rate))


async def _awake_period(*, mic: _MicThread, rate: int, **conv_kwargs) -> str:
    from .realtime_conversation import Conversation

    run_kwargs = {k: conv_kwargs.pop(k) for k in ("preroll", "greet", "woke_during_music")}
    aq: asyncio.Queue = asyncio.Queue()
    bridge = asyncio.create_task(_bridge(mic, aq, rate))
    try:
        return await Conversation(mic=aq, **conv_kwargs).run(**run_kwargs)
    finally:
        bridge.cancel()
        await asyncio.gather(bridge, return_exceptions=True)


def main(argv=None) -> None:
    from .audio_device import setup_audio_device
    from .audio_source import AudioSource
    from .education_modes import GameManager
    from .metrics import MetricsLogger
    from .realtime_audio import StreamSpeaker
    from .realtime_protocol import RealtimeConfig
    from .realtime_session import RealtimeSession, synthesize
    from .realtime_turn import PHRASES
    from .voice_cache import VoiceCache
    from .wake import make_detector

    ap = argparse.ArgumentParser(description="재하봇 — 전면 API(Realtime)")
    ap.add_argument("--no-wake", action="store_true", help="호출어 없이 바로 대화(노트북 확인용)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _add_file_log()
    models = settings.models
    wcfg = models.get("wake", {}) or {}
    wake_on = bool(wcfg.get("enabled", True)) and not a.no_wake
    api_key = check_startup(models, os.environ, wake=wake_on)
    cfg = RealtimeConfig.from_dict(models.get("realtime"))
    log.info("재하봇(Realtime) 시작 — %s / 목소리 %s / 받아적기 %s",
             cfg.model, cfg.voice, cfg.transcribe_model)

    ycfg = models.get("youtube", {}) or {}
    music_on = bool(ycfg.get("enabled", False)) and wake_on
    if not setup_audio_device(output_device=ycfg.get("output_device", "respk") if music_on else None):
        log.warning("공유 출력 장치가 없어 노래 틀기를 끈다")
        music_on = False
    music = _build_music(ycfg) if music_on else None
    mcfg = models.get("metrics", {}) or {}
    metrics = MetricsLogger(enabled=mcfg.get("enabled", True), tag=f"{mcfg.get('tag', 'pc')}-rt")
    fcfg = models.get("filler", {}) or {}
    filler_phrases = list(fcfg.get("phrases", [])) if fcfg.get("enabled", True) else []

    cache = VoiceCache(cfg.voice_cache_dir, cfg.voice, cfg.model, cfg.speed)
    made = cache.ensure(list(PHRASES.values()) + filler_phrases,
                        lambda p: asyncio.run(synthesize(cfg, api_key, p)))
    log.info("고정 문구 캐시 준비(새로 만듦 %d)", made)

    vcfg = (wcfg.get("onnx") or {}).get("verify") or {}
    speaker = StreamSpeaker()
    source = AudioSource(preroll=float(wcfg.get("preroll", 0.5)),
                         verify_window=float(vcfg.get("window_s", 2.0)),
                         embed_window=float(vcfg.get("embed_window_s", 3.0))).open()
    detector = make_detector(wcfg, None, source) if wake_on else None
    metrics.after_load()
    common = dict(speaker=speaker, cache=cache, cfg=cfg, music=music,
                  games=GameManager(render=None), sleep_words=wcfg.get("sleep_words"),
                  instructions=build_instructions(music_on), history=[], metrics=metrics,
                  sleep_timeout=float(wcfg.get("sleep_timeout", 30)),
                  filler_phrases=filler_phrases)

    log.info("준비 완료 (종료: Ctrl+C)")
    if wake_on:
        ready = cache.get(PHRASES["ready"])
        if ready is not None:
            speaker.push(ready)
            while speaker.busy:
                time.sleep(0.05)
    try:
        while True:
            preroll, greet, woke_during_music = None, True, False
            if wake_on:
                source.drain()
                result = detector.wait_for_wake()
                if result is None:
                    continue
                if music is not None and music.pause_for_wake():
                    log.info("노래 중 호출 → 일시정지하고 듣는다")
                    woke_during_music = True
                # 🔴 2026-09-19 호출어 **뒤** 소리만 보낸다. 프리롤(호출어 포함)째 보내면
                #    서버가 '하이 티드' 를 아이 말로 받아 적고 답한다('하이즈들' 실기).
                if result.continued and result.tail.size:
                    preroll, greet = (result.tail, source.samplerate), False
            mic = _MicThread(source)
            try:
                reason = asyncio.run(_awake_period(
                    mic=mic, rate=source.samplerate, session=RealtimeSession(cfg, api_key),
                    preroll=preroll, greet=greet, woke_during_music=woke_during_music,
                    **common))
            finally:
                mic.stop()
            if reason == "sleep" and music is not None:
                music.stop()
            if not wake_on and reason in ("sleep", "idle", "lost"):
                break
            if detector is not None:
                detector.reset()
    except KeyboardInterrupt:
        log.info("종료 신호(Ctrl+C) 수신")
    finally:
        if music is not None:
            music.close()
        source.close()
        speaker.close()


if __name__ == "__main__":
    main()
